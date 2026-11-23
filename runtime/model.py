"""Restormer v1.0 topology, native single-channel 256 I/O.

Adapted from swz30/Restormer (pinned original and license in vendor/).
Input/output adaptation, learned input bias, sigmoid and latent dropout differ.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from common import PUBLIC_WEIGHTS, sha256

CONFIG = dict(dim=48, num_blocks=[4, 6, 6, 8], heads=[1, 2, 4, 8],
              num_refinement_blocks=4, ffn_expansion_factor=2.66,
              latent_dropout=0.05)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = BiasFreeLayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        return self.body(x.permute(0, 2, 3, 1).reshape(b, h*w, c)).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class FeedForward(nn.Module):
    def __init__(self, channels, expansion):
        super().__init__()
        hidden = int(channels*expansion)
        self.project_in = nn.Conv2d(channels, hidden*2, 1, bias=False)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, padding=1, groups=hidden*2, bias=False)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        a, b = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(a)*b)


class Attention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels*3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels*3, channels*3, 3, padding=1,
                                    groups=channels*3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = [value.reshape(b, self.num_heads, c//self.num_heads, h*w)
                   for value in self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)]
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        weights = (q@k.transpose(-2, -1)*self.temperature).softmax(dim=-1)
        return self.project_out((weights@v).reshape(b, c, h, w))


class TransformerBlock(nn.Module):
    def __init__(self, channels, heads, expansion, dropout=0.0):
        super().__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = Attention(channels, heads)
        self.norm2 = LayerNorm(channels)
        self.ffn = FeedForward(channels, expansion)
        self.dropout = float(dropout)

    def forward(self, x):
        x = x+self.attn(self.norm1(x))
        branch = self.ffn(self.norm2(x))
        return x+F.dropout2d(branch, p=self.dropout, training=self.training)


class PatchEmbed(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        # Ordinary Conv2d bias, with its own RNG so existing initial weights stay fixed.
        generator = torch.Generator(device='cpu').manual_seed(2026)
        self.proj.bias = nn.Parameter(torch.empty(channels).uniform_(-1/3, 1/3, generator=generator))

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels//2, 3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels*2, 3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(self, config=None, activation_checkpointing=True):
        super().__init__()
        cfg = dict(CONFIG if config is None else config)
        d, blocks, heads = cfg['dim'], cfg['num_blocks'], cfg['heads']
        expansion = cfg['ffn_expansion_factor']
        self.activation_checkpointing = activation_checkpointing
        def sequence(channels, count, head, dropout=0.0):
            return nn.Sequential(*[TransformerBlock(channels, head, expansion, dropout)
                                   for _ in range(count)])
        self.patch_embed = PatchEmbed(d)
        self.encoder_level1 = sequence(d, blocks[0], heads[0])
        self.down1_2 = Downsample(d)
        self.encoder_level2 = sequence(d*2, blocks[1], heads[1])
        self.down2_3 = Downsample(d*2)
        self.encoder_level3 = sequence(d*4, blocks[2], heads[2])
        self.down3_4 = Downsample(d*4)
        self.latent = sequence(d*8, blocks[3], heads[3], cfg['latent_dropout'])
        self.up4_3 = Upsample(d*8)
        self.reduce_chan_level3 = nn.Conv2d(d*8, d*4, 1, bias=False)
        self.decoder_level3 = sequence(d*4, blocks[2], heads[2])
        self.up3_2 = Upsample(d*4)
        self.reduce_chan_level2 = nn.Conv2d(d*4, d*2, 1, bias=False)
        self.decoder_level2 = sequence(d*2, blocks[1], heads[1])
        self.up2_1 = Upsample(d*2)
        self.decoder_level1 = sequence(d*2, blocks[0], heads[0])
        self.refinement = sequence(d*2, cfg['num_refinement_blocks'], heads[0])
        self.output = nn.Conv2d(d*2, 1, 3, padding=1, bias=True)
        nn.init.zeros_(self.output.bias)

    def run_blocks(self, sequence, x):
        for block in sequence:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False, preserve_rng_state=True)
            else:
                x = block(x)
        return x

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (1, 256, 256):
            raise ValueError('Expected Bx1x256x256 original DAPI')
        enc1 = self.run_blocks(self.encoder_level1, self.patch_embed(image))
        enc2 = self.run_blocks(self.encoder_level2, self.down1_2(enc1))
        enc3 = self.run_blocks(self.encoder_level3, self.down2_3(enc2))
        latent = self.run_blocks(self.latent, self.down3_4(enc3))
        dec3 = self.run_blocks(self.decoder_level3,
            self.reduce_chan_level3(torch.cat((self.up4_3(latent), enc3), dim=1)))
        dec2 = self.run_blocks(self.decoder_level2,
            self.reduce_chan_level2(torch.cat((self.up3_2(dec3), enc2), dim=1)))
        dec1 = self.run_blocks(self.decoder_level1, torch.cat((self.up2_1(dec2), enc1), dim=1))
        return torch.sigmoid(self.output(self.run_blocks(self.refinement, dec1)))


def adapt_public_state(model, source):
    target = model.state_dict()
    new_biases = {'patch_embed.proj.bias', 'output.bias'}
    if set(source) != set(target)-new_biases:
        raise ValueError('Public checkpoint key mismatch')
    adapted = {}
    for key, value in target.items():
        if key in new_biases:
            adapted[key] = value.clone()
            continue
        original = source[key]
        shape = list(value.shape)
        if key == 'patch_embed.proj.weight':
            shape[1] = 3
        elif key == 'output.weight':
            shape[0] = 3
        if list(original.shape) != shape or not torch.isfinite(original).all():
            raise ValueError('Public checkpoint shape/value mismatch: ' + key)
        if key == 'patch_embed.proj.weight':
            adapted[key] = original.sum(dim=1, keepdim=True)
        elif key == 'output.weight':
            adapted[key] = value.clone()
        else:
            adapted[key] = original
    model.load_state_dict(adapted, strict=True)
    return {'strict_copied_tensors': len(target)-4,
            'adapted': ['patch_embed.proj.weight: RGB kernel sum',
                        'patch_embed.proj.bias: new learned fan-in uniform, private CPU seed2026',
                        'output.weight: seed2026 initialized single output',
                        'output.bias: new trainable scalar initialized zero']}


def load_public(model, path, expected_sha=None):
    allowed = {value['sha256'] for value in PUBLIC_WEIGHTS.values()}
    digest = sha256(path)
    if digest not in allowed or (expected_sha is not None and digest != expected_sha):
        raise ValueError('Pinned public checkpoint SHA256 mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if set(payload) != {'params'}:
        raise ValueError('Expected pinned public params checkpoint')
    return adapt_public_state(model, payload['params'])

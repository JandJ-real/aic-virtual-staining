"""One shared Restormer; fixed Marker ID selects FiLM and exactly one head."""
from __future__ import annotations

import copy

import torch
from torch import nn

from common import WEIGHT_SHA
from model import CONFIG, Restormer, load_public
from tasks import MARKER_IDS, validate_tasks


class MarkerFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(2, channels))
        self.beta = nn.Parameter(torch.zeros(2, channels))

    def forward(self, features, marker_id):
        gamma = self.gamma[marker_id].to(features.dtype)[None, :, None, None]
        beta = self.beta[marker_id].to(features.dtype)[None, :, None, None]
        return features*(1+gamma)+beta


class ConditionalRestormer(nn.Module):
    marker_conditioned = True

    def __init__(self, config=None, activation_checkpointing=True):
        super().__init__()
        self.config = copy.deepcopy(CONFIG if config is None else config)
        self.backbone = Restormer(self.config, activation_checkpointing)
        head = self.backbone.output
        del self.backbone.output
        self.heads = nn.ModuleDict({'CD68': head, 'CD45RO': copy.deepcopy(head)})
        d = self.config['dim']
        self.film_latent = MarkerFiLM(d*8)
        self.film_refinement = MarkerFiLM(d*2)

    @property
    def activation_checkpointing(self):
        return self.backbone.activation_checkpointing

    @activation_checkpointing.setter
    def activation_checkpointing(self, value):
        self.backbone.activation_checkpointing = bool(value)

    def validate_tasks(self, records):
        validate_tasks(records)

    def validate_gradients(self, marker_ids):
        expected = list(self.backbone.named_parameters())
        for marker, marker_id in MARKER_IDS.items():
            if marker_id in marker_ids:
                expected += [('heads.'+marker+'.'+n, p) for n, p in self.heads[marker].named_parameters()]
        expected += [('film_latent.'+n, p) for n, p in self.film_latent.named_parameters()]
        expected += [('film_refinement.'+n, p) for n, p in self.film_refinement.named_parameters()]
        missing = [name for name, p in expected if p.requires_grad and p.grad is None]
        if missing:
            raise RuntimeError('Missing active/shared M07 gradients: '+','.join(missing))

    def forward(self, image, marker_id):
        if type(marker_id) is not int or marker_id not in (0, 1):
            raise ValueError('Fixed integer Marker ID required: 0=CD68, 1=CD45RO')
        if image.ndim != 4 or image.shape[1:] != (1, 256, 256):
            raise ValueError('Expected Bx1x256x256 original DAPI')
        b = self.backbone
        enc1 = b.run_blocks(b.encoder_level1, b.patch_embed(image))
        enc2 = b.run_blocks(b.encoder_level2, b.down1_2(enc1))
        enc3 = b.run_blocks(b.encoder_level3, b.down2_3(enc2))
        latent = self.film_latent(b.run_blocks(b.latent, b.down3_4(enc3)), marker_id)
        dec3 = b.run_blocks(b.decoder_level3,
            b.reduce_chan_level3(torch.cat((b.up4_3(latent), enc3), dim=1)))
        dec2 = b.run_blocks(b.decoder_level2,
            b.reduce_chan_level2(torch.cat((b.up3_2(dec3), enc2), dim=1)))
        dec1 = b.run_blocks(b.decoder_level1, torch.cat((b.up2_1(dec2), enc1), dim=1))
        features = self.film_refinement(b.run_blocks(b.refinement, dec1), marker_id)
        marker = 'CD68' if marker_id == 0 else 'CD45RO'
        return torch.sigmoid(self.heads[marker](features))


def load_conditional_public(network, path, expected_sha=WEIGHT_SHA):
    if expected_sha != WEIGHT_SHA:
        raise ValueError('M07 first version is pinned to public Gaussian only')
    # A temporary CPU reference reuses the audited strict public adapter.
    # Its discarded random head must not consume the caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        reference = Restormer(network.config, activation_checkpointing=False)
        audit = load_public(reference, path, expected_sha=expected_sha)
    state = network.state_dict()
    copied = 0
    for name, value in reference.state_dict().items():
        if not name.startswith('output.'):
            state['backbone.'+name] = value
            copied += 1
    network.load_state_dict(state, strict=True)
    return dict(base_adapter=audit, backbone_tensors_copied=copied,
                conditional_parameters='two cloned seed2026 heads, zero gamma/beta',
                public_sha256=expected_sha, old_supervised_weights_used=False)

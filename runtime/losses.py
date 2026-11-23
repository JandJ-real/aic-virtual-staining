"""Existing Gaussian11 + normalized PSNR training proxy, no extra losses."""
import torch
from torch.nn import functional as F


def classic_gaussian_ssim(prediction, target):
    prediction, target = prediction.float(), target.float()
    if prediction.shape != target.shape or prediction.ndim != 4 or min(prediction.shape[-2:]) < 11:
        raise ValueError('SSIM requires matching NCHW tensors, at least 11x11')
    coordinates = torch.arange(11, device=prediction.device, dtype=torch.float32)-5.0
    one = torch.exp(-(coordinates ** 2)/(2.0*1.5**2))
    one = one/one.sum()
    window = torch.outer(one, one).expand(prediction.shape[1], 1, 11, 11).contiguous()
    def filt(value):
        return F.conv2d(value, window, groups=prediction.shape[1])
    mu_p, mu_t = filt(prediction), filt(target)
    mu_p2, mu_t2, mu_pt = mu_p.square(), mu_t.square(), mu_p*mu_t
    var_p = (filt(prediction.square())-mu_p2).clamp_min(0.0)
    var_t = (filt(target.square())-mu_t2).clamp_min(0.0)
    cov = filt(prediction*target)-mu_pt
    numerator = (2.0*mu_pt+.01**2)*(2.0*cov+.03**2)
    denominator = (mu_p2+mu_t2+.01**2)*(var_p+var_t+.03**2)
    return (numerator/denominator.clamp_min(1e-12)).clamp(-1, 1).mean()


def combined_loss(prediction, target):
    # Keep cancellation-sensitive SSIM moments and log MSE in float32.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        p, t = prediction.float(), target.float()
        mse = (p-t).pow(2).mean(dim=(-2, -1))
        psnr = -10.0*torch.log10(mse.clamp_min(10.0**(-50.0/10.0)))
        pixel = 1.0-(psnr/50.0).clamp(min=0.0, max=1.0).mean()
        ssim_loss = 1.0-classic_gaussian_ssim(p, t)
        # Preserve job603's expression and zero-weight edge branch for VJP parity.
        edge = (F.l1_loss(p[..., :, 1:]-p[..., :, :-1], t[..., :, 1:]-t[..., :, :-1])
                + F.l1_loss(p[..., 1:, :]-p[..., :-1, :], t[..., 1:, :]-t[..., :-1, :]))
        return .3*pixel+.7*ssim_loss+0.0*edge

"""FP32 non-DC focal frequency auxiliary; adaptive weights are stop-gradient."""
import torch

def frequency_loss(pred,target):
 p,t=pred.float(),target.float()
 if p.shape!=t.shape or p.ndim!=4:raise ValueError('Matching B,C,H,W tensors required')
 if not bool(torch.isfinite(p).all() and torch.isfinite(t).all()):raise FloatingPointError('Nonfinite frequency input; no repair')
 with torch.autocast(device_type=p.device.type,enabled=False):
  difference=torch.fft.fft2(p-t,norm='ortho')
  energy=difference.real.square()+difference.imag.square()
  with torch.no_grad():
   magnitude=energy.detach().sqrt();magnitude[...,0,0]=0
   weight=magnitude/magnitude.amax(dim=(-2,-1),keepdim=True).clamp_min(1e-12)
  loss=(weight*energy).mean()
 if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite frequency loss; step refused')
 return loss

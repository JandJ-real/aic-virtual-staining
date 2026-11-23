"""Strict equal-weight average of exactly the recorded tail model states."""
import torch

class MeanState:
 def __init__(self):self._sums=None;self._nonfloat=None;self._dtypes=None;self.count=0
 def add(self,state):
  if not state:raise ValueError('Empty state')
  if self._sums is None:
   sums={};nonfloat={};dtypes={}
   for k,v in state.items():
    if not isinstance(v,torch.Tensor):raise ValueError('State must contain tensors')
    dtypes[k]=(v.dtype,tuple(v.shape))
    if v.is_floating_point():
     if not torch.isfinite(v).all().item():raise FloatingPointError('Nonfinite state: '+k)
     sums[k]=v.detach().cpu().to(dtype=torch.float64).clone()
    else:nonfloat[k]=v.detach().cpu().clone()
   self._sums,self._nonfloat,self._dtypes=sums,nonfloat,dtypes;self.count=1;return
  if set(state)!=set(self._dtypes):raise ValueError('State keys changed')
  converted={}
  for k,v in state.items():
   if not isinstance(v,torch.Tensor) or (v.dtype,tuple(v.shape))!=self._dtypes[k]:raise ValueError('State shape/dtype changed: '+k)
   if v.is_floating_point():
    if not torch.isfinite(v).all().item():raise FloatingPointError('Nonfinite state: '+k)
    converted[k]=v.detach().cpu().to(dtype=torch.float64)
   elif not torch.equal(v.detach().cpu(),self._nonfloat[k]):raise ValueError('Nonfloating state changed: '+k)
  for k,v in converted.items():self._sums[k].add_(v)
  self.count+=1
 def finish(self,expected=10):
  if self.count!=expected or expected<1:raise ValueError('Incomplete tail average')
  return {k:((self._sums[k]/expected).to(dtype=dtype) if k in self._sums else self._nonfloat[k].clone())
          for k,(dtype,shape) in self._dtypes.items()}

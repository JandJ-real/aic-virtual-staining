"""Eight microbatch updates with original finite guards and method-specific changes."""
from pathlib import Path
import numpy as np
import torch
import m23_common as c
from common import COUNTS,ORGANS,WEIGHT_SHA
from dataio import paired_augmentation
from losses import combined_loss
from engine import (rng_state,atomic_torch_save,TrainingNumericsError,forward_record,
                    check_gradient_elements,checked_clip_gradients)
from sampling import domain_coefficients
from freq_loss import frequency_loss
from surgery import combine

def train_update(method,model,optimizer,reader,batch,step,device,loss_fn=combined_loss,
                 mixed_precision=True,clip_norm=1.,failure_path=None,ffl_weight=None,aux_fn=frequency_loss):
 if method not in c.METHODS:raise ValueError('Unknown method')
 expected=['colon']+['liver']*2+['stomach']*5
 if [r['organ'] for r in batch]!=expected or any(r['subset']!='train' for r in batch):raise ValueError('Ordered C1/L2/S5 Train-only update required')
 if method=='M13_sffl' and (ffl_weight is None or not np.isfinite(ffl_weight) or ffl_weight<=0):raise ValueError('Positive Train-calibrated FFL coefficient required')
 if getattr(model,'marker_conditioned',False):model.validate_tasks(batch)
 device=torch.device(device);model.train();optimizer.zero_grad(set_to_none=True)
 coeff=domain_coefficients();totals={o:0. for o in ORGANS};auxiliary={o:0. for o in ORGANS}
 entry=rng_state();buffers={n:v.detach().clone() for n,v in model.named_buffers()};micro_rng=[];task_grads={};names=list(model.named_parameters())
 context=dict(method=method,update=step,microbatch=0,phase='input',samples=batch,optimizer_step_performed=False,optimizer_step_started=False,mixed_precision=mixed_precision,ffl_weight=ffl_weight)
 diagnostics={}
 try:
  for position,row in enumerate(batch):
   micro_rng.append(rng_state());context.update(microbatch=position+1,sample=row,phase='input')
   x,y=paired_augmentation(*reader.pair(row),step,position,row['stem'])
   x=torch.from_numpy(x[None]).to(device);y=torch.from_numpy(y[None]).to(device)
   if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):raise TrainingNumericsError('nonfinite_input_or_target')
   context['phase']='forward'
   with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=mixed_precision):prediction=forward_record(model,x,row)
   if prediction.shape!=y.shape or not bool(torch.isfinite(prediction).all()):raise TrainingNumericsError('invalid_or_nonfinite_prediction')
   context['phase']='loss'
   with torch.autocast(device_type=device.type,enabled=False):
    main=loss_fn(prediction.float(),y.float());aux=prediction.float().sum()*0
    if method=='M13_sffl' and row['organ']=='stomach':aux=float(ffl_weight)*aux_fn(prediction.float(),y.float())
    loss=main+aux
   if not bool(torch.isfinite(main) and torch.isfinite(aux) and torch.isfinite(loss)):raise TrainingNumericsError('nonfinite_loss')
   context['phase']='backward';(loss*coeff[row['organ']]).backward()
   context['phase']='gradient_after_microbatch';check_gradient_elements(model)
   totals[row['organ']]+=float(main.detach())/COUNTS[row['organ']]
   auxiliary[row['organ']]+=float(aux.detach())/COUNTS[row['organ']]
   if method=='M23_sanchor' and position in (0,2,7):
    task_grads[row['organ']]={n:None if p.grad is None else p.grad.detach().clone() for n,p in names}
    optimizer.zero_grad(set_to_none=True)
  if method=='M23_sanchor':
   context['phase']='shared_gradient_projection'
   shared=[n for n,p in names if n.startswith('backbone.') and p.requires_grad]
   merged,diagnostics=combine(task_grads,shared)
   for n,p in names:p.grad=merged[n]
  context['phase']='gradient_after_accumulation'
  if hasattr(model,'validate_gradients'):model.validate_gradients({0,1})
  norm=checked_clip_gradients(model,clip_norm)
  context.update(phase='optimizer',optimizer_step_started=True);optimizer.step();context['optimizer_step_performed']=True
  if not bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all()):raise TrainingNumericsError('nonfinite_parameter_after_optimizer')
 except Exception as error:
  if failure_path is not None:
   path=Path(failure_path);path.parent.mkdir(parents=True,exist_ok=True)
   record=dict(context,**getattr(error,'details',{}),error=type(error).__name__+': '+str(error),interpretation='runtime failure, not a performance result')
   try:
    c.write(path,record);state=dict(model.state_dict())
    if not context['optimizer_step_started']:state.update(buffers)
    atomic_torch_save(path.parent/'failure_state.pt',dict(model=state,rng=entry,microbatch_rng=micro_rng,samples=batch,update=step,
      model_state_is_update_entry=not context['optimizer_step_started'],failure=record,
      contract=dict(method=method,public_sha256=WEIGHT_SHA,run_contract=getattr(model,'run_contract',None))))
    atomic_torch_save(path.parent/'failure_optimizer.pt',optimizer.state_dict())
    if task_grads:atomic_torch_save(path.parent/'failure_task_grads.pt',task_grads)
   except Exception as save_error:
    record['diagnostic_save_error']=repr(save_error)
    try:c.write(path,record)
    except OSError:pass
  raise
 return dict(losses=totals,auxiliary_losses=auxiliary,gradient_norm=float(norm.detach()),method_diagnostics=diagnostics)

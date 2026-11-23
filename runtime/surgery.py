"""Deterministic Stomach-anchored shared-backbone projection, a PCGrad-inspired variant."""
import torch

def combine(tasks,shared):
 organs=('colon','liver','stomach');keys=list(tasks['stomach'])
 if any(set(tasks[o])!=set(keys) for o in organs) or not shared:raise ValueError('Complete task gradient dictionaries and shared keys required')
 values=[v for o in organs for v in tasks[o].values() if v is not None]
 if not values or not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):raise FloatingPointError('Nonfinite task gradient; step refused')
 if any(v.dtype!=torch.float32 for v in values):raise ValueError('FP32 task gradient storage required')
 merged={}
 for k in keys:
  vs=[tasks[o][k] for o in organs if tasks[o][k] is not None]
  merged[k]=None if not vs else sum(vs[1:],vs[0].clone())
 if any(tasks[o][k] is None for o in organs for k in shared):raise ValueError('Missing shared task gradient')
 vectors={o:torch.cat([tasks[o][k].reshape(-1) for k in shared]) for o in organs}
 s=vectors['stomach'].double();ss=torch.dot(s,s)
 info=dict(projected=[],cos_colon_stomach=None,cos_liver_stomach=None,stomach_shared_norm=float(ss.sqrt()))
 projected={}
 for o in ('colon','liver'):
  a=vectors[o].double();dot=torch.dot(a,s);aa=torch.dot(a,a)
  info['dot_'+o+'_stomach']=float(dot)
  if float(ss)>0 and float(aa)>0:info['cos_'+o+'_stomach']=float(dot/(ss*aa).sqrt())
  if float(ss)>0 and float(dot)<0:
   a=a-(dot/ss)*s;info['projected'].append(o)
  projected[o]=a.float()
 joint=vectors['stomach']+projected['colon']+projected['liver']
 if not bool(torch.isfinite(joint).all()):raise FloatingPointError('Nonfinite projected gradient; step refused')
 info['joint_dot_stomach']=float(torch.dot(joint.double(),s))
 offset=0
 for k in shared:
  ref=tasks['stomach'][k];size=ref.numel();merged[k]=joint[offset:offset+size].view_as(ref);offset+=size
 return merged,info

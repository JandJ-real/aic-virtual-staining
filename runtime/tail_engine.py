"""Ten explicitly sourced tail weights, equal parameter average, no prediction ensemble."""
import hashlib,json,math,os,time
from pathlib import Path
import torch
from avgmath import MeanState

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 tmp.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'));os.replace(str(tmp),str(path))
def save(path,payload):
 path=Path(path);tmp=path.with_name(path.name+'.tmp')
 if path.exists() or tmp.exists():raise FileExistsError(str(path))
 torch.save(payload,tmp);os.replace(str(tmp),str(path))

def run_tail(model,update,folder,contract,interval=245,rounds=10):
 if interval<1 or rounds!=10:raise ValueError('Ten positive tail intervals required')
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'progress.json').exists() or (folder/'avg.pt').exists():raise FileExistsError('Tail exists, no automatic repeat')
 state=dict(status='running',updates=0,target=interval*rounds,points=0,sources=[]);avg=MeanState();started=time.monotonic()
 try:
  write(folder/'progress.json',state)
  for n in range(1,interval*rounds+1):
   result=update(n);norm=float(result['gradient_norm'])
   if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm')
   state.update(updates=n,last_gradient_norm=norm,seconds=time.monotonic()-started)
   if n%interval==0:
    pt=folder/('tail_%02d.pt'%(n//interval));weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    avg.add(weights);save(pt,dict(model=weights,contract=contract,tail_update=n))
    state['sources'].append(dict(update=n,file=pt.name,sha256=sha(pt)));state['points']=avg.count
   if n==1 or n%25==0 or n%interval==0:write(folder/'progress.json',state)
  merged=avg.finish(rounds);model.load_state_dict(merged,strict=True)
  save(folder/'avg.pt',dict(model=merged,contract=contract,tail_updates=interval*rounds,avg_sources=state['sources']))
  state.update(status='averaged',sha256=sha(folder/'avg.pt'),seconds=time.monotonic()-started);write(folder/'progress.json',state)
  return state
 except Exception as error:
  state.update(status='failed',error=type(error).__name__+': '+str(error),interpretation='runtime failure, not performance result')
  write(folder/'progress.json',state);raise

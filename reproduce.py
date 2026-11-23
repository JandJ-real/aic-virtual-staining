"""Offline reproduction entry. No scheduler, SSH, network or live-worker dependency."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime'))
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from repro_contract import paths,sha,main_lr,validate_endpoint,FINAL_SHA,DATA_SHA,PUBLIC_SHA,INDEX_SHA,TASKS,original_runtime

def emit(value):print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)

def load_model(path):
 import torch
 from conditional_model import ConditionalRestormer
 from model import CONFIG
 digest=sha(path);p=torch.load(path,map_location='cpu',weights_only=False)
 validate_endpoint(p,digest)
 if any(t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) for t in p['model'].values()):
  raise ValueError('Nonfinite/non-FP32 checkpoint')
 net=ConditionalRestormer(CONFIG,activation_checkpointing=False)
 net.load_state_dict(p['model'],strict=True)
 return net,p

def assets(data=None):
 p=paths();archive=Path(data) if data else p['data']
 if sha(archive)!=DATA_SHA or sha(p['public'])!=PUBLIC_SHA or sha(ROOT/'data/train_full.csv')!=INDEX_SHA:
  raise ValueError('Official archive, public initialization or FULL index SHA mismatch')
 from dataio import load_index,member_role
 rows=load_index(ROOT/'data/train_full.csv',subset='train')
 if len(rows)!=10356 or Counter(r['organ'] for r in rows)!=dict(colon=6296,liver=2380,stomach=1680):
  raise ValueError('FULL training population changed')
 if any(r['marker']!=TASKS[r['organ']] or member_role(r['dapi'])!='train' or member_role(r['target'])!='train' for r in rows):
  raise ValueError('Training input/target leakage or Marker mismatch')
 return p,archive,rows

def environment(device):
 import platform
 import torch
 from runtime_compat import check_runtime
 runtime=check_runtime()
 runtime.update(torch_build=str(torch.__version__),CUDA_runtime=torch.version.cuda,cudnn_version=torch.backends.cudnn.version())
 runtime['original_environment']=original_runtime(runtime['versions'],str(torch.__version__),torch.version.cuda,torch.backends.cudnn.version())
 torch.set_num_threads(2)
 if device.type=='cuda':
  if not runtime['original_environment']:raise RuntimeError('Use original Python3.9.18 and CUDA package versions for scored reproduction')
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 GPU required')
  torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.use_deterministic_algorithms(True)
  runtime.update(gpu=torch.cuda.get_device_name(device),cuda_checked=True)
 return runtime

def check(a):
 import torch
 from infer_io import load_inputs,verify_official_inputs
 from dataio import ArchiveReader
 p,archive,rows=assets(a.data);net,payload=load_model(p['model'])
 inputs=load_inputs(ROOT/'data/test.csv');audit=verify_official_inputs(inputs,archive)
 # CPU asset checks verify every supervised image byte without running a model.
 with ArchiveReader(archive,cache_size=0) as reader:
  for row in rows:reader.pair(row)
 result=dict(status='passed',train=len(rows),test=len(inputs),model_tensors=len(payload['model']),
  parameters=sum(x.numel() for x in net.parameters()),checkpoint_sha256=FINAL_SHA,
  main_updates=14000,tail_updates=2450,points=10,independent_validation=None,
  input_audit=audit,runtime=environment(torch.device('cpu')),model_forward_performed=False,
  cuda_initialized=torch.cuda.is_initialized())
 emit(result)

def infer(a):
 import torch
 from dataio import ArchiveReader
 from engine import model_digest
 from infer_io import load_inputs,verify_official_inputs,predict
 p,archive,_=assets(a.data);device=torch.device(a.device)
 env=environment(device);net,payload=load_model(p['model']);net.to(device);net.eval()
 rows=load_inputs(ROOT/'data/test.csv');verify_official_inputs(rows,archive)
 before=model_digest(net);output=Path(a.output).resolve();start=time.monotonic()
 def final_guard():
  if sha(p['model'])!=FINAL_SHA or model_digest(net)!=before:raise ValueError('Frozen model changed')
 def progress(n):emit(dict(completed=n,total=2186,seconds=time.monotonic()-start))
 with ArchiveReader(archive) as reader:
  result=predict(net,reader,rows,device,output,progress=progress,final_guard=final_guard)
 result.update(path=str(output),checkpoint_sha256=FINAL_SHA,zip_sha256=sha(output),environment=env,
  single_checkpoint=True,tta=False,task_markers=TASKS,seed=2026)
 output.with_name(output.name+'.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
 emit(result)

def fresh(device):
 from conditional_model import ConditionalRestormer,load_conditional_public
 from model import CONFIG
 from engine import seed_all
 seed_all(2026);net=ConditionalRestormer(CONFIG,activation_checkpointing=True)
 audit=load_conditional_public(net,paths()['public'])
 net.run_contract=dict(method='M23_full14',split='FULL',independent_validation=None)
 return net.to(device),audit

def train(a):
 import numpy as np
 import torch
 from common import write_json
 from dataio import ArchiveReader
 from engine import seed_all,rng_state,atomic_torch_save,evaluation_context,single_prediction
 from sampling import DomainSampler
 from long_update import train_update
 from tail_engine import run_tail
 p,archive,rows=assets(a.data);device=torch.device(a.device)
 if device.type!='cuda':raise ValueError('Full training requires CUDA')
 env=environment(device)
 folder=Path(a.output).resolve()
 if folder.exists():raise FileExistsError('New training output directory required; no automatic resume')
 folder.mkdir(parents=True);write_json(folder/'environment.json',env)
 def optimizer(net,lr=1e-5):return torch.optim.AdamW(net.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
 def update(net,opt,reader,sampler,step,where,mixed=True):
  return train_update('M23_sanchor',net,opt,reader,sampler.next_update(),step,device,
   mixed_precision=mixed,clip_norm=1.,failure_path=where/'failure.json')
 cfg=json.loads((ROOT/'config/final.json').read_text(encoding='utf-8'))
 contract=dict(cfg,environment=env,source_index_sha256=INDEX_SHA,preflight_weights_used=False,
  workflow='fresh offline numerical reconstruction of full14000; no A-Val weights or live PID')
 with ArchiveReader(archive,cache_size=30000) as reader:
  probe=folder/'preflight';probe.mkdir()
  for mixed,count in ((False,1),(True,100)):
   net,_=fresh(device);opt=optimizer(net);sampler=DomainSampler(rows,2026)
   for n in range(1,count+1):
    update(net,opt,reader,sampler,n,probe,mixed)
    if n==1 or n%25==0:write_json(probe/'progress.json',dict(precision='BF16' if mixed else 'FP32',step=n,weights_discarded=True))
   del net,opt;gc.collect();torch.cuda.empty_cache()
  net,_=fresh(device)
  with evaluation_context(net):
   for marker in ('CD68','CD45RO'):
    out=single_prediction(net,np.zeros((1,256,256),np.float32),device,row={'marker':marker})
    if not np.isfinite(out).all():raise FloatingPointError('FP32 evaluation preflight failed')
  del net;gc.collect();torch.cuda.empty_cache()
  write_json(probe/'result.json',dict(status='passed',fp32_updates=1,bf16_updates=100,weights_discarded=True))
  net,init=fresh(device);opt=optimizer(net);sampler=DomainSampler(rows,2026)
  contract['public']=init;write_json(folder/'contract.json',contract);start=time.monotonic()
  for n in range(1,14001):
   for group in opt.param_groups:group['lr']=main_lr(n)
   result=update(net,opt,reader,sampler,n,folder)
   if n==1 or n%25==0:
    write_json(folder/'progress.json',dict(status='running',step=n,target=14000,seconds=time.monotonic()-start,**result))
   if n%250==0:
    atomic_torch_save(folder/'latest.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),sampler=sampler.state_dict(),rng=rng_state(),step=n,contract=contract))
  atomic_torch_save(folder/'main.pt',dict(model=net.state_dict(),contract=contract,step=14000))
  write_json(folder/'progress.json',dict(status='completed',step=14000,target=14000,seconds=time.monotonic()-start))
  del opt;seed_all(2026);sampler=DomainSampler(rows,2026);opt=optimizer(net)
  tail=folder/'avg';tc=dict(method='M23_full14_retrained_AVG_F',source_contract=contract,
   anchor_sha256=sha(folder/'main.pt'),anchor_step=14000,tail_interval=245,tail_rounds=10,independent_validation=None)
  result=run_tail(net,lambda k:update(net,opt,reader,sampler,14000+k,tail),tail,tc)
  write_json(folder/'complete.json',dict(status='completed',main_updates=14000,AVG=result,
   final_candidate=str(tail/'avg.pt'),independent_validation=None))
  emit(dict(status='completed',retrained_checkpoint=str(tail/'avg.pt'),original_submitted_checkpoint_unchanged=True))

def main():
 parser=argparse.ArgumentParser(description='M23 FULL14000 + AVG10 offline package')
 sub=parser.add_subparsers(dest='action',required=True)
 for name in ('check','infer','train'):
  p=sub.add_parser(name);p.add_argument('--data',help='Original complete official ZIP; defaults to bundled archive')
  if name!='check':
   p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True)
 a=parser.parse_args();{'check':check,'infer':infer,'train':train}[a.action](a)

if __name__=='__main__':main()

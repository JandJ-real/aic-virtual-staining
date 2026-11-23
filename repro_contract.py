"""Frozen identity of the submitted best-scoring full-data model."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FINAL_SHA='042816bbab9fd844738f68348f12b761ba0e87c0ce703de3b541d63a83e88922'
MAIN_SHA='fe60d96537fb8851df272acf0f7e11fae3468846f846683844d748d1350c73e0'
ZIP_SHA='dc7a6fa01313fe950b8cfb4eb0b2eece7649e48a3256eafbfaf02be0e7c99722'
DATA_SHA='d7e66cc07facc289227a214897e93a6908242b2bbdb017a097b15156bb772e4d'
PUBLIC_SHA='4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd'
INDEX_SHA='45bb2fccb2ae34e65ad4b75abf1f2d837f38c6249458967e75befedcd2c1b346'
TASKS={'colon':'CD68','liver':'CD45RO','stomach':'CD68'}
ZIP_NAME='reproduced.zip'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4194304),b''):h.update(block)
 return h.hexdigest()

def main_lr(step):
 if not 1<=step<=14000:raise ValueError('Expected main update1..14000')
 return next(lr for end,lr in ((250,1e-5),(500,5e-5),(11200,1e-4),(14000,5e-5)) if step<=end)

def validate_endpoint(payload,digest):
 c=payload.get('contract',{});source=c.get('source_contract',{});points=payload.get('avg_sources',[])
 if (digest!=FINAL_SHA or c.get('method')!='M23_full14_AVG_F'
   or c.get('anchor_sha256')!=MAIN_SHA or c.get('anchor_step')!=14000
   or c.get('tail_interval')!=245 or c.get('tail_rounds')!=10
   or c.get('independent_validation') is not None or payload.get('tail_updates')!=2450
   or source.get('method')!='M23_full14' or source.get('main_updates')!=14000
   or source.get('split')!='FULL' or source.get('seed')!=2026
   or source.get('source_index_sha256')!=INDEX_SHA or source.get('task_markers')!=TASKS
   or source.get('independent_validation') is not None
   or source.get('initialization')!='fresh public Gaussian Restormer; no SSL or old supervised weights'
   or len(points)!=10 or [p.get('update') for p in points]!=list(range(245,2451,245))):
  raise ValueError('Not the frozen M23 full14000 + complete AVG10 checkpoint')
 return True

def paths():
 config=json.loads((ROOT/'config/paths.json').read_text(encoding='utf-8'))
 return {key:(ROOT/value).resolve() for key,value in config.items()}

def original_runtime(metadata,torch_build,cuda,cudnn):
 versions=dict(metadata);versions['torch']=versions.get('torch','').split('+')[0]
 expected=dict(python='3.9.18',torch='2.4.1',numpy='1.20.3',Pillow='8.4.0',scipy='1.7.1',**{'scikit-image':'0.18.3'})
 return versions==expected and torch_build=='2.4.1+cu121' and cuda=='12.1' and cudnn==90100

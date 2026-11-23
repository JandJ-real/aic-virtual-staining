"""Local M23 update contract and durable diagnostics writer."""
import json,os
from pathlib import Path
METHODS=('M23_sanchor',)
def write(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
 with tmp.open('w',encoding='utf-8',newline='\n') as s:json.dump(data,s,ensure_ascii=False,indent=2,allow_nan=False);s.write('\n')
 os.replace(str(tmp),str(path))

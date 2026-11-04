from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent;out=base/'memory-all'
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
command=['nsys','profile','--trace=cuda','--cuda-memory-usage=true','--cuda-trace-all-apis=true','--sample=none','--cpuctxsw=none','--output='+str(out/'all-models'),'python3',str(base/'memory_children.py')]
(out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
with (out/'profile.log').open('w') as log:subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('All child workloads captured; exporting SQLite',flush=True)
with (out/'export.log').open('w') as log:subprocess.run(['nsys','export','--type','sqlite','--output',str(out/'all-models.sqlite'),str(out/'all-models.nsys-rep')],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run(['python3',str(base/'audit_memory_all.py')],check=True)

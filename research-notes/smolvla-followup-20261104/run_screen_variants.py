"""Run three authorized screening variants sequentially; retain failures and outputs."""
from pathlib import Path
import json,subprocess,sys
b=Path(__file__).resolve().parent;root=b.parents[1];out=b/'screen40-variants'
out.mkdir(exist_ok=True)
if (out/'batch.json').exists():raise SystemExit('Batch already exists; no automatic overwrite')
if subprocess.run(['pgrep','-x','vla-server'],stdout=subprocess.DEVNULL).returncode==0:raise SystemExit('Existing server; refuse concurrent evaluation')
cases=[('flash-mm','/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf',['--flash-attn','--mm-prec','default']),('q8-lm',str(root/'research-notes/generated-models/smolvla-q8_0-lm.gguf'),[]),('q8-vision',str(root/'research-notes/generated-models/smolvla-q8_0-vision.gguf'),[])]
rows=[]
def status(state,current=None):
 (out/'batch.json').write_text(json.dumps(dict(state=state,current=current,expected_configurations=3,expected_episodes=120,results=rows),indent=2)+'\n')
status('starting')
for name,model,flags in cases:
 status('running',name)
 command=[sys.executable,'-u',str(b/'run_variant40.py'),name,model]+flags
 print('START',name,'10 tasks x4 episodes',flush=True)
 r=subprocess.run(command,cwd=root)
 rows.append(dict(name=name,exit=r.returncode,result=str(out/name/'result.json')))
 if r.returncode:
  status('failed',name);raise SystemExit(r.returncode)
 print('COMPLETE',name,flush=True)
status('complete')

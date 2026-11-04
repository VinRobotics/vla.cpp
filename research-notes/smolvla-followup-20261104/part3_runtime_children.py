from pathlib import Path
import json,os,subprocess,re
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent;out=base/'part3-runtime'
model='/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf'
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
configs=[('defaults',[]),('flash',['--flash-attn']),('mm-default',['--mm-prec','default']),('flash-mm',['--flash-attn','--mm-prec','default']),('bf16',['--weight-dtype','bf16']),('bf16-flash',['--weight-dtype','bf16','--flash-attn']),('f16',['--weight-dtype','f16']),('f16-flash',['--weight-dtype','f16','--flash-attn']),('steps-5',['--num-steps','5']),('steps-3',['--num-steps','3'])]
records=[]
mode=__import__('sys').argv[1]
for trial in range(3):
 for name,flags in (configs if trial%2==0 else list(reversed(configs))):
  command=[str(root/'build-cuda/vla-bench'),'--ckpt',model,'--images','2','--size','512','--tokens','16','--warmup','3','--reps','20']+flags
  path=out/f'{mode}-{name}-{trial}.log'
  with path.open('w') as log:
   p=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
  r=dict(name=name,trial=trial,pid=p.pid,command=command,exit=code)
  if code:raise RuntimeError(str(r))
  text=path.read_text();m=re.search(r'mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms',text)
  if not m:raise RuntimeError('Missing summary '+str(path))
  r.update(mean_ms=float(m[1]),p50_ms=float(m[2]),p90_ms=float(m[3]))
  records.append(r);(out/(mode+'-runs.json')).write_text(json.dumps(records,indent=2)+'\n')
  print(mode,name,trial+1,r['mean_ms'],flush=True)

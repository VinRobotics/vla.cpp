from pathlib import Path
import json,subprocess,os
out=Path(__file__).resolve().parent/'runtime-screen'
out.mkdir(exist_ok=True)
model='/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf'
configs=[('defaults',[]),('flash',['--flash-attn']),('mm-default',['--mm-prec','default']),('flash-mm',['--flash-attn','--mm-prec','default']),('bf16',['--weight-dtype','bf16']),('bf16-flash',['--weight-dtype','bf16','--flash-attn']),('f16',['--weight-dtype','f16']),('f16-flash',['--weight-dtype','f16','--flash-attn']),('steps-5',['--num-steps','5']),('steps-3',['--num-steps','3'])]
commands=[]
for name,flags in configs:
 cmd=['./build-cuda/vla-bench','--ckpt',model,'--images','2','--size','512','--tokens','16','--warmup','2','--reps','5']+flags
 commands.append({'name':name,'command':cmd})
 with (out/(name+'.log')).open('w') as log:
  r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 print(name,'bench exit',r.returncode,flush=True)
 if r.returncode: continue
 cmd=['./build-cuda/tests/vla_predict_check',model,'','2']+flags
 commands.append({'name':name+'-output','command':cmd,'VLA_IMG_SIZE':'512'})
 with (out/(name+'.out')).open('w') as stdout,(out/(name+'-predict.log')).open('w') as stderr:
  r=subprocess.run(cmd,env=dict(os.environ,VLA_IMG_SIZE='512'),stdout=stdout,stderr=stderr,timeout=180)
 print(name,'output exit',r.returncode,flush=True)
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')

from pathlib import Path
import subprocess,json,os
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
out=base/'memory-flash-mm'
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
commands=[]
for trial in range(1,4):
    prefix=out/f'flash-mm-{trial}'
    command=['nsys','profile','--trace=cuda','--cuda-memory-usage=true','--cuda-trace-all-apis=true','--sample=none','--cpuctxsw=none','--output='+str(prefix),str(root/'build-cuda/vla-bench'),'--ckpt','/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf','--images','2','--size','512','--tokens','16','--warmup','3','--reps','20','--flash-attn','--mm-prec','default']
    commands.append(command)
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/f'profile-{trial}.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,cwd=root)
    subprocess.run(['nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],check=True)
    print('memory flash-mm trial',trial,'complete',flush=True)
subprocess.run(['python3',str(out/'summarize_memory.py')],check=True)

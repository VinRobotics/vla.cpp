"""Run serial child workloads under one nsys process-tree memory capture."""
from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent;out=base/'memory-all'
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
cases=[('defaults',source)]+[(p.stem,p) for p in sorted((root/'research-notes/generated-models').glob('*.gguf'))]
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
records=[]
for trial in range(3):
    for name,model in cases:
        command=[str(root/'build-cuda/vla-bench'),'--ckpt',str(model),'--images','2','--size','512','--tokens','16','--warmup','3','--reps','20']
        with (out/f'{name}-{trial}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            code=process.wait()
        records.append(dict(name=name,trial=trial,pid=process.pid,command=command,exit=code))
        (out/'children.json').write_text(json.dumps(records,indent=2)+'\n')
        print(name,'memory workload',trial+1,'exit',code,flush=True)
        if code:raise SystemExit(code)

"""vla-bench full protocol for prepared GGUFs, with defaults control each round."""
from pathlib import Path
import json,os,re,subprocess,time
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
out=base/'candidate-bench';out.mkdir(exist_ok=False)
checks=json.loads((base/'candidate-checks/summary.json').read_text())
valid={r['name'] for r in checks if r['status']=='cuda_predict_ok'}
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
cases=[('defaults',source)]+[(p.stem,p) for p in sorted((root/'research-notes/generated-models').glob('*.gguf')) if p.stem in valid]
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
results=[]
for trial in range(3):
    for name,model in cases:
        command=[str(root/'build-cuda/vla-bench'),'--ckpt',str(model),'--images','2','--size','512','--tokens','16','--warmup','3','--reps','20']
        with (out/f'{name}-{trial}.log').open('w') as log,(out/f'{name}-{trial}-telemetry.jsonl').open('w') as telemetry:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                query=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,memory.used','--format=csv'],capture_output=True,text=True,timeout=10)
                telemetry.write(json.dumps({'time':time.time(),'pid':process.pid,'gpu':query.stdout+query.stderr})+'\n');telemetry.flush();time.sleep(.5)
        record=dict(name=name,model=str(model),trial=trial,command=command,exit=process.returncode)
        if process.returncode:record['status']='benchmark_failed'
        else:
            text=(out/f'{name}-{trial}.log').read_text();m=re.search(r'min ([\d.]+) ms\s+mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms\s+vision ([\d.]+) ms',text)
            if m is None:raise RuntimeError('Missing benchmark summary: '+name)
            record.update(status='ok',min_ms=float(m[1]),mean_ms=float(m[2]),p50_ms=float(m[3]),p90_ms=float(m[4]),vision_ms=float(m[5]))
        results.append(record);(out/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,trial,record['status'],record.get('mean_ms'),flush=True)
selected={}
for name,_ in cases:
    rows=[r for r in results if r['name']==name and r['status']=='ok']
    if len(rows)==3:selected[name]=min(rows,key=lambda r:r['mean_ms'])
(out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')

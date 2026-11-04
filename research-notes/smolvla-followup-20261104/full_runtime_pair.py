"""Full vla-bench protocol, interleaved defaults vs flash+mm-default."""
from pathlib import Path
import json,os,re,subprocess,time
root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent
screen=json.loads((base/'screen40-defaults/progress.json').read_text())
if screen['state']!='complete':raise SystemExit('LIBERO screening still active: do not share GPU')
out=base/'runtime-full-pair';out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
results=[]
for trial in range(3):
    order=['defaults','flash-mm'] if trial%2==0 else ['flash-mm','defaults']
    for name in order:
        flags=[] if name=='defaults' else ['--flash-attn','--mm-prec','default']
        command=[str(root/'build-cuda/vla-bench'),'--ckpt','/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf','--images','2','--size','512','--tokens','16','--warmup','3','--reps','20']+flags
        with (out/f'{name}-{trial}.log').open('w') as log,(out/f'{name}-{trial}-telemetry.jsonl').open('w') as telemetry:
            process=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                data={'time':time.time(),'pid':process.pid}
                for label,fields in [('gpu','--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,memory.used'),('processes','--query-compute-apps=pid,process_name,used_gpu_memory')]:
                    query=subprocess.run(['nvidia-smi',fields,'--format=csv'],capture_output=True,text=True,timeout=10)
                    data[label]=query.stdout+query.stderr
                telemetry.write(json.dumps(data)+'\n');telemetry.flush();time.sleep(.5)
        if process.returncode:raise RuntimeError(f'{name} trial {trial} exit {process.returncode}')
        text=(out/f'{name}-{trial}.log').read_text()
        match=re.search(r'min ([\d.]+) ms\s+mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms\s+vision ([\d.]+) ms',text)
        if match is None:raise RuntimeError('Missing benchmark summary')
        record=dict(variant=name,trial=trial,command=command,min_ms=float(match[1]),mean_ms=float(match[2]),p50_ms=float(match[3]),p90_ms=float(match[4]),vision_ms=float(match[5]))
        results.append(record);(out/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,trial,record['mean_ms'],flush=True)
selected={name:min((r for r in results if r['variant']==name),key=lambda r:r['mean_ms']) for name in ['defaults','flash-mm']}
selected['relative_mean_change']=selected['flash-mm']['mean_ms']/selected['defaults']['mean_ms']-1
(out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')

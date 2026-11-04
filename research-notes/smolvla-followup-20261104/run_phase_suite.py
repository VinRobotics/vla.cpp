from pathlib import Path
import json,subprocess,os
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
out=base/'phase-suite';out.mkdir(exist_ok=False)
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
cases=[('defaults',source,'defaults'),('flash-mm',source,'flash-mm')]+[(name,root/'research-notes/generated-models'/(name+'.gguf'),'defaults') for name in ['smolvla-vision-tail-10','smolvla-vision-spaced-10','smolvla-coupled-tail-30']]
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
results=[]
for trial in range(3):
    for name,model,variant in cases:
        folder=out/f'{name}-{trial}'
        command=[str(root/'.venv/bin/python'),str(base/'profile_phase.py'),'--model',str(model),'--variant',variant,'--mode','phase','--output',str(folder)]
        with (out/f'{name}-{trial}.log').open('w') as log:
            process=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:raise RuntimeError(f'{name}/{trial} exit {process.returncode}')
        r=json.loads((folder/'result.json').read_text())
        results.append(dict(case=name,trial=trial,summary=r['summary'],result=str(folder/'result.json'),command=command))
        (out/'runs.json').write_text(json.dumps(results,indent=2)+'\n')
        print(name,'phase trial',trial,'total mean',r['summary']['ms_total']['mean'],flush=True)
selected={name:min((r for r in results if r['case']==name),key=lambda r:r['summary']['ms_total']['mean']) for name,_,_ in cases}
(out/'selected.json').write_text(json.dumps(selected,indent=2)+'\n')

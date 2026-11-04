from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
out=base/'kernel-phase-traces';out.mkdir(exist_ok=False)
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
cases=[('defaults',source,'defaults'),('flash-mm',source,'flash-mm')]+[(name,root/'research-notes/generated-models'/(name+'.gguf'),'defaults') for name in ['smolvla-vision-spaced-10','smolvla-coupled-tail-30']]
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
records=[]
for name,model,variant in cases:
    print('PHASE NODE TRACE starting',name,flush=True)
    prefix=out/name
    cmd=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--cpuctxsw=none','--output='+str(prefix),str(root/'.venv/bin/python'),str(base/'profile_phase.py'),'--model',str(model),'--variant',variant,'--mode','phase','--capture','--output',str(out/(name+'-workload'))]
    with (out/(name+'-profile.log')).open('w') as log:subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/(name+'-export.log')).open('w') as log:subprocess.run(['nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/(name+'-kernel-analysis.log')).open('w') as log:subprocess.run(['python3',str(base/'summarize_kernels.py'),str(prefix)+'.sqlite'],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/(name+'-module-analysis.log')).open('w') as log:result=subprocess.run(['python3',str(base/'summarize_graph_modules.py'),str(prefix)+'.sqlite','--mode','phase'],stdout=log,stderr=subprocess.STDOUT)
    records.append(dict(name=name,model=str(model),command=cmd,module_analysis_exit=result.returncode))
    (out/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print('PHASE NODE TRACE complete',name,'module analysis exit',result.returncode,flush=True)

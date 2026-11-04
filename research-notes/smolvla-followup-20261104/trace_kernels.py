from pathlib import Path
import json,os,subprocess
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
out=base/'kernel-traces';out.mkdir(exist_ok=False)
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')}
commands=[]
for variant in ['defaults','flash-mm']:
    prefix=out/variant
    command=['nsys','profile','--trace=cuda','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none','--cpuctxsw=none','--output='+str(prefix),str(root/'.venv/bin/python'),str(base/'profile_phase.py'),'--variant',variant,'--mode','none','--capture','--output',str(out/(variant+'-workload'))]
    commands.append(command);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/(variant+'-profile.log')).open('w') as log:
        subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/(variant+'-export.log')).open('w') as log:
        subprocess.run(['nsys','export','--type','sqlite','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/(variant+'-analysis.log')).open('w') as log:
        subprocess.run(['python3',str(base/'summarize_kernels.py'),str(prefix)+'.sqlite'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(variant,'node trace + analysis complete',flush=True)

"""CUDA fixed-input checks for exports; not a LIBERO smoke test."""
from pathlib import Path
import json,os,subprocess
import numpy as np
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent
if json.loads((base/'screen40-defaults/progress.json').read_text())['state']!='complete':raise SystemExit('Wait for LIBERO to finish')
out=base/'candidate-checks';out.mkdir(exist_ok=False)
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
candidates=[('defaults',source,[]),('flash-mm',source,['--flash-attn','--mm-prec','default'])]+[(p.stem,p,[]) for p in sorted((root/'research-notes/generated-models').glob('*.gguf'))]
env={k:v for k,v in os.environ.items() if not k.startswith('VLA_')};env['VLA_IMG_SIZE']='512'
results=[];reference=None
for name,model,flags in candidates:
    command=[str(root/'build-cuda/tests/vla_predict_check'),str(model),'','2']+flags
    with (out/(name+'.out')).open('w') as stdout,(out/(name+'.log')).open('w') as stderr:
        result=subprocess.run(command,cwd=root,env=env,stdout=stdout,stderr=stderr,timeout=180)
    record=dict(name=name,model=str(model),command=command,exit=result.returncode,status='failed')
    if result.returncode==0:
        lines=(out/(name+'.out')).read_text().splitlines()
        index=next(i for i,s in enumerate(lines) if s.startswith('action_len='))
        actions=np.asarray([float(s) for s in lines[index+1:]],dtype=np.float32).reshape(50,32)
        assert np.isfinite(actions).all(),name
        assert np.all(actions[:,7:]==0),name
        banner='\n'.join(lines)+(out/(name+'.log')).read_text()
        assert 'backend = CUDA' in banner,name
        if reference is None:reference=actions
        d=actions[:,:7].astype(float)-reference[:,:7].astype(float)
        record.update(status='cuda_predict_ok',shape=list(actions.shape),max_abs_real7=float(np.abs(d).max()),rmse_real7=float(np.sqrt(np.mean(d*d))),changed_real7=int(np.count_nonzero(actions[:,:7].copy().view(np.uint32)!=reference[:,:7].copy().view(np.uint32))))
    results.append(record);(out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(name,record['status'],record.get('max_abs_real7'),flush=True)

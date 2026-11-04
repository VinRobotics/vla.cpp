"""One process of phase timing; run only when GPU is free of LIBERO.

Synthetic input matches vla-bench (2x512, token IDs 1..16, fixed state/noise).
Phase mode changes graph construction/padding; this is not baseline throughput.
"""
from pathlib import Path
import argparse,json,os,sys,time,ctypes
root=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',default='/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
ap.add_argument('--variant',choices=['defaults','flash-mm'],default='defaults')
ap.add_argument('--mode',choices=['none','phase'],default='phase')
ap.add_argument('--output',required=True)
ap.add_argument('--capture',action='store_true',help='cudaProfilerStart after warmups; stop after 20 reps')
args=ap.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
os.environ['VLA_LIBRARY']=str(root/'build-cuda/libvla.so')
sys.path.insert(0,str(root/'bindings/python'))
import numpy as np
import vla_cpp
config=None
if args.variant=='flash-mm':
    config=out/'runtime.json';config.write_text(json.dumps({'runtime':{'flash_attn':True,'mm_prec':'default'}})+'\n')
y,x,c=np.indices((512,512,3))
images=[((x+2*y+40*c+17*v)&255).astype(np.uint8) for v in range(2)]
tokens=list(range(1,17))
rows=[]
with vla_cpp.load(args.model,config_path=str(config) if config else None) as model:
    cfg=model.config
    state=np.zeros(cfg.max_state_dim,dtype=np.float32)
    state[:cfg.real_state_dim]=np.float32(.01)*np.arange(1,cfg.real_state_dim+1,dtype=np.float32)
    count=cfg.max_action_dim*cfg.n_suffix
    noise=np.float32(.001)*((np.arange(count,dtype=np.uint64)*2654435761)%1000).astype(np.float32)-np.float32(.5)
    timing=vla_cpp.TIMING_PHASE if args.mode=='phase' else vla_cpp.TIMING_NONE
    cudart=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so') if args.capture else None
    for iteration in range(23):
        if iteration==3 and cudart is not None:
            assert cudart.cudaDeviceSynchronize()==0
            assert cudart.cudaProfilerStart()==0
        before=time.perf_counter()
        actions=model.predict(images,tokens,state=state,noise=noise,timing=timing)
        elapsed=(time.perf_counter()-before)*1000
        assert actions.shape==(cfg.n_suffix,cfg.max_action_dim) and np.isfinite(actions).all()
        if iteration==0:np.save(out/'actions.npy',actions)
        if iteration>=3:
            stats=model.last_stats()
            row={name:float(getattr(stats,name)) for name,_ in stats._fields_}
            row['python_wall_ms']=elapsed
            row['unaccounted_ms']=row['ms_total']-row['ms_vision']-row['ms_inference']
            rows.append(row)
    if cudart is not None:
        assert cudart.cudaDeviceSynchronize()==0
        assert cudart.cudaProfilerStop()==0
    resolved={name:getattr(cfg,name) for name,_ in cfg._fields_}
summary={name:{'mean':float(np.mean([r[name] for r in rows])),'p50':float(np.percentile([r[name] for r in rows],50)),'p90':float(np.percentile([r[name] for r in rows],90))} for name in rows[0]}
if args.mode=='none':
    summary['ms_prefill']=None;summary['ms_denoise']=None
result=dict(model=args.model,variant=args.variant,mode=args.mode,warmup=3,reps=20,views=2,size=512,token_ids=tokens,noise='vla-bench fixed formula',resolved=resolved,samples=rows,summary=summary,limitations=['PHASE graph/padding differ from defaults','vision includes connector/preprocess/copies','prefill/denoise do not account for every overhead','prefill includes KV graph, denoise includes cross-KV reprojection','not CUDA kernel-only timing'])
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))

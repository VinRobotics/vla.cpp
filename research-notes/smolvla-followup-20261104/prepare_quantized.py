"""Prepare stock GGUF quantizations on CPU; no inference or SR claims."""
from pathlib import Path
import os,sys,json,subprocess,hashlib
root=Path(__file__).resolve().parents[2]
vendor=root/'build-cuda/_deps/llama-src/gguf-py'
sys.path.insert(0,str(vendor));sys.path.insert(0,str(root/'scripts'))
import numpy as np
import gguf
import quantize_gguf as q
source=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
out=root/'research-notes/generated-models'
out.mkdir(exist_ok=True)
logs=Path(__file__).resolve().parent/'quantization'
logs.mkdir(exist_ok=True)
manifest=[]
reader=gguf.GGUFReader(str(source))
env=dict(os.environ,PYTHONPATH=str(vendor),CUDA_VISIBLE_DEVICES='')
for kind in ['Q8_0','Q5_0','Q4_0']:
    for vision in [False,True]:
        name='smolvla-'+kind.lower()+('-vision' if vision else '-lm')
        destination=out/(name+'.gguf')
        if destination.exists():raise SystemExit('Refuse overwrite: '+str(destination))
        command=[str(root/'.venv/bin/python'),str(root/'scripts/quantize_gguf.py'),'--in',str(source),'--out',str(destination),'--type',kind]+(['--vision'] if vision else [])
        record=dict(name=name,type=kind,vision=vision,command=command,path=str(destination),status='running',inference_validated=False)
        manifest.append(record)
        (logs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        with (logs/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        record['exit']=result.returncode
        if result.returncode:
            record['status']='export_failed'
        else:
            exported=gguf.GGUFReader(str(destination))
            by_name={t.name:t for t in exported.tensors}
            assert len(by_name)==len(reader.tensors)
            changed=[];preserved=0;expert=0
            skip=q.SKIP if vision else q.SKIP+q.SKIP_VISION
            for original in reader.tensors:
                tensor=by_name[original.name]
                assert np.array_equal(original.shape,tensor.shape),original.name
                eligible=q.eligible(original.name,original.shape,kind,skip) and original.tensor_type in (gguf.GGMLQuantizationType.BF16,gguf.GGMLQuantizationType.F32)
                if eligible:
                    assert tensor.tensor_type==getattr(gguf.GGMLQuantizationType,kind),original.name
                    changed.append(original.name)
                else:
                    assert tensor.tensor_type==original.tensor_type,original.name
                    assert np.array_equal(original.data.view(np.uint8),tensor.data.view(np.uint8)),original.name
                    preserved+=1
                if original.name.startswith('aex.'):
                    assert not eligible,original.name
                    expert+=1
            assert expert>0
            for key,field in reader.fields.items():
                if key.startswith('GGUF.'):continue
                assert exported.fields[key].contents()==field.contents(),key
            record.update(status='export_and_integrity_ok',bytes=destination.stat().st_size,quantized_tensors=len(changed),preserved_tensors=preserved,expert_tensors_preserved=expert)
            digest=hashlib.sha256()
            with destination.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
            record['sha256']=digest.hexdigest()
            del exported,by_name
        (logs/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,record['status'],record.get('bytes'),flush=True)

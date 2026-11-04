"""Structured depth ablation for this SmolVLA GGUF; not a CLP implementation.

Removes blocks and reindexes retained tensor names, preserving raw weights.
VLM/action expert are coupled because this loader shares their layer count/KV mapping.
"""
from pathlib import Path
import argparse,json,re,sys,hashlib
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'build-cuda/_deps/llama-src/gguf-py'))
sys.path.insert(0,str(root/'scripts'))
import gguf
import numpy as np
from gguf_common import copy_kv,copy_tensor
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--input',required=True)
ap.add_argument('--output',required=True)
ap.add_argument('--vision-drop',default='')
ap.add_argument('--coupled-drop',default='')
args=ap.parse_args()
source=Path(args.input);destination=Path(args.output)
if destination.resolve()==source.resolve() or destination.exists():raise SystemExit('Refuse overwrite')
reader=gguf.GGUFReader(str(source))
if reader.fields['general.architecture'].contents()!='smolvla':raise SystemExit('SmolVLA only')
counts={'vit':int(reader.fields['smolvla.vit_layers'].contents()),'vlm':int(reader.fields['smolvla.n_layers'].contents())}
counts['aex']=counts['vlm']
def parse(text,count):
    values=set(int(v) for v in text.split(',') if v.strip())
    if not values.issubset(range(count)) or len(values)>=count:raise SystemExit('Invalid layer removal set')
    return values
drops={'vit':parse(args.vision_drop,counts['vit']),'vlm':parse(args.coupled_drop,counts['vlm'])}
drops['aex']=drops['vlm']
if not any(drops.values()):raise SystemExit('No blocks selected for removal')
maps={module:{old:new for new,old in enumerate(i for i in range(counts[module]) if i not in drops[module])} for module in counts}
every=int(reader.fields['smolvla.self_attn_every_n_layers'].contents())
if every>0 and any((old%every==0)!=(new%every==0) for old,new in maps['aex'].items()):
    raise SystemExit('Reindexing would change SA/CA block type; remove compatible blocks instead')
pattern=re.compile(r'^(vit|vlm|aex)\.blk\.(\d+)\.(.+)$')
plan=[];removed=[]
for tensor in reader.tensors:
    match=pattern.match(tensor.name)
    if match:
        module,index,suffix=match[1],int(match[2]),match[3]
        if index in drops[module]:removed.append(tensor.name);continue
        name=f'{module}.blk.{maps[module][index]}.{suffix}'
    else:name=tensor.name
    plan.append((tensor,name))
assert len({name for _,name in plan})==len(plan)
writer=gguf.GGUFWriter(str(destination),'smolvla')
copy_kv(reader,writer,skip=('smolvla.vit_layers','smolvla.n_layers'))
writer.add_uint32('smolvla.vit_layers',len(maps['vit']))
writer.add_uint32('smolvla.n_layers',len(maps['vlm']))
for tensor,name in plan:
    # copy_tensor preserves dtype/shape; expose the new name without changing source.
    renamed=tensor._replace(name=name)
    copy_tensor(writer,renamed)
writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
exported=gguf.GGUFReader(str(destination));by_name={t.name:t for t in exported.tensors}
assert len(by_name)==len(plan)
for tensor,name in plan:
    result=by_name[name]
    assert tensor.tensor_type==result.tensor_type,name
    assert np.array_equal(tensor.shape,result.shape),name
    assert np.array_equal(tensor.data.view(np.uint8),result.data.view(np.uint8)),name
for key,field in reader.fields.items():
    if key.startswith('GGUF.') or key in ('smolvla.vit_layers','smolvla.n_layers'):continue
    assert exported.fields[key].contents()==field.contents(),key
assert exported.fields['smolvla.vit_layers'].contents()==len(maps['vit'])
assert exported.fields['smolvla.n_layers'].contents()==len(maps['vlm'])
digest=hashlib.sha256()
with destination.open('rb') as f:
    for data in iter(lambda:f.read(8*1024*1024),b''):digest.update(data)
report=dict(method='structured depth ablation; no CKA/calibration/finetuning',source=str(source),output=str(destination),drop={k:sorted(v) for k,v in drops.items()},retained_layer_maps=maps,source_tensors=len(reader.tensors),removed_tensors=removed,retained_tensors=len(plan),bytes=destination.stat().st_size,sha256=digest.hexdigest(),raw_weights_preserved=True,inference_validated=False,latency_measured=False,sr_measured=False)
destination.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['method','bytes','retained_tensors','sha256','inference_validated']},indent=2))

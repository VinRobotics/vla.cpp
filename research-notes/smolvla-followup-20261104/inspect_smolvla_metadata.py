"""Read GGUF metadata only; never read tensor payloads."""
from pathlib import Path
import struct,json
p=Path('/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf')
formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
with p.open('rb') as f:
    def number(fmt):
        return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
    def string():
        return f.read(number('Q')).decode('utf-8')
    def value(kind):
        if kind in formats:return number(formats[kind])
        if kind==8:return string()
        if kind==9:
            element,count=number('I'),number('Q')
            return [value(element) for _ in range(count)]
        raise ValueError(kind)
    assert f.read(4)==b'GGUF'
    version=number('I');tensors=number('Q');count=number('Q')
    metadata={}
    for _ in range(count):
        key=string();metadata[key]=value(number('I'))
filtered={k:v for k,v in metadata.items() if k.startswith('smolvla.') or k in ['general.architecture','general.name']}
result=dict(file=str(p),gguf_version=version,tensor_count=tensors,metadata=filtered)
Path(__file__).with_name('smolvla-metadata.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

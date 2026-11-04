from pathlib import Path
import json,re
import numpy as np
p=Path(__file__).resolve().parent/'runtime-screen'
def read(name):
    lines=(p/(name+'.out')).read_text().splitlines()
    index=next(i for i,s in enumerate(lines) if s.startswith('action_len='))
    return np.asarray([float(s) for s in lines[index+1:]],dtype=np.float32).reshape(50,32)
reference=read('defaults')
rows=[]
for f in sorted(p.glob('*.out')):
    name=f.stem
    actions=read(name)
    difference=actions.astype(np.float64)-reference.astype(np.float64)
    match=re.search(r'mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms',(p/(name+'.log')).read_text())
    rows.append(dict(config=name,screen_mean_ms=float(match[1]),p50_ms=float(match[2]),p90_ms=float(match[3]),max_abs=float(np.abs(difference).max()),rmse=float(np.sqrt(np.mean(difference*difference))),bit_differences=int(np.count_nonzero(actions.view(np.uint32)!=reference.view(np.uint32))),rmse_real7=float(np.sqrt(np.mean(difference[:,:7]**2))),max_abs_real7=float(np.abs(difference[:,:7]).max()),rmse_scope_legacy='all 50x32 padded output elements',primary_scope='50x7 real action elements (exclude zero padding)'))
(p/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Saved',len(rows),'runtime screen results; RMSE_real7 excludes padding.')

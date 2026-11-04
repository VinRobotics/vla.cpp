"""Map PHASE GPU kernels using observed vision launches + validated K/V copy boundary.

Handles direct launches and CUDA graph replay. Reports target GPU active kernel
union and kernel span, never equates these with end-to-end wall latency.
"""
from pathlib import Path
import argparse,json,sqlite3,statistics
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('sqlite');ap.add_argument('--workload',required=True);args=ap.parse_args()
p=Path(args.sqlite);work=json.loads(Path(args.workload).read_text());cfg=work['resolved'];reps=work['reps'];views=work['views']
assert work['mode']=='phase' and views==2
prefix=views*int(cfg['n_img'])+len(work['token_ids'])+int(cfg['n_state'])
copy_bytes=prefix*int(cfg['head_dim'])*int(cfg['n_kv_heads'])*4
copies_per_predict=2*int(cfg['n_layers'])
db=sqlite3.connect(p);db.row_factory=sqlite3.Row
kernels=list(db.execute('SELECT rowid AS rid,start,end,graphId,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'))
kind=db.execute("SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE label='Device-to-Device'").fetchone()[0]
copies=list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=? AND bytes=? ORDER BY start',(kind,copy_bytes)))
assert len(copies)==reps*copies_per_predict,(len(copies),reps*copies_per_predict)
launches=list(db.execute('SELECT graphId,correlationId,min(start) AS start,max(end) AS end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL AND graphId!=0 GROUP BY graphId,correlationId ORDER BY min(start)'))
assert len(launches)>=2 and launches[0]['graphId']==launches[1]['graphId']
vision_id=launches[0]['graphId'];vision=[r for r in launches if r['graphId']==vision_id]
assert len(vision)==reps*views,(len(vision),reps*views)

def stats(items):
    assert items
    intervals=[]
    for k in sorted(items,key=lambda r:r['start']):
        if intervals and k['start']<=intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],k['end'])
        else:intervals.append([k['start'],k['end']])
    return dict(kernel_count=len(items),gpu_active_kernel_ms=sum(end-start for start,end in intervals)/1e6,gpu_kernel_span_ms=(max(k['end'] for k in items)-min(k['start'] for k in items))/1e6,direct_kernel_count=sum(k['graphId'] is None or k['graphId']==0 for k in items))
assigned=set();samples=[]
for i in range(reps):
    pair=vision[i*views:(i+1)*views]
    start=pair[0]['start'];end=vision[(i+1)*views]['start'] if i+1<reps else max(k['end'] for k in kernels)+1
    chunk=[k for k in kernels if k['start']>=start and k['end']<=end]
    transfer=[r for r in copies if r['start']>=start and r['end']<=end]
    assert len(transfer)==copies_per_predict,(i,len(transfer),copies_per_predict)
    first_copy=min(r['start'] for r in transfer);last_copy=max(r['end'] for r in transfer)
    vision_k=[k for k in chunk if k['graphId']==vision_id and k['correlationId'] in [x['correlationId'] for x in pair]]
    prefill=[k for k in chunk if k['start']>=pair[-1]['end'] and k['end']<=first_copy]
    denoise=[k for k in chunk if k['start']>=last_copy]
    groups={'vision_plus_connector':vision_k,'kv_prefill':prefill,'denoise_with_cross_kv':denoise}
    ids=[k['rid'] for group in groups.values() for k in group]
    assert len(ids)==len(set(ids))
    assert set(ids)=={k['rid'] for k in chunk},f'Unassigned/ambiguous kernel at prediction {i}'
    assigned.update(ids)
    result={'prediction':i,'kv_copy_count':len(transfer),'kv_copy_span_ms':(last_copy-first_copy)/1e6}
    for name,group in groups.items():
        result[name]=stats(group)
        if name=='vision_plus_connector':
            # Two view graph spans do not include the CPU/copy gap between views.
            result[name]['gpu_kernel_span_ms']=sum((v['end']-v['start'])/1e6 for v in pair)
    samples.append(result)
assert assigned=={k['rid'] for k in kernels},'Some trace kernels lie outside mapped predictions'
summary={}
for name in ['vision_plus_connector','kv_prefill','denoise_with_cross_kv']:
    summary[name]={}
    for metric in ['gpu_active_kernel_ms','gpu_kernel_span_ms','kernel_count','direct_kernel_count']:
        values=sorted(r[name][metric] for r in samples)
        def pct(q):
            index=q*(len(values)-1);lo=int(index);hi=min(lo+1,len(values)-1);return values[lo]+(values[hi]-values[lo])*(index-lo)
        summary[name][metric]=dict(mean=statistics.mean(values),p50=pct(.5),p90=pct(.9))
result=dict(trace=str(p),workload=args.workload,reps=reps,prefix_tokens=prefix,kv_copy_bytes=copy_bytes,kv_copies_per_predict=copies_per_predict,matched_kernel_count=len(assigned),summary=summary,samples=samples,attribution='source phase K/V tensor-copy boundary, validated byte size/count and full kernel coverage',limitations=['GPU active time excludes host setup, inter-kernel/phase gaps and transfer cost','GPU span can include CPU launch gaps for direct kernels','vision excludes preprocessing/inter-view transfers','prefill is cache-producing graph; denoise includes cross-KV reprojection','phase graph/padding differ from normal throughput; node tracing overhead'])
p.with_suffix('.boundaries.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))

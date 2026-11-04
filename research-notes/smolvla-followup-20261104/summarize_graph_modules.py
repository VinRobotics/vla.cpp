"""Attribute GPU graph launches by verified source call order, not kernel-name guesses.

NONE: 20 x [vision view0, vision view1, combined inference].
PHASE: 20 x [vision view0, vision view1, KV prefill, denoise].
Reports first-to-last kernel GPU spans; excludes CPU setup and inter-graph copies/gaps.
"""
from pathlib import Path
import argparse,json,sqlite3,statistics
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('sqlite');ap.add_argument('--mode',choices=['none','phase'],required=True);ap.add_argument('--predictions',type=int,default=20);args=ap.parse_args()
p=Path(args.sqlite);db=sqlite3.connect(p);db.row_factory=sqlite3.Row
rows=list(db.execute('SELECT graphId,correlationId,count(*) AS kernels,min(start) AS start,max(end) AS end,sum(end-start) AS kernel_sum FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId!=0 GROUP BY graphId,correlationId ORDER BY min(start)'))
per=3 if args.mode=='none' else 4
if len(rows)!=args.predictions*per:raise SystemExit(f'Unexpected launch pattern: {len(rows)} vs {args.predictions*per}; do not assume module attribution')
samples=[]
for i in range(args.predictions):
    r=rows[i*per:(i+1)*per]
    if r[0]['graphId']!=r[1]['graphId']:raise SystemExit('First two launches are not the same vision graph')
    sample={'vision_plus_connector_gpu_span_ms':sum((x['end']-x['start'])/1e6 for x in r[:2]),'vision_kernel_sum_ms':sum(x['kernel_sum']/1e6 for x in r[:2])}
    labels=['combined_inference'] if args.mode=='none' else ['kv_prefill','denoise_with_cross_kv_projection']
    for row,label in zip(r[2:],labels):
        sample[label+'_gpu_span_ms']=(row['end']-row['start'])/1e6
        sample[label+'_kernel_sum_ms']=row['kernel_sum']/1e6
    samples.append(sample)
extra=db.execute('SELECT count(*),coalesce(sum(end-start),0) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=0').fetchone()
summary={}
for key in samples[0]:
    values=sorted(r[key] for r in samples)
    def pct(q):
        index=q*(len(values)-1);lo=int(index);hi=min(lo+1,len(values)-1);return values[lo]+(values[hi]-values[lo])*(index-lo)
    summary[key]=dict(mean=statistics.mean(values),p50=pct(.5),p90=pct(.9))
result=dict(mode=args.mode,predictions=args.predictions,graph_launches=len(rows),summary=summary,samples=samples,extra_non_graph_kernels={'count':extra[0],'summed_ms':extra[1]/1e6},attribution='source predict_impl order, guarded by launch cardinality and paired vision graph IDs',limitations=['GPU first-to-last kernel spans, not full CPU+GPU phase wall time','vision sums two graph launches; excludes preprocessing and inter-view copies/gaps','KV prefill is cache-producing graph, not arbitrary standalone full VLM','denoise includes cross-KV reprojection and action projections','PHASE graph/padding differ from default throughput graph'])
p.with_suffix('.modules.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))

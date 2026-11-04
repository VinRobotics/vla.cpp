"""Summarize a CUDA graph NODE trace; not a semantic module profiler."""
from pathlib import Path
import argparse,collections,csv,json,re,sqlite3
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('sqlite');args=ap.parse_args()
p=Path(args.sqlite);db=sqlite3.connect(p);db.row_factory=sqlite3.Row
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables and db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0]:
    raise SystemExit('Whole-graph aggregate events found: use --cuda-graph-trace=node before kernel attribution')
if 'DIAGNOSTIC_EVENT' in tables:
    loss=[r['text'] for r in db.execute('SELECT text FROM DIAGNOSTIC_EVENT') if any(k in r['text'].lower() for k in ['lost','drop'])]
    if loss:raise SystemExit('Inspect trace completeness diagnostics before analysis: '+str(loss))
names={r['id']:r['value'] for r in db.execute('SELECT * FROM StringIds')}
rows=list(db.execute('SELECT start,end,deviceId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'))
if not rows:raise SystemExit('No kernel events. Verify node tracing/capture.')
if len({r['deviceId'] for r in rows})!=1:raise SystemExit('Expected one GPU; analyze each device separately')
if any(r['end']<r['start'] for r in rows):raise SystemExit('Invalid kernel timestamps')
by_kernel=collections.defaultdict(lambda:[0,0])
by_group=collections.defaultdict(lambda:[0,0])
def group(name):
    lower=name.lower()
    if any(k in lower for k in ['flash_attn','fmha','attn_exact','k_attn','attention']):return 'attention_named_kernels_including_layout'
    if any(k in lower for k in ['gemm','mul_mat','wmma','mma_sync']):return 'GEMM_or_matmul_including_attention_products'
    if any(k in lower for k in ['softmax','soft_max','rope']):return 'softmax_or_RoPE_not_all_attention'
    if any(k in lower for k in ['norm','rms','ln_affine','ln_modulate']):return 'norm_or_fused_norm'
    return 'other_unclassified'
intervals=[]
for r in rows:
    name=names[r['demangledName']];duration=r['end']-r['start']
    by_kernel[name][0]+=1;by_kernel[name][1]+=duration
    by_group[group(name)][0]+=1;by_group[group(name)][1]+=duration
    intervals.append((r['start'],r['end']))
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
for table in ['CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET']:
    if table in tables:intervals.extend((r['start'],r['end']) for r in db.execute('SELECT start,end FROM '+table))
def merged(sequence):
    output=[]
    for start,end in sorted(sequence):
        if output and start<=output[-1][1]:output[-1][1]=max(output[-1][1],end)
        else:output.append([start,end])
    return output
union=merged(intervals);span=union[-1][1]-union[0][0];busy=sum(end-start for start,end in union)
gaps=[union[i+1][0]-union[i][1] for i in range(len(union)-1)]
total=sum(v[1] for v in by_kernel.values())
summary=dict(trace=str(p),kernel_events=len(rows),summed_kernel_ms=total/1e6,groups={k:{'calls':v[0],'summed_ms':v[1]/1e6,'fraction_of_summed_kernel_time':v[1]/total} for k,v in by_group.items()},gpu_activity_span_ms=span/1e6,gpu_activity_union_ms=busy/1e6,gap_ms=(span-busy)/1e6,gap_fraction=(span-busy)/span,max_gap_ms=max(gaps,default=0)/1e6,gaps_over_100us=sum(g>100000 for g in gaps),scope='target process activities only: first to last GPU activity within captured 20 predictions; excludes edge gaps and other processes',limitations=['summed kernel durations may overlap; use union for busy fraction','name-based categories are not exact module attribution','GEMM includes attention matmuls','node tracing overhead affects timings','gaps do not prove whole-GPU idle or a CPU bottleneck'])
p.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with p.with_suffix('.kernels.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['kernel','calls','summed_ms'])
    for name,(calls,duration) in sorted(by_kernel.items(),key=lambda item:item[1][1],reverse=True):writer.writerow([name,calls,duration/1e6])
print(json.dumps(summary,indent=2))

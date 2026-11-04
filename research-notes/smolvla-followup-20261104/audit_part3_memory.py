from pathlib import Path
import collections,json,sqlite3
base=Path(__file__).resolve().parent;out=base/'part3-runtime'
children=json.loads((out/'memory-runs.json').read_text())
assert len(children)==30 and all(r['exit']==0 for r in children)
con=sqlite3.connect(out/'runtime.sqlite');con.row_factory=sqlite3.Row
kinds={r['label']:r['id'] for r in con.execute('SELECT * FROM ENUM_CUDA_MEM_KIND')}
ops={r['label']:r['id'] for r in con.execute('SELECT * FROM ENUM_CUDA_DEV_MEM_EVENT_OPER')}
pids={r['globalPid']:r['pid'] for r in con.execute('SELECT * FROM PROCESSES')}
expected={r['pid']:r for r in children}
live=collections.defaultdict(int);peak=collections.defaultdict(int);ledger={};unmatched=[]
for event in con.execute('SELECT rowid,* FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start,rowid'):
    if event['memKind'] not in (kinds['Device'],kinds['Device Static']):continue
    pid=pids.get(event['globalPid'])
    if pid not in expected:raise RuntimeError('Unexpected/unmapped CUDA process: '+str(pid))
    key=(pid,event['deviceId']);address=(*key,event['address'])
    if event['memoryOperationType']==ops['Allocation']:
        if address in ledger:raise RuntimeError('Duplicate active address')
        ledger[address]=event['bytes'];delta=event['bytes']
    elif event['memoryOperationType']==ops['Deallocation']:
        if ledger.pop(address,None)!=event['bytes']:unmatched.append(dict(event))
        delta=-event['bytes']
    else:raise RuntimeError('Unknown operation')
    live[key]+=delta
    if live[key]<0:raise RuntimeError('Negative live memory')
    peak[key]=max(peak[key],live[key])
assert not unmatched
assert {key[0] for key in peak}==set(expected),'Missing CUDA memory capture for some child process'
loss=[r['text'] for r in con.execute('SELECT text FROM DIAGNOSTIC_EVENT') if any(x in r['text'].lower() for x in ['lost','drop'])]
assert not loss,loss
rows=[]
for (pid,device),amount in peak.items():
    record=expected[pid]
    rows.append(dict(name=record['name'],trial=record['trial'],pid=pid,device=device,peak_bytes=amount,peak_MiB=amount/1048576,live_at_trace_end_bytes=live[(pid,device)]))
summary={}
for name in sorted({r['name'] for r in rows}):
    group=[r for r in rows if r['name']==name];assert len(group)==3
    summary[name]=dict(peak_MiB=max(r['peak_MiB'] for r in group),runs=group)
# Cross-check the same defaults workload against the earlier independent capture.
old=json.loads((base.parent/'smolvla-vram-20261104/memory-summary.json').read_text())
reference=max(g['peak_MiB'] for r in old for g in r['process_device_groups'])
summary['_control_comparison']=dict(previous_defaults_peak_MiB=reference,current_defaults_peak_MiB=summary['defaults']['peak_MiB'],difference_MiB=summary['defaults']['peak_MiB']-reference)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v['peak_MiB'] for k,v in summary.items() if not k.startswith('_')},indent=2))

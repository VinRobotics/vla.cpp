"""Extract per-process CUDA allocation high-water marks from nsys SQLite exports.

Run: python3 research-notes/smolvla-vram-20261104/summarize_memory.py
Includes Device and Device Static events, excludes host/pinned memory.
This is not an NVML resident-VRAM reading and excludes unreported driver overhead.
"""
from pathlib import Path
import collections
import csv
import json
import sqlite3

out = Path(__file__).resolve().parent
summaries = []
for path in sorted(out.glob('baseline-memory-*.sqlite')):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        kinds = {r['label']: r['id'] for r in db.execute('SELECT * FROM ENUM_CUDA_MEM_KIND')}
        ops = {r['label']: r['id'] for r in db.execute('SELECT * FROM ENUM_CUDA_DEV_MEM_EVENT_OPER')}
        events = db.execute('SELECT rowid, * FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start,rowid').fetchall()
        live = collections.defaultdict(int)
        peak = collections.defaultdict(int)
        counts = collections.Counter()
        ledger = {}
        unmatched = []
        negative = []
        rows = []
        for event in events:
            if event['memKind'] not in (kinds['Device'], kinds['Device Static']):
                continue
            group = (event['globalPid'], event['deviceId'])
            address = (*group, event['address'])
            operation = event['memoryOperationType']
            if operation == ops['Allocation']:
                if address in ledger:
                    raise ValueError(f'duplicate active allocation address: {address}')
                ledger[address] = event['bytes']
                delta = event['bytes']
                counts['allocations'] += 1
            elif operation == ops['Deallocation']:
                previous = ledger.pop(address, None)
                if previous != event['bytes']:
                    unmatched.append(dict(event))
                delta = -event['bytes']
                counts['deallocations'] += 1
            else:
                raise ValueError(f'unknown memory operation: {operation}')
            live[group] += delta
            peak[group] = max(peak[group], live[group])
            if live[group] < 0:
                negative.append(dict(event))
            rows.append([event['start'], *group, event['address'], event['memKind'], operation, event['bytes'], live[group]])
        vmm = [dict(r) for r in db.execute("""
            SELECT s.value AS api, e.start, e.bytes, e.memoryOperationType
            FROM CUDA_GPU_MEMORY_USAGE_EVENTS e
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON e.correlationId=r.correlationId
            JOIN StringIds s ON r.nameId=s.id
            WHERE s.value IN ('cuMemSetAccess','cuMemMap','cuMemUnmap')
            ORDER BY e.start
        """)]
        diagnostics = [dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        lost = [r for r in diagnostics if 'lost' in r['text'].lower() or 'drop' in r['text'].lower()]
        if unmatched or negative or lost or len(peak) != 1:
            raise ValueError(f'check trace completeness for {path}: unmatched={unmatched}, negative={negative}, lost={lost}, groups={list(peak)}')
        with path.with_suffix('.allocations.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp_ns', 'globalPid', 'deviceId', 'address', 'memKind', 'operation', 'bytes', 'live_bytes'])
            writer.writerows(rows)
        summary = {
            'trace': path.name,
            'metric': 'peak traced CUDA allocations (Device + Device Static), not NVML resident VRAM',
            'process_device_groups': [dict(globalPid=k[0], deviceId=k[1], peak_bytes=v, peak_MiB=v/1048576, live_at_trace_end_bytes=live[k]) for k,v in peak.items()],
            'counts': dict(counts),
            'vmm_events': vmm,
            'unmatched_frees': len(unmatched),
            'negative_live_events': len(negative),
            'lost_or_dropped_diagnostics': lost,
            'allocations_still_live_at_trace_end': len(ledger),
        }
        summaries.append(summary)
(out/'memory-summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
for summary in summaries:
    print(summary['trace'], summary['process_device_groups'])
if summaries:
    print('Peak over all processes (MiB):', max(g['peak_MiB'] for s in summaries for g in s['process_device_groups']))

"""CPU-only audit of existing Task 5 evidence; never runs inference.

Usage (repo root): python3 research-notes/task5-profiling/audit_evidence.py
Outputs a NEW timestamped directory; all existing evidence is opened read-only.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import re
import sqlite3
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research-notes/smolvla-followup-20261104'
OUT = Path(__file__).parent / 'logs' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        assert end >= start
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def trace(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    names = dict(db.execute('SELECT id,value FROM StringIds'))
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables:
        assert db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE').fetchone()[0] == 0
    diagnostics = [r[0] for r in db.execute('SELECT text FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
    assert not any(re.search(r'lost|drop', s, re.I) for s in diagnostics), diagnostics
    kernels = list(db.execute('SELECT start,end,demangledName,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start'))
    assert kernels and len({r[3] for r in kernels}) == 1
    groups = collections.defaultdict(lambda: [0, 0])
    top = collections.defaultdict(lambda: [0, 0])
    for start, end, name_id, _ in kernels:
        name = names[name_id]
        lower = name.lower()
        group = 'other'
        for label, keys in [('attention_named', ['flash_attn', 'fmha', 'attn_exact', 'k_attn', 'attention']),
                            ('GEMM_including_attention_products', ['gemm', 'mul_mat', 'wmma', 'mma_sync']),
                            ('softmax_or_RoPE', ['softmax', 'soft_max', 'rope']),
                            ('norm', ['norm', 'rms', 'ln_affine', 'ln_modulate'])]:
            if any(k in lower for k in keys):
                group = label
                break
        groups[group][0] += 1
        groups[group][1] += end-start
        top[name][0] += 1
        top[name][1] += end-start
    intervals = [(r[0], r[1]) for r in kernels]
    copies = []
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        copies = list(db.execute('SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'))
        intervals.extend((r[0], r[1]) for r in copies)
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        intervals.extend(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET'))
    merged = union(intervals)
    span = merged[-1][1]-merged[0][0]
    active = sum(e-s for s,e in merged)
    api = []
    if 'CUPTI_ACTIVITY_KIND_RUNTIME' in tables:
        api = [dict(name=names[n], calls=c, summed_ms=t/1e6) for n,c,t in db.execute(
            'SELECT nameId,count(*),sum(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY nameId ORDER BY sum(end-start) DESC')]
    result = dict(kernel_count=len(kernels), summed_kernel_ms=sum(e-s for s,e,_,_ in kernels)/1e6,
                  groups={k:dict(calls=v[0], summed_ms=v[1]/1e6) for k,v in groups.items()},
                  activity_span_ms=span/1e6, activity_union_ms=active/1e6,
                  gap_ms=(span-active)/1e6, gap_fraction=(span-active)/span,
                  max_gap_ms=max((merged[i+1][0]-merged[i][1] for i in range(len(merged)-1)), default=0)/1e6,
                  memcpy=dict(count=len(copies), bytes=sum(r[2] for r in copies), summed_ms=sum(r[1]-r[0] for r in copies)/1e6),
                  runtime_api=api, diagnostics=diagnostics,
                  completeness_warnings=[s for s in diagnostics if 'not all' in s.lower() or 'lost' in s.lower() or 'drop' in s.lower()],
                  top_kernels=[dict(name=k,calls=v[0],summed_ms=v[1]/1e6) for k,v in sorted(top.items(),key=lambda kv:kv[1][1],reverse=True)[:12]])
    db.close()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    runs_path = BASE/'runtime-full-pair/runs.json'
    runs = json.loads(runs_path.read_text())
    checked = []
    for r in runs:
        p = BASE/f"runtime-full-pair/{r['variant']}-{r['trial']}.log"
        text = p.read_text()
        match = re.search(r'min ([\d.]+) ms\s+mean ([\d.]+) ms\s+p50 ([\d.]+) ms\s+p90 ([\d.]+) ms\s+vision ([\d.]+) ms', text)
        assert match, p
        assert list(map(float, match.groups())) == [r[k] for k in ['min_ms','mean_ms','p50_ms','p90_ms','vision_ms']], p
        checked.append(p)
    benchmark = {name: {k: statistics.median(r[k] for r in runs if r['variant']==name)
                       for k in ['mean_ms','p50_ms','p90_ms','vision_ms']}
                 for name in ['defaults','flash-mm']}
    traces = {}
    for name in benchmark:
        path = BASE/f'kernel-traces/{name}.sqlite'
        traces[name] = trace(path)
        previous = json.loads(path.with_suffix('.summary.json').read_text())
        assert traces[name]['kernel_count'] == previous['kernel_events']
        assert abs(traces[name]['summed_kernel_ms']-previous['summed_kernel_ms']) < 1e-8
        assert abs(traces[name]['gap_fraction']-previous['gap_fraction']) < 1e-12
        checked.extend([path,path.with_suffix('.nsys-rep'),path.with_suffix('.summary.json')])
    phase = {}
    for name in benchmark:
        paths = sorted((BASE/'phase-suite').glob(name+'-*/result.json'))
        assert len(paths)==3
        data = [json.loads(p.read_text()) for p in paths]
        assert all(d['warmup']==3 and d['reps']==20 and d['mode']=='phase' for d in data)
        phase[name] = {k: statistics.mean(d['summary'][k]['mean'] for d in data)
                       for k in ['ms_total','ms_vision','ms_prefill','ms_denoise','unaccounted_ms']}
        checked.extend(paths)
    checked.extend([runs_path, BASE/'smolvla-metadata.json', BASE/'profile_phase.py',
                    BASE/'trace_kernels.py',ROOT/'src/models/smolvla.cpp',ROOT/'src/serving/vla-bench.cpp',
                    ROOT/'build-cuda/libvla.so', ROOT/'build-cuda/vla-bench'])
    hashes = {str(p.relative_to(ROOT)): hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in checked}
    result = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  scope='Offline reanalysis of historical evidence; no new GPU measurements',
                  benchmark_median_of_three_run_statistics=benchmark,
                  phase_mean_of_three_20_prediction_runs=phase,traces=traces,sha256=hashes)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'git-status.txt').write_text(subprocess.check_output(['git','--no-optional-locks','status','--short','--branch'],cwd=ROOT,text=True))
    print('PASS: six benchmark logs match JSON; both node traces match previous kernel totals/gaps.')
    print('Benchmark median of three run statistics:',json.dumps(benchmark))
    print('PHASE mean of 60 samples per configuration:',json.dumps(phase))
    for name,t in traces.items():
        print(name,'kernels',t['kernel_count'],'kernel ms',t['summed_kernel_ms'],'gap %',100*t['gap_fraction'])
        print('Top CUDA APIs:',json.dumps(t['runtime_api'][:6]))
        print('Trace completeness warnings:',json.dumps(t['completeness_warnings']))
    print('Full evidence:',OUT/'audit.json')


if __name__ == '__main__':
    main()

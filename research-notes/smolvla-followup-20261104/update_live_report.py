"""Update only a managed status block; preserve report text outside markers."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,re
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent

def load(relative,default=None):
    try:return json.loads((base/relative).read_text())
    except (OSError,ValueError):return default

screen=load('screen40-defaults/result.json',{})
bench=load('candidate-bench/selected.json',{})
checks={r['name']:r for r in load('candidate-checks/summary.json',[])}
memory=load('memory-all/summary.json',{})
smokes={r['name']:r for r in load('candidate-smokes/results.json',[])}
progress=load('candidate-smokes/progress.json',{})
pair=load('runtime-full-pair/selected.json',{})
now=datetime.now(timezone(timedelta(hours=7))).strftime('%d/%m/%Y %H:%M:%S UTC+07')
lines=['<!-- SMOLVLA_LIVE_START -->','## Hiện trạng mới nhất',f'Cập nhật: {now}. Khối này tổng hợp dữ liệu mới; các mục phía dưới giữ lịch sử số đo.','',f"- Defaults screening: **{screen.get('successes','?')}/40**; không phải official200.",'- CUDA tests: **13/15**, hai lỗi bit-exact còn mở. Dependency client còn7metadata incompatibilities.',f"- Smoke biến thể: **{progress.get('completed',0)}/{progress.get('expected',10)} cấu hình đã hoàn tất**, đang chạy `{progress.get('current','—')}`.",'- Server cũ5555 đã được người dùng dừng sau các lượt benchmark/trace; không gán lại điều kiện đo trước đó.','']
if pair:
    lines+=['### Runtime flags — full protocol','', '| Config | Mean / p50 / p90 (ms) | Peak traced allocation (MiB) | SR biến thể |','|---|---:|---:|---|']
    for name in ['defaults','flash-mm']:
        r=pair[name]
        peak=1912.4694557189941 if name=='defaults' else 1040.268383026123
        sr='38/40 screening' if name=='defaults' else (str(smokes[name].get('successes'))+'/2 smoke' if name in smokes else 'chưa đo')
        lines.append(f"| {name} | {r['mean_ms']} / {r['p50_ms']} / {r['p90_ms']} | {peak:.2f} | {sr} |")
    lines+=['','3process/config,3warmup+20reps,2views512²/16tokens. Best process theo mean; flash+mm giảmmean19,67% trong batch đối chiếu. Allocation không bao hết context/driver, không phải NVML resident VRAM.','']
if bench:
    lines+=['### GGUF — cùng runtime defaults','', '| Model | File MiB | p50 / p90 ms | Peak allocation MiB | Max abs real7 | Smoke task0 |','|---|---:|---:|---:|---:|---|']
    for name,r in bench.items():
        file=Path(r['model']);size=f'{file.stat().st_size/1048576:.2f}' if file.exists() else '—'
        peak=f"{memory[name]['peak_MiB']:.2f}" if name in memory else 'chưa xác nhận'
        delta=f"{checks[name]['max_abs_real7']:.7g}" if name in checks else '—'
        if name=='defaults':smoke='2/2 (lượt gốc)'
        elif name in smokes:
            s=smokes[name];smoke=f"{s.get('successes','?')}/2; {s['status']}"
        else:smoke='chưa hoàn tất'
        lines.append(f"| {name} | {size} | {r['p50_ms']} / {r['p90_ms']} | {peak} | {delta} | {smoke} |")
    lines+=['','Latency bảng GGUF thuộc candidate-bench, có control riêng (khác batch runtime-pair). Mỗi config3process×(3warmup+20reps). Max abs dùng harness6token/noise cố định, không phải input16token của benchmark. Quant-lm bao gồm connector; action expert289tensor giữ float. Smoke2episode không thay screening40/official200.','']
phase=[]
for name in ['defaults','flash-mm','smolvla-vision-spaced-10','smolvla-coupled-tail-30']:
    r=load(f'kernel-phase-traces/{name}.boundaries.json')
    if r:phase.append((name,r))
if phase:
    lines+=['### Phase — GPU active kernel time','', '| Config | Vision+connector ms | KV prefill ms | Denoise+crossKV ms |','|---|---:|---:|---:|']
    for name,r in phase:
        x=r['summary'];lines.append(f"| {name} | {x['vision_plus_connector']['gpu_active_kernel_ms']['mean']:.3f} | {x['kv_prefill']['gpu_active_kernel_ms']['mean']:.3f} | {x['denoise_with_cross_kv']['gpu_active_kernel_ms']['mean']:.3f} |")
    lines+=['','20predict sau3warmup/trace. Gán module bằng biênD2D KVcopy từ source và kiểm tra toàn bộ kernel không thiếu/trùng; có cả direct launches và graph replay. Đây là active GPU time, loại CPU/copy/gaps; phase dùng16token thực thay48slot padded nên không phải phân rã chính xác baseline throughput. Denoise chưa phải riêng action expert, vision chưa tách connector.','']
lines+=['### Còn cần thực hiện','', '- Hoàn tất smoke các biến thể; sau đó screening40 cho cấu hình được chọn.','- Baseline official10×20 đã xếp hàng tự chạy sau smoke (`official200-runner.log`); biến thể official chưa chạy.','- Chưa có CLP/CKA/calibration/finetuning; pruning hiện là depth ablation với lựa chọn tail/spaced.','- Chưa chốt cấu hình tốt nhất: tốc độ/memory và output synthetic không chứng minh giữ SR.','','Raw data: `research-notes/smolvla-followup-20261104/` — `screen40-defaults`, `candidate-bench`, `memory-all`, `candidate-checks`, `candidate-smokes`, `kernel-phase-traces`.','<!-- SMOLVLA_LIVE_END -->']
block='\n'.join(lines)+'\n'
p=root/'REPORT_SMOLVLA.md';text=p.read_text()
pattern=r'<!-- SMOLVLA_LIVE_START -->.*?<!-- SMOLVLA_LIVE_END -->\n?'
if re.search(pattern,text,flags=re.S):text=re.sub(pattern,lambda _:block,text,count=1,flags=re.S)
else:
    first,rest=text.split('\n',1);text=first+'\n\n'+block+'\n'+rest
p.write_text(text)
print('Updated managed report block:',now)

"""Reproduce derived tables; never synthesizes experimental results."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent

def read(name):
    return json.loads((DATA / name).read_text())

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    c = (p + z*z/(2*n))/(1+z*z/n)
    h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return c-h, c+h

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + ['| '+' | '.join(map(str,r))+' |' for r in rows])

out = ['# Phụ lục tính toán cho đủ 6 task',
'''Các bảng dưới đây là **tính toán từ số đo đã có** hoặc **dự toán có giả định**, không phải benchmark mới. Chạy lại bằng `python3 research-notes/smolvla-followup-20261104/complete_calculations.py` tại gốc repo. Kết quả screening đọc từ result.json hoàn tất, được đối chiếu CSV bằng audit_screening.py.

## Task 1 — tỷ lệ kiểm thử

13/15 = 86,67% test pass; 2/15 = 13,33% fail. Đây là tỷ lệ test, không phải độ chính xác model. CLI kiểm tra inference một lần; server kiểm tra transport/request; binding kiểm tra C ABI từ Python; predict_check kiểm tra input/noise cố định và output. Binding/server khớp bit với harness trên input kiểm tra. Không có phép tính nào thay thế việc sửa hai lỗi CUDA bit-exact.

## Task 2 — baseline và khả năng suy luận

Baseline gốc mean 112,0 ms, p50 111,8 ms, p90 112,8 ms; peak traced allocation 1912,47 MiB. Tần suất lý tưởng từ mean = 1000/112 = **8,93 chunk/s**; từ p90 = **8,87 chunk/s**. Đây là engine rate, chưa gồm simulator/network và không nhân 50 thành control Hz vì LIBERO dùng n_action_steps=1. Allocation bằng 31,13% của 6144 MiB vật lý; 4231,53 MiB còn lại chỉ là phép trừ danh nghĩa, chưa trừ context/driver hoặc process khác. Không thể tính resident VRAM chính xác từ allocation.

## Task 3 — mức giảm so control cùng batch

Giảm latency = 100×(1−T/T0), speedup = T0/T, giảm allocation = 100×(1−M/M0). Giá trị âm nghĩa là chậm hơn control. Runtime lấy từ bảng PART3; các số đầu vào đã làm tròn 0,1 ms nên kết quả chỉ có độ chính xác tương ứng.''']
# Parse reviewed runtime table, avoiding use of the older 5-rep screen.
text = (ROOT/'PART3_OPTIMIZATION.md').read_text()
rows=[]
for line in text.split('## Quantization')[0].splitlines():
    c=[x.strip() for x in line.split('|')[1:-1]]
    if len(c)==6 and c[0] not in ('Cấu hình','---'):
        try: t=float(c[2].split('/')[0]); m=float(c[3])
        except ValueError: continue
        rows.append([c[0],f'{100*(1-t/104):.2f}%',f'{104/t:.3f}×',f'{100*(1-m/1912.47):.2f}%'])
out.append(table(['Runtime','Giảm mean','Speedup','Giảm allocation'],rows))
bench=read('candidate-bench/selected.json'); mem=read('memory-all/summary.json')
rows=[]
for name,d in bench.items():
    if 'q' not in name: continue
    rows.append([name,f'{100*(1-d["mean_ms"]/bench["defaults"]["mean_ms"]):.2f}%',f'{100*(1-mem[name]["peak_MiB"]/mem["defaults"]["peak_MiB"]):.2f}%'])
out.append(table(['Quantization','Giảm mean so 113,7 ms','Giảm allocation'],rows))
out.append('''Q8-lm và Q8-vision là ứng viên để đánh giá giảm bộ nhớ; flash-mm là ứng viên ưu tiên latency. Chưa xác nhận giữ chất lượng. Đây là lựa chọn để đánh giá tiếp, chưa phải winner official200. Không cộng speedup quantization với flash-mm khi chưa đo cấu hình kết hợp.

## Task 4 — SR, độ bất định và dự toán official

Wilson 95% dùng z=1,959964, p=k/n; tâm=(p+z²/2n)/(1+z²/n), nửa rộng=z√(p(1−p)/n+z²/4n²)/(1+z²/n). Interval giả sử episode Bernoulli độc lập; tương quan giữa episode cùng task và chỉ một seed làm độ bất định thực tế có thể lớn hơn. Không dùng interval này để khẳng định non-inferiority.''')
rows=[]
for label,path in [('Defaults screening','screen40-defaults/result.json'),('Flash-mm screening','screen40-variants/flash-mm/result.json'),('Q8-lm screening','screen40-variants/q8-lm/result.json'),('Q8-vision screening','screen40-variants/q8-vision/result.json')]:
    d=read(path); n=d['completed_episodes']; k=d['successes']; assert n==40
    lo,hi=wilson(k,n)
    rows.append([label,f'{k}/{n}',f'{100*k/n:.1f}%',f'{100*lo:.1f}–{100*hi:.1f}%',f'{200*k/n:.0f}/200 (dự toán)'])
lo,hi=wilson(177,200)
rows.append(['Defaults official','177/200','88,5%',f'{100*lo:.1f}–{100*hi:.1f}%','đã đo'])
out.append(table(['Cấu hình','Thành công','SR đo','Wilson 95% tính','Quy đổi tuyến tính'],rows))
out.append('''Dự toán 175/200 cho flash-mm và 180/200 cho Q8-lm và Q8-vision chỉ là 200×SR screening, giả định cùng phân phối; không được đưa vào cột official đo thật. Defaults screening dự toán 190/200 nhưng official thực tế 177/200 cho thấy phép ngoại suy có thể lệch đáng kể. Q8-vision đã hoàn tất 36/40 = 90%; không suy SR từ RMSE hoặc smoke. Hai bản Q8 cùng số thành công nhưng không chứng minh tương đương chất lượng. Official của các biến thể vẫn chưa đo. Với 3 biến thể tối đa, ngân sách còn 600 episode; chọn 2 biến thể thì 400 episode. Thời gian chạy không thể tính chỉ từ engine latency vì số bước/task và simulator thay đổi.

## Task 5 — phân bổ GPU và mô hình dự báo

Từ GPU-active PHASE đã đo: defaults = 52,708+14,956+36,318 = 103,982 ms; flash-mm = 85,682 ms. Tỷ trọng defaults vision+connector/prefill/denoise tương ứng 50,69% / 14,38% / 34,93%. Tổng active giảm 17,60%; không thay thế speedup end-to-end vì khác graph/padding và loại khoảng trống. Connector riêng và attention riêng chưa đủ instrumentation để cho số đo độc lập.

Mô hình tuyến tính theo depth: vision 12→10 tiết kiệm tối đa xấp xỉ 52,708×2/12 = 8,785 ms active; tổng dự báo 95,197 ms, speedup 1,092×. Coupled VLM/expert 32→30 dự báo tiết kiệm (14,956+36,318)×2/32 = 3,205 ms; tổng 100,777 ms, speedup 1,032×. Giả định mỗi tầng cùng chi phí, fixed overhead không đổi; dùng toàn phase cho layer là xấp xỉ lạc quan. GPU-active đo thực tế vision-spaced giảm vision 17,02%; coupled giảm prefill 5,76% và denoise 4,92%.

Default-path nsys: kernel duration 2260,227→1733,185 ms trên 20 predict, giảm 23,32%. GEMM-name share 58,30→51,85%; gaps 10,54→9,13% (giảm 1,41 điểm phần trăm). GEMM có cả attention matmul; không thể tính riêng thời gian attention bằng phần bù GEMM. Gaps chỉ trong activity đã trace, không chứng minh toàn GPU idle. Raw và cơ chế attribution: ARCHITECTURE_AND_PROFILE.md.

## Task 6 — ba pruning đã implement và kiểm tra

`prune_depth.py` cắt block thật, reindex tensor và cập nhật metadata; giữ nguyên bytes của weight còn lại. Vision-tail bỏ 10,11; vision-spaced bỏ 4,8; coupled-tail bỏ 30,31 ở cả VLM/expert. Coupled giữ parity SA/CA; loader dùng chung số layer nên chưa prune VLM/expert độc lập. Đây là structured depth pruning không train, chưa phải CLP/CKA. Zero weight không giảm kích thước dense GEMM nên không chọn làm bằng chứng tăng tốc.''')
checks={d['name']:d for d in read('candidate-checks/summary.json')}
rows=[]
for d in read('pruning-summary.json'):
    name=d['name']; b=bench[name]; c=checks[name]
    rows.append([name,f'{d["bytes"]/2**20:.2f}',f'{b["mean_ms"]:.1f}',f'{b["p50_ms"]:.1f}/{b["p90_ms"]:.1f}',f'{100*(1-b["mean_ms"]/113.7):.2f}%',f'{mem[name]["peak_MiB"]:.2f}',f'{c["max_abs_real7"]:.6g}/{c["rmse_real7"]:.6g}','0/2'])
out.append(table(['Pruning','File MiB','Mean ms','p50/p90 ms','Giảm mean','Allocation MiB','Maxabs/RMSE real7','Smoke'],rows))
out.append('''Benchmark dùng control 113,7 ms cùng candidate batch. Smoke cả ba thất bại; không ngoại suy thành official 0/200. Wilson cho 0/2 có upper bound 65,76% theo mô hình Bernoulli, cho thấy mẫu rất nhỏ; vẫn đủ lý do chưa ưu tiên chạy official tốn kém. Export sidecar `inference_validated=false` là trạng thái lúc export; inference về sau đã được `candidate-checks/summary.json` xác nhận, không chỉnh lịch sử export.

Kết luận task 6: đã implement hai cách chọn layer vision (tail/spaced) và một coupled pruning, chứng minh giảm module active/end-to-end ở các trường hợp đã đo; chưa có phương pháp giữ SR đạt yêu cầu triển khai. Bước CKA cần activation calibration riêng và finetuning/đánh giá mới, không có phép tính thay thế được thí nghiệm đó.

## Bằng chứng và cách đọc

Tất cả đường dẫn dữ liệu tương đối dưới `research-notes/smolvla-followup-20261104/`: `candidate-bench/selected.json`, `memory-all/summary.json`, `candidate-checks/summary.json`, `candidate-smokes/results.json`, `pruning-summary.json`, `phase-suite/selected.json`, `kernel-phase-traces/*.boundaries.json`, `kernel-traces/*.summary.json`, các `screen40-*/result.json`. Bảng runtime và protocol: [PART3_OPTIMIZATION.md](PART3_OPTIMIZATION.md). Checklist và kết luận chính: [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md). Các link ở phụ lục này tính từ gốc repo.
''')
(ROOT/'CALCULATIONS_SMOLVLA.md').write_text('\n\n'.join(out))
print('Wrote CALCULATIONS_SMOLVLA.md')

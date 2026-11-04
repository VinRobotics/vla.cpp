# Phụ lục tính toán cho đủ 6 task

Các bảng dưới đây là **tính toán từ số đo đã có** hoặc **dự toán có giả định**, không phải benchmark mới. Chạy lại bằng `python3 research-notes/smolvla-followup-20261104/complete_calculations.py` tại gốc repo. Snapshot screening đọc từ result/progress tại lúc sinh tài liệu.

## Task 1 — tỷ lệ kiểm thử

13/15 = 86,67% test pass; 2/15 = 13,33% fail. Đây là tỷ lệ test, không phải độ chính xác model. CLI kiểm tra inference một lần; server kiểm tra transport/request; binding kiểm tra C ABI từ Python; predict_check kiểm tra input/noise cố định và output. Binding/server khớp bit với harness trên input kiểm tra. Không có phép tính nào thay thế việc sửa hai lỗi CUDA bit-exact.

## Task 2 — baseline và khả năng suy luận

Baseline gốc mean 112,0 ms, p50 111,8 ms, p90 112,8 ms; peak traced allocation 1912,47 MiB. Tần suất lý tưởng từ mean = 1000/112 = **8,93 chunk/s**; từ p90 = **8,87 chunk/s**. Đây là engine rate, chưa gồm simulator/network và không nhân 50 thành control Hz vì LIBERO dùng n_action_steps=1. Allocation bằng 31,13% của 6144 MiB vật lý; 4231,53 MiB còn lại chỉ là phép trừ danh nghĩa, chưa trừ context/driver hoặc process khác. Không thể tính resident VRAM chính xác từ allocation.

## Task 3 — mức giảm so control cùng batch

Giảm latency = 100×(1−T/T0), speedup = T0/T, giảm allocation = 100×(1−M/M0). Giá trị âm nghĩa là chậm hơn control. Runtime lấy từ bảng PART3; các số đầu vào đã làm tròn 0,1 ms nên kết quả chỉ có độ chính xác tương ứng.

| Runtime | Giảm mean | Speedup | Giảm allocation |
| --- | --- | --- | --- |
| bf16-flash | -7.02% | 0.934× | 45.40% |
| bf16 | -18.37% | 0.845× | 40.69% |
| defaults | 0.00% | 1.000× | 0.00% |
| f16-flash | -4.90% | 0.953× | 45.40% |
| f16 | -21.06% | 0.826× | 41.47% |
| flash-mm | 17.79% | 1.216× | 45.61% |
| flash | 11.73% | 1.133× | 4.92% |
| mm-default | 6.35% | 1.068× | 40.69% |
| steps-3 | 23.85% | 1.313× | 0.03% |
| steps-5 | 17.12% | 1.206× | 0.02% |

| Quantization | Giảm mean so 113,7 ms | Giảm allocation |
| --- | --- | --- |
| smolvla-q4_0-lm | 3.25% | 54.77% |
| smolvla-q4_0-vision | 3.52% | 61.84% |
| smolvla-q5_0-lm | 2.73% | 52.73% |
| smolvla-q5_0-vision | 2.02% | 59.28% |
| smolvla-q8_0-lm | 3.69% | 46.63% |
| smolvla-q8_0-vision | 3.08% | 51.59% |

Q8-lm là ứng viên ưu tiên khi cần giảm bộ nhớ và giữ chất lượng; flash-mm là ứng viên ưu tiên latency. Đây là lựa chọn để đánh giá tiếp, chưa phải winner official200. Không cộng speedup quantization với flash-mm khi chưa đo cấu hình kết hợp.

## Task 4 — SR, độ bất định và dự toán official

Wilson 95% dùng z=1,959964, p=k/n; tâm=(p+z²/2n)/(1+z²/n), nửa rộng=z√(p(1−p)/n+z²/4n²)/(1+z²/n). Interval giả sử episode Bernoulli độc lập; tương quan giữa episode cùng task và chỉ một seed làm độ bất định thực tế có thể lớn hơn. Không dùng interval này để khẳng định non-inferiority.

| Cấu hình | Thành công | SR đo | Wilson 95% tính | Quy đổi tuyến tính |
| --- | --- | --- | --- | --- |
| Defaults screening | 38/40 | 95.0% | 83.5–98.6% | 190/200 (dự toán) |
| Flash-mm screening | 35/40 | 87.5% | 73.9–94.5% | 175/200 (dự toán) |
| Q8-lm screening | 36/40 | 90.0% | 76.9–96.0% | 180/200 (dự toán) |
| Defaults official | 177/200 | 88,5% | 83.3–92.2% | đã đo |

Dự toán 175/200 cho flash-mm và 180/200 cho Q8-lm chỉ là 200×SR screening, giả định cùng phân phối; không được đưa vào cột official đo thật. Defaults screening dự toán 190/200 nhưng official thực tế 177/200 cho thấy phép ngoại suy có thể lệch đáng kể. Q8-vision chưa đủ screening: không thể suy SR từ RMSE hoặc 2/2 smoke; khoảng có thể về mặt logic là 0–100%, không có point estimate được dữ liệu hỗ trợ. Official của các biến thể vẫn chưa đo. Với 3 biến thể tối đa, ngân sách còn 600 episode; chọn 2 biến thể thì 400 episode. Thời gian chạy không thể tính chỉ từ engine latency vì số bước/task và simulator thay đổi.

## Task 5 — phân bổ GPU và mô hình dự báo

Từ GPU-active PHASE đã đo: defaults = 52,708+14,956+36,318 = 103,982 ms; flash-mm = 85,682 ms. Tỷ trọng defaults vision+connector/prefill/denoise tương ứng 50,69% / 14,38% / 34,93%. Tổng active giảm 17,60%; không thay thế speedup end-to-end vì khác graph/padding và loại khoảng trống. Connector riêng và attention riêng chưa đủ instrumentation để cho số đo độc lập.

Mô hình tuyến tính theo depth: vision 12→10 tiết kiệm tối đa xấp xỉ 52,708×2/12 = 8,785 ms active; tổng dự báo 95,197 ms, speedup 1,092×. Coupled VLM/expert 32→30 dự báo tiết kiệm (14,956+36,318)×2/32 = 3,205 ms; tổng 100,777 ms, speedup 1,032×. Giả định mỗi tầng cùng chi phí, fixed overhead không đổi; dùng toàn phase cho layer là xấp xỉ lạc quan. GPU-active đo thực tế vision-spaced giảm vision 17,02%; coupled giảm prefill 5,76% và denoise 4,92%.

Default-path nsys: kernel duration 2260,227→1733,185 ms trên 20 predict, giảm 23,32%. GEMM-name share 58,30→51,85%; gaps 10,54→9,13% (giảm 1,41 điểm phần trăm). GEMM có cả attention matmul; không thể tính riêng thời gian attention bằng phần bù GEMM. Gaps chỉ trong activity đã trace, không chứng minh toàn GPU idle. Raw và cơ chế attribution: ARCHITECTURE_AND_PROFILE.md.

## Task 6 — ba pruning đã implement và kiểm tra

`prune_depth.py` cắt block thật, reindex tensor và cập nhật metadata; giữ nguyên bytes của weight còn lại. Vision-tail bỏ 10,11; vision-spaced bỏ 4,8; coupled-tail bỏ 30,31 ở cả VLM/expert. Coupled giữ parity SA/CA; loader dùng chung số layer nên chưa prune VLM/expert độc lập. Đây là structured depth pruning không train, chưa phải CLP/CKA. Zero weight không giảm kích thước dense GEMM nên không chọn làm bằng chứng tăng tốc.

| Pruning | File MiB | Mean ms | p50/p90 ms | Giảm mean | Allocation MiB | Maxabs/RMSE real7 | Smoke |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smolvla-vision-tail-10 | 1047.10 | 106.4 | 106.6/108.1 | 6.42% | 1885.39 | 1.75651/0.519006 | 0/2 |
| smolvla-vision-spaced-10 | 1047.10 | 104.7 | 104.5/105.7 | 7.92% | 1885.39 | 0.0949649/0.0416718 | 0/2 |
| smolvla-coupled-tail-30 | 1024.75 | 110.4 | 110.3/110.9 | 2.90% | 1814.40 | 0.0633535/0.0224658 | 0/2 |

Benchmark dùng control 113,7 ms cùng candidate batch. Smoke cả ba thất bại; không ngoại suy thành official 0/200. Wilson cho 0/2 có upper bound 65,76% theo mô hình Bernoulli, cho thấy mẫu rất nhỏ; vẫn đủ lý do chưa ưu tiên chạy official tốn kém. Export sidecar `inference_validated=false` là trạng thái lúc export; inference về sau đã được `candidate-checks/summary.json` xác nhận, không chỉnh lịch sử export.

Kết luận task 6: đã implement hai cách chọn layer vision (tail/spaced) và một coupled pruning, chứng minh giảm module active/end-to-end ở các trường hợp đã đo; chưa có phương pháp giữ SR đạt yêu cầu triển khai. Bước CKA cần activation calibration riêng và finetuning/đánh giá mới, không có phép tính thay thế được thí nghiệm đó.

## Bằng chứng và cách đọc

Tất cả đường dẫn dữ liệu tương đối dưới `research-notes/smolvla-followup-20261104/`: `candidate-bench/selected.json`, `memory-all/summary.json`, `candidate-checks/summary.json`, `candidate-smokes/results.json`, `pruning-summary.json`, `phase-suite/selected.json`, `kernel-phase-traces/*.boundaries.json`, `kernel-traces/*.summary.json`, các `screen40-*/result.json`. Bảng runtime và protocol: [PART3_OPTIMIZATION.md](PART3_OPTIMIZATION.md). Checklist và kết luận chính: [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md). Các link ở phụ lục này tính từ gốc repo.

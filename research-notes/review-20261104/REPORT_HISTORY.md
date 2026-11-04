> ARCHIVE: các trạng thái trong file này có thể đã hết hiệu lực. Xem REPORT_SMOLVLA.md ở root cho snapshot hiện tại.

# Báo cáo tiến độ thực nghiệm SmolVLA

<!-- SMOLVLA_LIVE_START -->
## Hiện trạng mới nhất
Cập nhật: 04/11/2026 15:00:32 UTC+07. Khối này tổng hợp dữ liệu mới; các mục phía dưới giữ lịch sử số đo.

- Defaults screening: **38/40**; không phải official200.
- CUDA tests: **13/15**, hai lỗi bit-exact còn mở. Dependency client còn7metadata incompatibilities.
- Smoke biến thể: **10/10 cấu hình đã hoàn tất**, đang chạy `—`.
- Server cũ5555 đã được người dùng dừng sau các lượt benchmark/trace; không gán lại điều kiện đo trước đó.

### Runtime flags — full protocol

| Config | Mean / p50 / p90 (ms) | Peak traced allocation (MiB) | SR biến thể |
|---|---:|---:|---|
| defaults | 113.9 / 113.9 / 115.4 | 1912.47 | 38/40 screening |
| flash-mm | 91.5 / 91.5 / 91.8 | 1040.27 | 1/2 smoke |

3process/config,3warmup+20reps,2views512²/16tokens. Best process theo mean; flash+mm giảmmean19,67% trong batch đối chiếu. Allocation không bao hết context/driver, không phải NVML resident VRAM.

### GGUF — cùng runtime defaults

| Model | File MiB | p50 / p90 ms | Peak allocation MiB | Max abs real7 | Smoke task0 |
|---|---:|---:|---:|---:|---|
| defaults | 1074.18 | 113.7 / 114.6 | 1912.47 | 0 | 2/2 (lượt gốc) |
| smolvla-coupled-tail-30 | 1024.75 | 110.3 / 110.9 | 1814.40 | 0.06335353 | 0/2; complete |
| smolvla-q4_0-lm | 626.76 | 110.2 / 111.0 | 865.07 | 0.08243826 | 1/2; complete |
| smolvla-q4_0-vision | 510.32 | 109.9 / 110.4 | 729.75 | 0.1761732 | 0/2; complete |
| smolvla-q5_0-lm | 665.67 | 110.5 / 111.2 | 903.98 | 0.03389797 | 1/2; complete |
| smolvla-q5_0-vision | 559.35 | 111.0 / 111.9 | 778.78 | 0.1002122 | 0/2; complete |
| smolvla-q8_0-lm | 782.38 | 109.3 / 110.3 | 1020.70 | 0.008421391 | 2/2; complete |
| smolvla-q8_0-vision | 706.45 | 110.3 / 111.0 | 925.88 | 0.04125226 | 2/2; complete |
| smolvla-vision-spaced-10 | 1047.10 | 104.5 / 105.7 | 1885.39 | 0.09496488 | 0/2; complete |
| smolvla-vision-tail-10 | 1047.10 | 106.6 / 108.1 | 1885.39 | 1.75651 | 0/2; complete |

Latency bảng GGUF thuộc candidate-bench, có control riêng (khác batch runtime-pair). Mỗi config3process×(3warmup+20reps). Max abs dùng harness6token/noise cố định, không phải input16token của benchmark. Quant-lm bao gồm connector; action expert289tensor giữ float. Smoke2episode không thay screening40/official200.

### Phase — GPU active kernel time

| Config | Vision+connector ms | KV prefill ms | Denoise+crossKV ms |
|---|---:|---:|---:|
| defaults | 52.708 | 14.956 | 36.318 |
| flash-mm | 40.520 | 11.687 | 33.475 |
| smolvla-vision-spaced-10 | 43.740 | 14.920 | 36.591 |
| smolvla-coupled-tail-30 | 52.722 | 14.095 | 34.531 |

20predict sau3warmup/trace. Gán module bằng biênD2D KVcopy từ source và kiểm tra toàn bộ kernel không thiếu/trùng; có cả direct launches và graph replay. Đây là active GPU time, loại CPU/copy/gaps; phase dùng16token thực thay48slot padded nên không phải phân rã chính xác baseline throughput. Denoise chưa phải riêng action expert, vision chưa tách connector.

### Còn cần thực hiện

- Hoàn tất smoke các biến thể; sau đó screening40 cho cấu hình được chọn.
- Baseline official10×20 đã xếp hàng tự chạy sau smoke (`official200-runner.log`); biến thể official chưa chạy.
- Chưa có CLP/CKA/calibration/finetuning; pruning hiện là depth ablation với lựa chọn tail/spaced.
- Chưa chốt cấu hình tốt nhất: tốc độ/memory và output synthetic không chứng minh giữ SR.

Raw data: `research-notes/smolvla-followup-20261104/` — `screen40-defaults`, `candidate-bench`, `memory-all`, `candidate-checks`, `candidate-smokes`, `kernel-phase-traces`.
<!-- SMOLVLA_LIVE_END -->


**Cập nhật:** 04/11/2026 (UTC+07). **Model:** `vrfai/smolvla-libero-gguf`.

Đã xác minh các giao diện CUDA, đo baseline, screen 10 cấu hình runtime, cài LIBERO và hoàn thành smoke task0 **2/2 episode thành công**. Đã có SR sàng lọc defaults38/40=95,0%; chưa có SR chính thức200 hoặc speedup pruning. Còn hai test CUDA fail và bảy xung đột dependency metadata cần được công bố rõ.

## 1. Trạng thái sáu chặng

| Chặng | Trạng thái thực tế | Phần còn mở |
|---|---|---|
| **1. Cài đặt** | Build hiện có cập nhật; CLI, server, binding, predict_check chạy CUDA; ctest 13/15 | Hai test bit-exact fail; đã có ablation fused/unfused, chưa sửa nguyên nhân |
| **2. Baseline** | p50/p90 đủ protocol; peak allocation CUDA riêng process đã đo | Resident VRAM NVML vẫn N/A; allocation không bao hết context/driver |
| **3. Tối ưu** | Screen 10 cấu hình runtime; đã export 6 GGUF quant và kiểm tra cấu trúc/raw tensors | Chưa full rerun, memory mỗi cấu hình, inference/smoke các GGUF mới hoặc SR theo flags |
| **4. LIBERO** | Smoke2/2; screening10task×4 đủ40episode: **38/40=95,0%**, có CSV/video | 7 metadata incompatibilities; chưa official200 hoặc SR biến thể |
| **5. Profile** | Đã đọc paper/source, có sơ đồ kiến trúc và source map; nsys memory đã đo | Chưa đo phase hoặc phân tích GEMM/attention/GPU idle |
| **6. Pruning** | Có exporter depth-ablation và 3 GGUF prototype đã kiểm tra cấu trúc | Chưa CKA/CLP, inference, latency module hoặc SR; chưa chứng minh speedup |

## 2. Môi trường tái hiện

| Thành phần | Đã xác minh |
|---|---|
| Repo / commit | `/home/doris/projects/vla.cpp` trên Linux WSL2; `4a542ed3bfb8f330388aceb8926294a951156172` |
| GPU | RTX 3060 Laptop, 6144 MiB, compute capability 8.6 |
| Driver / toolkit / profiler | 610.62 / nvcc13.3.73 / nsys2026.1.3 |
| Build benchmark/server | `build-cuda`, Release, CUDA ON, architecture86, VLA_BUILD_TESTS ON, CUDA graphs ON |
| GGUF | `~/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf`; 1.126.360.736 bytes |
| GGUF SHA256 | `6fb2d475c98b4c2cef3e27c4eff4e67b483740cbf983fff320a3b8a5e5f74fe8` |
| Điều kiện laptop baseline | Người dùng xác nhận cắm sạc, chế độ bình thường, không tản nhiệt ngoài |
| Client LIBERO | Python3.10.22; torch2.5.1+cpu; torchvision0.20.1+cpu; transformers4.51.3; numpy1.26.4 |
| Simulator / phụ trợ | mujoco2.3.2, robosuite1.4.0, lerobot0.4.3, gymnasium0.29.1, pyarrow12.0.1 |

PyTorch CPU cho client là lựa chọn đã được người dùng xác nhận. Server C++ inference vẫn CUDA; `torch.cuda.is_available=False` ở client là dự kiến. Config LIBERO riêng trong `research-notes/smolvla-followup-20261104/libero-config`; không xóa/ghi đè `~/.libero`. Phiên bản đầy đủ: [libero-freeze.txt](research-notes/smolvla-followup-20261104/libero-freeze.txt).

Không sửa source mô hình, test tolerance, commit hoặc push. Các số baseline và screen được tổng hợp từ log đã lưu, không coi là một lượt đo duy nhất cùng thời điểm. Ứng dụng GPU khác, nhiệt độ phòng và vị trí kê máy chưa xác nhận đầy đủ.

## 3. Chạy thử và lỗi CUDA còn mở

- Incremental build exit0 (`ninja: no work to do`), chưa phải clean rebuild.
- ctest: **13/15 đạt**, 21,76 giây. Hai fail: `foldquant_cuda_op`, `fused_cuda_ops`.
- CLI chạy one-shot CUDA và trả action. Input smoke là ảnh có sẵn/token synthetic, chưa xác minh tokenizer tự nhiên ở lượt CLI.
- Binding, server loopback và predict_check: output hữu hạn `(50,32)`, khớp bit với cùng ảnh/token/state/noise cố định, max_abs=0. Chỉ 7 cột đầu là action thực.
- FoldQuant fail đầu tiên: `w8a8_rot64_bias K=128 N=64 T=5`, CUDA `-3.4419589` vs CPU `-3.44195867`. Test dùng so từng bit. Log no-fma cũ cũng fail; target FoldQuant mặc định đã có `-fmad=false`.
- Chuyển kernel FoldQuant sang WMMA vẫn fail cùng giá trị. Chưa xác định nguyên nhân hoặc tác động SR.

Ablation predict_check trên cùng input: default lặp lại khớp bit; `VLA_FUSED_OPS=0` so default khác 350/1600 giá trị, max_abs **0,000545874**, RMSE toàn32 **0,000100497**. Tắt exact GEMM riêng không đổi output trên input này; tắt exact attention có đổi output nhưng vẫn chưa khớp unfused. Không kết luận kernel cụ thể từ các ablation này.

Bằng chứng: [ctest](research-notes/smolvla-audit-20261104/ctest-build-cuda.log), [interfaces](research-notes/smolvla-audit-20261104/interfaces.log), [ablation](research-notes/smolvla-followup-20261104/ablation-summary.json), [isolate](research-notes/smolvla-followup-20261104/isolate-summary.json).

## 4. Baseline latency và memory

Engine `predict()` trên synthetic input: defaults, 2 views×512×512, 16 tokens, chunk50, real action7/padded32, 10 denoise steps. Mỗi process 3 warmup+20 reps, ba process tuần tự. Chọn process có mean thấp nhất theo docs/benchmark/rtx-3060.md.

| Process | min ms | mean ms | p50 ms | p90 ms | vision ms |
|---|---:|---:|---:|---:|---:|
| **1 — được chọn** | 111,0 | 112,0 | **111,8** | **112,8** | 55,4 |
| 2 | 112,5 | 113,6 | 113,6 | 114,6 | 55,6 |
| 3 | 113,1 | 114,7 | 114,8 | 115,4 | 56,1 |

Không gồm simulator/transport/video. Số làm tròn từ benchmark, không có latency raw từng rep. Telemetry sampled gồm cả load/idle: 47–62°C, SM210–2017MHz, power10,02–112,83W; chưa đủ kết luận throttle. Chu kỳ thực tế lớn hơn0,5s vì thời gian query NVML.

| Chỉ số bộ nhớ | Kết quả | Giới hạn |
|---|---:|---|
| **Peak traced CUDA allocation riêng benchmark** | **1912,47 MiB** | 2.005.369.572bytes, giống cả3 process nsys; Device+Device Static, có VMM |
| Resident VRAM process qua NVML | **N/A** | Chưa có số theo phương pháp gốc của repo |
| Peak sampled VRAM toàn GPU, lượt baseline trước | 3944MiB | Có cả phần khác, chỉ tham khảo |

Peak allocation được tính bằng ledger allocation/deallocation theo timestamp, từng process/device; không đếm host/pinned RAM hoặc virtual address reservation. Không có free thiếu allocation, live âm hoặc diagnostic lost/drop. Không bao hết context/driver không được xuất thành event. Peak tính cả load/warmup/inference/cleanup. Latency dưới tracing không thay baseline; hai số1912,47 và3944 thuộc các lượt khác nhau, không trừ để tính overhead.

Bằng chứng: [baseline1](research-notes/smolvla-audit-20261104/baseline-1.log), [baseline2](research-notes/smolvla-audit-20261104/baseline-2.log), [baseline3](research-notes/smolvla-audit-20261104/baseline-3.log), [memory summary](research-notes/smolvla-vram-20261104/memory-summary.json).

## 5. Runtime screen — kết quả sơ bộ

2 warmup+5 reps mỗi cấu hình, một process/config, cùng benchmark input512²/2views/16tokens. Độ lệch action dùng **harness riêng** với 6 tokens và ảnh/state/noise cố định, so defaults cùng lượt. RMSE dưới đây chỉ tính50×7 chiều thực, loại padding zero. Các chiều có thể khác đơn vị; không dùng độ lệch này suy ra SR.

| Runtime flags | Mean screen (ms) | Max abs, real7 | RMSE, real7 |
|---|---:|---:|---:|
| `defaults` | 111.9 | 0 | 0 |
| `--flash-attn` | 99.9 | 0.00078821182 | 0.00028052564 |
| `--mm-prec default` | 106.1 | 0.0020601451 | 0.00066080622 |
| `--flash-attn --mm-prec default` | 93.0 | 0.0027671456 | 0.00061441981 |
| `--weight-dtype bf16` | 134.3 | 0 | 0 |
| `--weight-dtype bf16 --flash-attn` | 121.9 | 0.00078821182 | 0.00028052564 |
| `--weight-dtype f16` | 135.8 | 0.00097492337 | 0.00026844391 |
| `--weight-dtype f16 --flash-attn` | 118.7 | 0.0012158751 | 0.0002933432 |
| `--num-steps 5` | 95.6 | 0.059932411 | 0.018432219 |
| `--num-steps 3` | 89.2 | 0.30797094 | 0.042685641 |

`flash+mm-default` là ứng viên screen nhanh hơn defaults, chưa phải cấu hình được xác nhận tốt nhất. `steps3` có mean thấp nhất trong screen nhưng độ lệch action lớn hơn đáng kể. Chưa full rerun theo protocol, memory từng cấu hình, quantization hoặc SR cho các flags. Không gộp số screen5 reps vào baseline20 reps.

Bằng chứng: [screen summary](research-notes/smolvla-followup-20261104/runtime-screen/summary.json), [script screen](research-notes/smolvla-followup-20261104/screen_runtime.py), [script đối chiếu](research-notes/smolvla-followup-20261104/summarize_runtime.py).

## 6. LIBERO — cài đặt và smoke

Kiểm tra cài đặt exit0, marker `LIBERO_INSTALL_AND_RENDER_OK`: import client/adapters OK; suite libero_object có10 task; task0 reset, render agentview/eye_in_hand đều128×128×3, thực hiện1 dummy action thành công. Đây là kiểm tra riêng trước smoke.

Smoke GGUF gốc/defaults: libero_object/task0 “pick up the alphabet soup and place it in the basket”, seed42, n_action_steps1; serverCUDA loopback15556, clientCPU+EGL. Client exit0, hai video được lưu và mở/giải mã frame đầu thành công (135/160 frame, 256×512, 30fps), không skip/abort.

| Episode | Success | Số bước | Average inference/step do client báo |
|---|---|---:|---:|
| 0 | True | 134 | 151,03ms |
| 1 | True | 159 | 137,71ms |

**Smoke: 2/2 thành công.** Summary client ghi average144,37ms/step. Chưa phải screening40 episode hoặc official200 episode; không dùng client timing thay engine baseline. Không suy ra SR tổng quát từ hai episode của một task.

Giới hạn và cảnh báo đã giữ nguyên:

- `uv pip check` có7 xung đột: pettingzoo/Gymnasium; datasets/PyArrow; OpenCV headless/NumPy; LeRobot/TorchVision; LeRobot/Gymnasium; rerun/NumPy; rerun/PyArrow. Các pin repo chạy được import/reset/render/smoke, nhưng chưa chứng minh mọi tính năng phụ của các package này hoạt động. Không báo môi trường dependency sạch.
- Warmup synthetic startup server dùng224², không hợp SmolVLA512² nên fail và skip; request thật của client512² đã chạy CUDA. Chưa sửa warmup.
- Observation space cảnh báo: đã xác nhận Text mặc định không chấp nhận dấu cách trong task_description. Chưa sửa declaration hoặc observation; chưa loại trừ mọi nguyên nhân khác ở từng trường.
- Cảnh báo thiếu dataset path không ngăn smoke (task/init/assets có sẵn); chưa tải training dataset. Tokenizer fork warning được giữ trong log.

Bằng chứng: [verify](research-notes/smolvla-followup-20261104/libero-verify.log), [pip check](research-notes/smolvla-followup-20261104/libero-pip-check.log), [smoke client](research-notes/smolvla-followup-20261104/smoke-original/client.log), [smoke server](research-notes/smolvla-followup-20261104/smoke-original/server.log), [CSV](research-notes/smolvla-followup-20261104/smoke-original/outputs/smolvla/libero_object/task_0/episodes.csv), [summary](research-notes/smolvla-followup-20261104/smoke-original/outputs/smolvla/libero_object/task_0/summary.txt).

## 7. Quyết định và bước tiếp theo

**Baseline đã chốt:** người dùng chọn defaults của repo, công bố rõ hai test fail và độ lệch fused/unfused. Screening bản gốc 10 task × 4 episode được triển khai riêng với seed42 và n_action_steps1; kết quả/progress ở screen40-defaults. Chưa có official SR.

Sau khi chốt: full rerun ứng viên runtime, đo allocation từng cấu hình cùng định nghĩa; quantization nhiều mức có/không vision, smoke mỗi GGUF mới; screening10×4, official10×20 cho bản gốc và tối đa3 cấu hình tốt nhất. Kiến trúc/phase/nsys kernel và pruning vẫn chưa có kết quả.

Lệnh tái hiện đầy đủ, diễn biến cài và nguyên văn mentor: [NOTES_SMOLVLA.md](NOTES_SMOLVLA.md). Không commit/push.


Bổ sung kiến trúc: [ARCHITECTURE_AND_PROFILE.md](research-notes/smolvla-followup-20261104/ARCHITECTURE_AND_PROFILE.md), dựa trên paper và metadata checkpoint thật. Source cfg32 VLM/expert, vision12; chưa dùng số minh họa paper để thay cấu hình đang chạy. Phase có thay đổi topology/padding nên chưa coi là phân rã chính xác của baseline. Screening hiện giữ noise mặc định ngẫu nhiên; seed42 môi trường không ghim noise action. Các lượt smoke/screen không bảo đảm tái hiện trajectory từng bit giữa hai server.


Các file model chuẩn bị, chưa inference/SR: [quantization manifest](research-notes/smolvla-followup-20261104/quantization/manifest.json), [pruning summary](research-notes/smolvla-followup-20261104/pruning-summary.json). Mô hình gốc không bị sửa.


## Model quantization đã export (chưa inference)

| File | Bytes | MiB | Scope |
|---|---:|---:|---|
| smolvla-q8_0-lm | 820389536 | 782.38 | VLM + connector; giữ ViT float |
| smolvla-q8_0-vision | 740763296 | 706.45 | VLM + connector + ViT |
| smolvla-q5_0-lm | 698001056 | 665.67 | VLM + connector; giữ ViT float |
| smolvla-q5_0-vision | 586524320 | 559.35 | VLM + connector + ViT |
| smolvla-q4_0-lm | 657204896 | 626.76 | VLM + connector; giữ ViT float |
| smolvla-q4_0-vision | 535111328 | 510.32 | VLM + connector + ViT |

Action expert giữ float và được kiểm tra raw payload không đổi (289tensor). Giữ791tensor và metadata. Không điền latency/VRAM/SR cho các model chưa chạy.


## Kết quả screening defaults đã chốt

**38/40 (95,0%)**, 10task ×4episode,0abort/missing. Task3 và5 đạt3/4, tám task khác4/4; hai failure đều500step. Đây là screening, không thay official10×20. Seed42 chỉ cố định môi trường; noise server vẫn random theo defaults. Raw [result.json](research-notes/smolvla-followup-20261104/screen40-defaults/result.json), [protocol](research-notes/smolvla-followup-20261104/screen40-defaults/protocol.json), [artifact audit](research-notes/smolvla-followup-20261104/screen40-artifact-audit.json). Cả40video đã mở frame đầu và kiểm tra sốframe khớp steps+1, chưa decode toàn bộ mọi frame.

## Phần 3 — đã bổ sung đủ bảng tối ưu không train

Xem [PART3_OPTIMIZATION.md](PART3_OPTIMIZATION.md): 10 runtime configs và 6 quantized GGUF, mỗi hàng có kích thước, peak traced allocation, latency p50/p90/mean và max abs/RMSE action real7. Runtime bổ sung đủ 3 process × (3 warmup +20 reps), memory trace riêng đủ 30 PID; giữ dữ liệu batch quantization riêng. Runtime mới defaults mean104.0ms, flash+mm85.5ms; không thay thế các số lịch sử bằng số mới. WSL vẫn chưa đo được toàn bộ peak resident VRAM process. Không chạy SR/pruning trong đợt này.

## Sàng lọc biến thể — đang chạy

Đã bắt đầu flash+mm → Q8 không vision → Q8 có vision, mỗi cấu hình10task×4episode (tổng120). Giữ seed42, actionsteps1, repo default random noise. Nhận ZMQ không timeout; mỗi task có giới hạn1800giây ở runner. Nếu thiếu/abort hoặc client lỗi sẽ dừng batch và giữ artifact, không giảm mẫu số. Raw: `research-notes/smolvla-followup-20261104/screen40-variants/`; log tổng `screen-variants-runner.log`. Đây là screening, chưa phải official200 của biến thể.

# SmolVLA — nhật ký thực nghiệm

Cập nhật trạng thái ngày 05/11/2026 (UTC+07). Các mục phía sau là nhật ký theo thời điểm; những dòng “chưa đo/chưa cài” cũ không đại diện cho hiện trạng. Phạm vi và kết luận hiện tại ở [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md).

## Checklist 6 yêu cầu (trạng thái hiện tại)

| Mục | Đã hoàn thành | Giới hạn còn mở |
|---|---|---|
| 1. Cài đặt và chạy thử | CUDA arch86; CLI/server/binding/predict_check hoạt động | Tests 13/15; hai lỗi bit-exact tạm hoãn |
| 2. Baseline | Mean/p50/p90; nsys peak allocation 1912,47 MiB | Process resident VRAM N/A; allocation không bao hết context/driver |
| 3. Runtime flags / quantization | 10 cấu hình runtime; Q8/Q5/Q4 có/không vision; file, latency, allocation, output delta | Chưa xác nhận giữ SR; chỉ ba ứng viên có screening |
| 4. LIBERO SR | Defaults official 177/200; screening defaults 38/40, flash-mm 35/40, hai Q8 36/40 | Official biến thể và repeated-seed chưa chạy, ngoài phạm vi chốt đợt này |
| 5. Architecture / profile | Metadata checkpoint, sơ đồ, phase và nsys node traces | Connector/attention chưa tách riêng; GPU-active khác wall time |
| 6. Pruning | Ba GGUF depth ablation, kiểm tra output, latency module và smoke | Cả ba smoke 0/2; chưa implement CLP/CKA hoặc finetuning |

Kiểm tra lại CSV cho đủ 360 episode và bảng theo task: [TASK_RESULTS.md](research-notes/smolvla-followup-20261104/TASK_RESULTS.md). Phần còn thiếu và tiêu chí hoàn tất: [MENTOR_REVIEW.md](MENTOR_REVIEW.md).

## Nguyên văn mentor

         Model: SmolVLA, bản vrfai/smolvla-libero-gguf 
1. Cài đặt và chạy thử. Build vla.cpp có CUDA qua WSL2 theo docs/backend/wsl.md, tự chọn CUDA architecture cho RTX 3060 Laptop và bật build test. Sau đó chạy thử test, CLI, server, Python binding và vla_predict_check để hiểu mỗi công cụ dùng làm gì.
2. Baseline. Đo latency (p50/p90) và VRAM đỉnh bằng vla-bench theo protocol và cấu hình input cho SmolVLA trong docs/benchmark/ (tham khảo rtx-3060.md). Ghi rõ điều kiện đo vì laptop dễ bị throttle.
3. Tối ưu không cần train. Thử các runtime flag trong docs/USAGE.md và docs/backend/cuda.md (attention, dtype, số bước denoise, …). Quantize theo docs/MODELS.md ở nhiều mức, có và không quantize vision tower. Mỗi cấu hình ghi kích thước file, VRAM, latency và độ lệch output so với bản gốc.
4. Đo SR trên LIBERO. Cài theo mục "End-to-end evaluation (LIBERO client on WSL)" trong docs/backend/wsl.md. 
Mức
Episode
Dùng cho
Smoke test
1 task × 2
Mỗi lần đổi GGUF
Sàng lọc
10 task × 4 = 40
Cấu hình flag, quantize, pruning
Chính thức
10 task × 20 = 200
Bản gốc và tối đa 3 cấu hình tốt nhất
5. Profile. Mô tả hoặc vẽ lại kiến trúc SmolVLA (vision, connector, VLM, action expert, vòng denoise) từ paper và src/models/smolvla.cpp. Đo latency từng thành phần bằng cơ chế timing theo phase của repo và ghi rõ nó đo gì, giới hạn gì. Với bản gốc và bản tốt nhất, dùng nsys để xem thời gian nằm ở GEMM, attention hay phần khác, GPU có bị trống giữa các kernel không, và ghi lại hạn chế của nsys trên WSL2.
6. Bạn tự tìm hiểu các phương pháp pruning phù hợp với SmolVLA, chọn và implement lại một vài phương pháp để đánh giá. Nếu cần định dạng checkpoint safetenssors có thể tham khảo HuggingFaceVLA/smolvla_libero.  Dùng vla-bench và nsys để kiểm tra latency của từng module (vision, VLM, action expert) có giảm thật không.
1 số Paper tham khảo: [SmolVLA (arXiv 2506.01844)](https://arxiv.org/abs/2506.01844 "https://arxiv.org/abs/2506.01844"), [CLP: Finetuning VLA Models Requires Fewer Layers Than You Think (arXiv 2606.20246)](https://arxiv.org/html/2606.20246v1 "https://arxiv.org/html/2606.20246v1").


## Kiểm kê và bằng chứng

- Repo: `/home/doris/projects/vla.cpp`, Linux WSL2, không nằm trên /mnt/c. Git ban đầu sạch.
- HEAD: `4a542ed3bfb8f330388aceb8926294a951156172`.
- origin: https://github.com/hientran-ai/vla.cpp.git; upstream: https://github.com/VinRobotics/vla.cpp.git.
- Không có AGENTS.md áp dụng tại gốc; các AGENTS.md tìm thấy thuộc dependency llama-src, không chỉnh sửa dependency.
- `build-cuda` và `build-cuda-no-fma`: Release, GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=86, VLA_BUILD_TESTS=ON, GGML_CUDA_GRAPHS=ON. Architecture 86 khớp compute capability 8.6 trong banner GPU thực tế. `BUILD_TESTING` không phải cờ test chính của VLA.
- `build-cuda-no-fma` thêm CMAKE_CUDA_FLAGS=-fmad=false; không coi tên thư mục là bằng chứng đã sửa lỗi.
- GPU ngoài sandbox: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 610.62. nvcc 13.3.73. nsys 2026.1.3.425.
- Trong sandbox: NVML báo GPU access blocked, /dev/dxg và /dev/dri không thấy. Truy vấn ngoài sandbox thành công: đây là giới hạn môi trường chạy lệnh, không kết luận driver WSL hỏng. Lệnh CUDA sau đó chạy ngoài sandbox.
- Model cache: `/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf`, 1,126,360,736 bytes. SHA256: `6fb2d475c98b4c2cef3e27c4eff4e67b483740cbf983fff320a3b8a5e5f74fe8`. Không dùng số 532 MB từ ví dụ docs làm kích thước file hiện tại.
- Log/model tìm ở repo (bao gồm build bị gitignore), research-notes và /home/doris sâu tối đa 5 cấp; chưa tìm thấy log server cũ riêng. Không khẳng định không tồn tại nơi khác.

## Lệnh đã chạy và ý nghĩa

Chạy từ gốc repo. Log nằm ở `research-notes/smolvla-audit-20261104/`.

```bash
pwd
git status --short
git rev-parse HEAD
git remote -v
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version
nsys --version
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
VLA_IMG_SIZE=512 ./build-cuda/tests/vla_predict_check /home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf '' 2
./build-cuda/vla-cli --ckpt /home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf --image assets/front.jpg --image assets/front.jpg --tokens 1,100,200,300,400,2 --pretty
.venv/bin/python research-notes/smolvla-audit-20261104/check_interfaces.py
.venv/bin/python research-notes/smolvla-audit-20261104/run_baseline.py
```

- Build: `build.log`: `ninja: no work to do.` exit 0. Đây là xác minh build hiện có cập nhật, chưa phải clean rebuild.
- Test: `ctest-build-cuda.log`: 13/15 đạt, 21.76 s. `foldquant_cuda_op` và `fused_cuda_ops` fail. Các test CUDA thực sự thấy GPU; không dựa vào skip do thiếu GPU.
- Lỗi FoldQuant mới: w8a8_rot64_bias K=128 N=64 T=5, y[5] = -3.4419589 vs CPU -3.44195867. Test dùng memcmp bit-exact. Sai khác nhỏ vẫn là fail đúng hợp đồng test. fused_cuda_ops cũng so từng bit. Chưa kết luận nguyên nhân hoặc tác động lên SR.
- Log cũ được giữ nguyên trong `previous-build-cuda-LastTest.log`, `previous-build-cuda-no-fma-LastTest.log` trước khi ctest ghi đè. Build no-fma cũ có cùng lỗi FoldQuant và nhiều lỗi fused. Không chạy lại toàn bộ build no-fma trong chặng này.
- CLI (`cli.log`, `cli-actions.out`): one-shot, hai lần dùng front.jpg và token synthetic; trả 50 hàng, 32 cột, backend CUDA. Không xác minh tokenizer với instruction tự nhiên; không so CLI với harness vì input/noise khác.
- predict_check (`predict-default.log`, `predict-default.out`): ảnh pattern cố định 2×512², 6 tokens, state/noise cố định; real_state=8, max_state=32, real_action=7, max_action=32, chunk=50, steps=10; action_len=1600. Output có banner trước action_len, parser cần bỏ banner.
- Binding (`interfaces.log`, `binding-actions.npy`): C ABI dùng libvla.so hiện có; predict trả (50,32), hữu hạn; so với predict_check max_abs=0, bit_equal=True.
- Server (`server.log`, `server-actions.npy`): bind loopback tcp://127.0.0.1:15555, warmup=0; nhận request protobuf RGB_U8, cùng token/state/noise; trả (50,32), hữu hạn, max_abs=0 và bit_equal=True so binding. Script tự terminate server trong finally.
- Token synthetic chỉ kiểm tra kỹ thuật inference; không chứng minh hiểu instruction hoặc task success. So bit các giao diện cũng không xác nhận mọi kernel đúng.

## Baseline latency đã đo

Theo docs/benchmark/rtx-3060.md: engine predict(), không transport/simulator. Defaults, 2 ảnh 512×512, 16 language tokens, chunk=50, padded action dim=32 (real=7), denoise=10, 3 warmup + 20 reps, 3 process chạy tuần tự, gap mặc định 0. Không chạy CLI/server/test đồng thời trong thời gian baseline. Chính script vla-bench sinh synthetic input theo source; không dùng input 6 tokens của predict_check để tính baseline.

| Process | min ms | mean ms | p50 ms | p90 ms | vision ms |
|---|---:|---:|---:|---:|---:|
| 1 (mean thấp nhất, được chọn) | 111.0 | 112.0 | 111.8 | 112.8 | 55.4 |
| 2 | 112.5 | 113.6 | 113.6 | 114.6 | 55.6 |
| 3 | 113.1 | 114.7 | 114.8 | 115.4 | 56.1 |

Đây là số in làm tròn từ vla-bench; không có raw latency từng rep. Raw log: baseline-{1,2,3}.log; command đầy đủ baseline-command.json; runner exit 0 cả ba process. Không lấy bảng desktop RTX 3060 12GB trong docs làm số đo laptop này.

VRAM process theo NVML: **chưa đo được** (bổ sung nsys allocation bên dưới). `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory` trả N/A (kể cả process benchmark). VRAM toàn GPU sampled peak = 3944 MiB cả ba process, chỉ tham khảo, có cả phần khác/Windows; không phải peak riêng vla-bench. Sampling ngủ 0.5 s sau hai truy vấn NVML nên chu kỳ thực tế lớn hơn 0.5 s; có thể bỏ lỡ peak ngắn. Không dùng weight_buf hoặc after weights alloc làm peak inference. Cần cơ chế đo tương đương hoặc thống nhất cách báo hạn chế trước khi coi mục VRAM hoàn tất.

Điều kiện người dùng xác nhận: cắm sạc, chế độ bình thường, không tản nhiệt ngoài. Chưa xác nhận ứng dụng GPU khác, nhiệt độ phòng và vị trí kê máy. Telemetry JSONL có timestamp epoch, GPU và danh sách process, không lọc mất giá trị N/A. Toàn khoảng sampled: 47–62°C, SM 210–2017 MHz, power 10.02–112.83 W (bao gồm load/idle); power limit N/A. Không suy ra không throttle chỉ từ nhiệt độ. Mean tăng nhẹ qua ba lượt, chưa đủ để kết luận nguyên nhân.

## Khoanh vùng, giả thuyết và phần chưa đo

Sự thật: lỗi bit-exact tái hiện ở build-cuda; log cũ no-fma cũng fail. Lỗi FoldQuant nằm trong custom quantization kernel test; chưa có bằng chứng nó gây sai model GGUF thường đang chạy. Fused tests liên quan nhiều phép toán, chưa được phép bỏ qua.

Giả thuyết cần kiểm tra: khác thứ tự tích lũy/rounding hoặc toolchain có thể liên quan sai khác số; **chưa chứng minh**. Không sửa flags/test tolerance hoặc default runtime để che fail. Bước khoanh vùng tiếp: so predict_check cùng input/noise với VLA_FUSED_OPS=0, lưu max_abs/RMSE và kiểm tra kernel cụ thể; ghi đây là ablation, giữ baseline defaults.

LIBERO: eval/sim/libero/libero_uv/.venv/bin/python không tồn tại; uv không trên PATH; eval/sim/libero/LIBERO có .git nhưng HEAD chưa resolve, chưa phải checkout dùng được. Setup script repo tạo Python 3.10 venv và có rm -rf venv cùng ~/.libero; chưa chạy script trong chặng kiểm kê. Môi trường chính .venv là Python 3.14, không dùng thay venv LIBERO mà chưa kiểm tra tương thích. Smoke đúng yêu cầu: 1 task × 2 episodes cho mỗi GGUF; screening 10×4; official 10×20, bản gốc và tối đa 3 cấu hình. Chưa có SR.

Timing theo phase: source src/models/smolvla.cpp đo vision gồm preprocessing/vision graph/connector và tensor readback; phase tách prefill graph với denoise graph, copy KV ở giữa. Phase dùng in.n_lang thay cfg.n_lang và dựng graph riêng; tổng các phase không tự bao hết mọi overhead. Không coi số phase là pure kernel time hoặc mặc nhiên bằng default latency. Chưa chạy phase hoặc phân tích kernel bằng nsys nên chưa có số đo thành phần; trace nsys memory được bổ sung ở phần dưới.

Runtime/quantization kế tiếp: screen defaults, flash-attn, mm-prec default, kết hợp, weight bf16/f16 ± flash; 2 warmup+5 reps, cấu hình thắng >3% rồi đo full. Denoise ít bước cần SR riêng. Quant Q8_0/Q5_0/Q4_0, mỗi mức có/không --vision, giữ action expert float theo MODELS.md; chưa tạo các file hoặc giả định cải thiện. Pruning cần đọc paper và chốt phương pháp/calibration/protocol trước khi implement.

## Mình cần hiểu

1. Build cache CUDA không chứng minh GPU truy cập được; cần banner GPU thật. Vì sao binary CUDA có thể vẫn không chạy GPU trong một terminal khác?
2. ctest, CLI, server, binding và predict_check kiểm tra các hợp đồng khác nhau. Vì sao khớp bit giữa các giao diện chưa chứng minh success rate?
3. p50/p90 baseline là engine latency trên synthetic input. Vì sao không dùng VRAM toàn GPU làm peak process hoặc dùng số baseline để suy ra SR?


## Bổ sung: peak bộ nhớ CUDA riêng process bằng nsys (2026-11-04)

Yêu cầu tiếp theo: xử lý phần thiếu peak VRAM do WSL trả N/A. Không đổi GGUF, binary hoặc runtime flags, không sửa source. Kết quả **đo được chỉ số thay thế** sau; không coi nó tương đương số resident VRAM theo process mà docs/benchmark/rtx-3060.md lấy từ NVML.

| Lượt nsys riêng | Peak traced CUDA allocation (MiB) | Bytes | Check ledger |
|---|---:|---:|---|
| 1 | 1912.469456 | 2005369572 | Không free thiếu allocation, không live âm |
| 2 | 1912.469456 | 2005369572 | Không free thiếu allocation, không live âm |
| 3 | 1912.469456 | 2005369572 | Không free thiếu allocation, không live âm |
| **Peak qua 3 process** | **1912.47 MiB** | **2005369572** | Không thấy diagnostic mất/drop event |

### Định nghĩa phép đo và giới hạn

- Nsight Systems 2026.1.3 bật CUDA memory tracing, scope process-tree mặc định; dữ liệu thực tế chỉ có một globalPid và device 0 mỗi lượt. Không lấy bộ nhớ của process khác hoặc trừ hai số toàn GPU để ước lượng.
- Export SQLite `CUDA_GPU_MEMORY_USAGE_EVENTS`; dùng enum từ chính report. Lọc `Device` và `Device Static`, tăng số live bytes ở Allocation, giảm ở Deallocation, lấy high-water mark theo thời gian cho từng process/device. Không cộng host/pinned RAM, memcpy bytes hoặc virtual address reservation.
- Đã đối chiếu VMM: report có 2 sự kiện allocation 2 MiB tương quan `cuMemSetAccess`, và 2 deallocation tương quan `cuMemUnmap`. Không cộng thêm cuMemCreate/cuMemMap để tránh đếm trùng. Số event và VMM chi tiết nằm trong memory-summary.json; ledger từng event nằm trong *.allocations.csv.
- Đây là **peak các CUDA allocation được trace**, không phải tổng resident VRAM của process. Chi phí CUDA context, driver/kernel code và cấp phát nội bộ không được xuất thành event có thể chưa bao gồm. Không gọi đây là số NVML hoặc dùng nó so trực tiếp với peak VRAM desktop trong docs mà không chú thích.
- Kết thúc trace còn 70,681,904 bytes allocation chưa có free (gồm Device + Device Static). Không suy ra memory leak: một số lifetime kết thúc khi process/context bị hủy. Script không ép tổng về zero hoặc bỏ các allocation này.
- Help cài đặt tại máy xác nhận --cuda-memory-usage có thể gây significant runtime overhead. Vì vậy giữ nguyên baseline latency **p50 111.8 ms, p90 112.8 ms** đã đo không profiler; không thay bằng latency dưới nsys.
- Ba process profiled vẫn dùng 2 ảnh 512×512, 16 tokens, 10 steps, 3 warmup + 20 reps; peak tính cả load/warmup/inference/cleanup, không chỉ reps. Latency dưới tracing chỉ để đối chiếu: p50/p90 lần lượt 114.1/114.7, 113.5/114.8, 113.3/114.4 ms.
- Diagnostic nsys có cảnh báo không trace được Unified Memory do thiết bị không hỗ trợ peer-to-peer; không có Managed/Managed Static allocation trong tập memory event đã xuất. Không bật Unified Memory để né giới hạn này.
- Điều kiện laptop ở lần baseline trước người dùng xác nhận cắm sạc/chế độ bình thường/không tản nhiệt ngoài; chưa xác nhận lại ở thời điểm tracing. Đây là lượt đo memory riêng, không dùng để kết luận latency/throttle.
- Truy vấn NVML cuối lượt vẫn cho process memory N/A (raw nvml-processes.txt). Phần **resident VRAM riêng process đúng phương pháp NVML** vẫn là giới hạn WSL, đã có peak allocation riêng process để báo cáo minh bạch.

### Lệnh tái hiện

Chạy từ gốc repo, với GPU được truy cập ngoài sandbox. Dùng tên output mới nếu report đã tồn tại; nsys không ghi đè mặc định.

```bash
nsys profile --trace=cuda --cuda-memory-usage=true --cuda-trace-all-apis=true \
  --sample=none --cpuctxsw=none \
  --output=research-notes/smolvla-vram-20261104/baseline-memory-1 \
  ./build-cuda/vla-bench \
  --ckpt /home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf \
  --images 2 --size 512 --tokens 16 --warmup 3 --reps 20
nsys export --type sqlite \
  --output research-notes/smolvla-vram-20261104/baseline-memory-1.sqlite \
  research-notes/smolvla-vram-20261104/baseline-memory-1.nsys-rep
# Lặp với output baseline-memory-2 và baseline-memory-3, tuần tự.
python3 research-notes/smolvla-vram-20261104/summarize_memory.py
```

Lưu kết quả ở `research-notes/smolvla-vram-20261104/`: profile-{1,2,3}.log, *.nsys-rep, *.sqlite, *.allocations.csv, memory-summary.json, profile-commands.json, GPU/NVML evidence. Script kiểm tra unique allocation address còn live, free khớp size/address, live không âm, một process/device, và diagnostic lost/drop trước khi báo kết quả. Không tự xóa report thô.

**Mình cần hiểu:** p50/p90 mô tả thời gian; peak allocation mô tả số byte CUDA cấp phát đồng thời cao nhất. CUDA context/driver có thể dùng thêm VRAM ngoài các allocation này. Câu hỏi tự giải thích: vì sao 1912.47 MiB allocation riêng benchmark không mâu thuẫn với 3944 MiB toàn GPU đã thấy trước đó?


## Báo cáo hiện trạng

Đã tổng hợp báo cáo ngày 2026-11-04 tại [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md), đối chiếu log baseline, ctest, interfaces và memory-summary.json. Lần lập báo cáo không chạy lại benchmark. Chặng 1 còn hai test fail; chặng 2 có latency và peak allocation CUDA với giới hạn NVML; chặng 3–6 chưa có kết quả đánh giá tương ứng.


## Tiếp tục thực nghiệm và cài LIBERO (04/11/2026)

### Khoanh vùng sai khác CUDA đã đo

Raw artifacts: `research-notes/smolvla-followup-20261104/`.

- predict_check 2 ảnh 512², 6 token, state/noise cố định: hai lần default khớp bit hoàn toàn.
- `VLA_FUSED_OPS=0` so default: 350/1600 phần tử khác bit; max_abs=0.0005458742380142212, RMSE=0.00010049711168250623. Các giá trị hữu hạn. Không suy ra SR hoặc coi là bit-exact.
- `VLA_GEMM_EXACT=0` so default: không khác bit trên input này.
- `VLA_ATTN_EXACT=0`, có/không thêm `VLA_GEMM_EXACT=0`: khác default 350 phần tử, max_abs=0.0011349022388458252; vẫn khác unfused. Đây là ablation, chưa chứng minh kernel cụ thể gây lỗi.
- `VLA_FQ_GEMM=wmma` chạy test FoldQuant vẫn fail đúng giá trị đầu tiên như MMA mặc định. Cả target FoldQuant trong build mặc định đã có `-fmad=false` theo build.ninja; chưa kết luận thêm cờ này sẽ sửa lỗi.
- Chưa sửa source/test tolerance hoặc thay baseline. So action là phép kiểm kỹ thuật, không phải success rate.

### Cài LIBERO — trạng thái đang thực hiện

Kiểm tra mới: LIBERO checkout đã resolve HEAD `8f1084e3132a39270c3a13ebe37270a43ece2a01`, venv Python 3.10.22 đã tồn tại. Đây là trạng thái mới so với lần kiểm kê đầu.

Đã cài uv 0.13.7 vào `research-notes/tools/uv`. Lệnh thử tạo venv mới báo existing venv; giữ lại và dùng venv hiện có. Không rm venv hoặc ~/.libero.

Đã dừng hai installer trùng (PID 41453 mới, 33186 cũ) trong lượt trước; quyết định dừng tiến trình cũ là quá sớm khi chưa kiểm tra kỹ cache/trạng thái và người dùng yêu cầu tiếp tục cài. Đã nhận lỗi và khởi chạy lại requirements với log verbose, sau đó tiếp nối các bước setup đúng repo. Cache cũ ~/.cache/uv còn khoảng 5.5 GiB; cache lượt resume là ~/.cache/uv-vla. Không xóa cache đã tải. Không khẳng định cài xong khi chỉ mới download/build.

Log cài: `libero-resume.log`; script tiếp nối: `continue_install.py`, `finish_libero.sh`; script xác minh import/reset/render: `verify_libero.py`. Kết quả kiểm tra cuối được ghi bổ sung khi cài xong.

Sàng lọc runtime đã chạy trước yêu cầu ưu tiên cài: 10 cấu hình, mỗi cấu hình 2 warmup + 5 reps, cùng benchmark input 2×512²/16 tokens; output comparison dùng predict_check với input/noise cố định riêng. Raw log và summary.json nằm trong `runtime-screen/`. Chưa full re-run, chưa đo memory từng cấu hình hoặc SR, chưa chọn cấu hình tốt nhất. Hiện ưu tiên hoàn tất LIBERO.


### Tiếp tục cài bằng PyTorch CPU (người dùng đã chọn)

Người dùng xác nhận “Chuyển ngay sang PyTorch CPU cần dùng”. Cài `torch==2.5.1+cpu`, `torchvision==0.20.1+cpu` cho client; C++ vla-server CUDA không đổi. Đây là biến thể CPU của môi trường client so với lệnh PyPI mặc định trong docs, không báo torch.cuda.is_available=False là lỗi CUDA của server.

Đã tải wheel CPU từ index chính thức PyTorch. Lượt pip đầu timeout ở 62.4/174.7 MB; lượt sau tăng timeout lên 600 giây và bật --resume-retries 10 đã tải thành công hai wheel. Raw log: download-cpu-wheels-retry.log. Hai partial GPU trung gian không dùng được giữ bằng hardlink trong research-notes/tools/preserved-gpu-partials trước khi dừng installer mới PID 42602 và script chờ PID 43400. Cache hoàn tất ~/.cache/uv-vla và cache cũ ~/.cache/uv không bị xóa.

Pipeline mới `complete_cpu_route.sh`: cài wheel CPU -> requirements LIBERO từ cache -> editable LIBERO -> phần còn lại trong finish_libero.sh. Cài LeRobot 0.4.3 --no-deps rồi cài danh sách base dependency từ metadata chính thức, loại torch/torchvision để giữ phiên bản CPU đã chọn và tránh bản trung gian. Các pin cuối của repo vẫn được áp dụng sau đó. Có kiểm tra dependency (`uv pip check`), import client/adapters và reset/render, không coi package install success là đủ.

Tại lần tiếp tục sau khi người dùng xem log test fail: PyTorch CPU, requirements LIBERO, editable LIBERO và LeRobot đã cài; bộ cài đang bổ sung base dependency LeRobot. Log chính: complete-cpu-route.log. Chưa có kết quả reset/render hoặc SR tại thời điểm ghi này.


### Xác minh cài LIBERO đã đạt

`complete_cpu_route.sh` exit 0; `libero-verify.log` kết thúc bằng `LIBERO_INSTALL_AND_RENDER_OK`. Versions thực tế: Python 3.10.22; torch 2.5.1+cpu; torchvision 0.20.1+cpu; transformers 4.51.3; numpy 1.26.4; mujoco 2.3.2; robosuite 1.4.0; lerobot 0.4.3; gymnasium 0.29.1; pyarrow 12.0.1. Freeze lưu trong libero-freeze.txt.

Đã import client/adapters của repo; load suite libero_object có 10 task; reset task 0; render hai camera agentview/eye_in_hand đều (128,128,3); step dummy action thành công. Đây là xác minh cài đặt, không phải episode hoặc SR. LIBERO_CONFIG_PATH trỏ config riêng trong research-notes, không xóa/ghi đè ~/.libero.

`uv pip check` còn 7 incompatibilities từ các pin repo: pettingzoo và lerobot yêu cầu Gymnasium mới hơn; datasets yêu cầu PyArrow >=21; OpenCV headless 4.12 yêu cầu NumPy >=2; LeRobot yêu cầu TorchVision >=0.21; rerun-sdk yêu cầu NumPy >=2 và PyArrow >=18. Giữ raw libero-pip-check.log, không báo môi trường metadata sạch. Các module client/adapters và reset/render đã chạy thực tế với bộ pin này; chưa chứng minh mọi tính năng phụ (video dataset, rerun, PettingZoo) hoạt động.

### Smoke end-to-end đang chạy

Script run_smoke.py dùng server build-cuda, bind loopback 15556, GGUF gốc/defaults; client CPU, MUJOCO_GL=egl, seed42, libero_object/task0, n_episodes=2, n_action_steps=1. Lệnh/env trong smoke-original/commands.json; log server/client lưu cùng thư mục, video/summary trong smoke-original/outputs. Phân loại smoke, không dùng làm screening hay official SR.

Phát hiện mới trong startup server: warmup synthetic mặc định dùng ảnh 224², SmolVLA yêu cầu 512² nên log warmup predict failed; server vẫn ready. Chưa sửa source. Cần kiểm tra request thật từ client 512²; không dùng warmup này làm bằng chứng inference thành công. Startup warmup lỗi cũng không thay số baseline vla-bench trước đó (bench đã chỉ định size512).


### Kết quả smoke end-to-end đã hoàn tất

Client exit 0, tất cả episode hoàn tất, không abort/skip. Bản gốc/defaults, libero_object/task0, seed42, n_action_steps=1:

| Episode | Success | Số bước | Average inference/step do client báo |
|---|---|---:|---:|
| 0 | True | 134 | 151.03 ms |
| 1 | True | 159 | 137.71 ms |

Summary: 2/2 (100% trong smoke này), average inference/step 144.37 ms theo client. Không dùng số này làm SR chính thức hoặc thay baseline engine latency. CSV/video/summary ở smoke-original/outputs/smolvla/libero_object/task_0; lệnh/env trong smoke-original/commands.json. Hai MP4 đã tạo. Server đã tự dừng theo finally của runner.

Request thật chạy CUDA, server log trả total khoảng 116–120 ms ở các dòng quan sát, không thấy lỗi kích thước input trên request thật. Warmup synthetic startup 224² fail được giữ trong log, khác request thật 512².

Khoanh vùng observation warning: kiểm tra `gymnasium.spaces.Text(max_length=512).contains('pick up the alphabet soup and place it in the basket')` trả False; ký tự ngoài charset mặc định là dấu cách. Đã xác nhận nguyên nhân ít nhất ở trường task_description. Chưa sửa declaration hoặc dữ liệu observation; không khẳng định chỉ có nguyên nhân này khi chưa kiểm tra đầy đủ từng trường.

### Bảng runtime screen và cách đọc độ lệch

Đã lưu script summarize_runtime.py để tái tạo summary.json. Mean so trên screen cùng lượt (defaults mean111.9ms); không gộp screen 5 reps với baseline full20reps. Denoise3: mean89.2ms nhưng max_abs_real7=0.307970941, RMSE_real7=0.042685641; flash+mm-default: mean93.0ms, max_abs_real7=0.002767146, RMSE_real7=0.000614420. Chưa full rerun, memory từng cấu hình, smoke cho flags hoặc screening40episode; chưa chọn winner.

RMSE ban đầu của ablation và summary cũ tính trên 50×32 (bao gồm các cột padded zero). Script hiện giữ field legacy `rmse` và thêm `rmse_real7` trên 50×7 chiều hành động thực, để không làm RMSE nhỏ giả do padding. Max_abs không đổi khi bỏ các cột zero. Các chiều action có thể khác đơn vị; đây là độ lệch số học, không phải thước đo SR.

Đang chờ người dùng chọn baseline cho bước so sánh/SR tiếp theo: giữ defaults kèm công bố lỗi bit-exact, hoặc chuyển VLA_FUSED_OPS=0 và đo lại baseline. Chưa chạy screening/official trong lúc chờ quyết định này. Các log/notes có thể kiểm tra độc lập.


Đã kiểm tra hai MP4 bằng PyAV: episode_000000.mp4 có135frame, episode_000001.mp4 có160frame; khung đầu đều256×512, fps30. Raw video-verify.log. Frame count bằng số bước+1 do có frame reset. Không dùng tốc độ phát video làm tốc độ inference.


## Baseline đã được người dùng chốt

Người dùng chọn: **Giữ mặc định của repo, công bố rõ lỗi test và độ lệch fused/unfused**. Thay thế trạng thái chờ quyết định ở các mục trước. Giữ số baseline latency/memory cũ. Chuẩn bị screening defaults: libero_object task0–9, 4episode/task, seed42, n_action_steps1, initial state defaults của repo (episode_index0, stride1). Không cộng smoke vào40episode. Abort/missing episode sẽ dừng và đánh dấu incomplete, không âm thầm giảm mẫu số. Protocol/log/progress lưu ở smolvla-followup-20261104/screen40-defaults; script run_screen40.py. Các file on.txt/off.txt người dùng mới tạo giữ nguyên.

Screen40 defaults: task 0 hoàn tất 4/4; tổng tạm thời 4/4. Log screen40-defaults/task-0.log; chưa phải official SR.


Đã đọc paper SmolVLA/CLP và đối chiếu src/models/smolvla.cpp, viết ARCHITECTURE_AND_PROFILE.md (cùng thư mục smolvla-followup-20261104), có sơ đồ và phạm vi timing. Metadata checkpoint đọc trực tiếp, 791tensor, VLM32/expert32, vision12, hidden960/expert480. Chưa chạy phase/kernel profiling trong lúc GPU dành cho screening.

Phát hiện về seed SR: --seed42 cố định môi trường nhưng mặc định server SmolVLA rng khởi tạo random_device, noise mỗi predict không được ghim. Client có VLA_FIXED_NOISE_SEED tạo noise theo seed/episode/step. Đã hỏi người dùng chọn protocol; chưa đổi lượt hiện tại. Screening đang chạy giữ nguyên noise ngẫu nhiên mặc định. Seed42 môi trường không đảm bảo tái hiện action/trajectory từng bit; không gộp smoke2 vào screening40.

Screen40 defaults: task 1 hoàn tất 4/4; tổng tạm thời 8/8. Log screen40-defaults/task-1.log; chưa phải official SR.

Screen40 defaults: task 2 hoàn tất 4/4; tổng tạm thời 12/12. Log screen40-defaults/task-2.log; chưa phải official SR.

Screen40 defaults: task 3 hoàn tất 3/4; tổng tạm thời 15/16. Log screen40-defaults/task-3.log; chưa phải official SR.

Screen40 defaults: task 4 hoàn tất 4/4; tổng tạm thời 19/20. Log screen40-defaults/task-4.log; chưa phải official SR.

Screen40 defaults: task 5 hoàn tất 3/4; tổng tạm thời 22/24. Log screen40-defaults/task-5.log; chưa phải official SR.


Pruning exporter thử lần1 fail: AttributeError cannot set attribute tại renamed.name=name. Đã kiểm tra gguf_reader.py: ReaderTensor là NamedTuple bất biến. Log gốc smolvla-vision-tail-10-export.failed-1.log; không liên quan inference, GGUF gốc không đổi. Sửa bằng tensor._replace(name=name), giữ các field khác nguyên vẹn; kiểm tra payload/dtype/shape sau export. File partial nếu có được đổi tên .failed-1, không dùng đánh giá.


Chuẩn bị CPU đã hoàn tất: 6 GGUF Q8_0/Q5_0/Q4_0 × vision false/true bằng script quantize_gguf.py nguyên bản. Manifest quantization/manifest.json có size/hash/lệnh: mỗi bản791tensor, action expert289tensor được giữ raw; các tensor không eligible đúng dtype/shape/payload, metadata không đổi. Số tensor quant225 khi không vision,297 khi thêm vision. Chưa load GPU hoặc smoke từng GGUF.

Pruning prototype đã export thành công sau sửa NamedTuple: vision-tail10, vision-spaced10 (drop4,8), coupled-tail30 (drop30,31 đồng thời VLM/expert). Metadata số tầng được cập nhật, tensor giữ lại reindex và bit-preserve; không phải CLP, không CKA, không finetune. Sidecar JSON trong generated-models có mapping, removed tensor list, hash, size; pruning-summary.json tổng hợp. Chưa có inference/latency/SR. Không dùng kết quả export này để báo đã đạt mục6.

Screen40 defaults: task 6 hoàn tất 4/4; tổng tạm thời 26/28. Log screen40-defaults/task-6.log; chưa phải official SR.

Screen40 defaults: task 7 hoàn tất 4/4; tổng tạm thời 30/32. Log screen40-defaults/task-7.log; chưa phải official SR.


Đọc lại cặp on.txt/off.txt người dùng tạo (giữ nguyên file): mỗi file1600giá trị hữu hạn; max_abs=0.0005458742380142212, RMSE toàn32=0.00010049711168250623,350giá trị khác bit. Khớp phép đo trước của agent; không có exitcode riêng được lưu để xác minh lại. Result user-on-off-summary.json.

Đã chuẩn bị full_runtime_pair.py (3process/config,3warmup+20reps, interleave defaults/flash-mm và telemetry) và profile_phase.py (input khớp vla-bench; phase/none riêng; optional capture sauwarmup). Chưa chạy GPU vìscreeningđang hoạt động. Bộsummarize_kernels.py từ chối graph-only trace; rejection check thực tế lưu kernel-parser-graph-rejection.log.


Lưu ý tên quant -lm là nhãn ngắn cho mặc định script repo: 225tensor gồm224matrix VLM và mm.fc.weight (connector)1tensor. Action expert giữfloat; --vision thêm72matrix tower, tổng297. Không gọi mặc định này là chỉ VLM mà không có connector khi giải thích với mentor.

Screen40 defaults: task 8 hoàn tất 4/4; tổng tạm thời 34/36. Log screen40-defaults/task-8.log; chưa phải official SR.

Screen40 defaults: task 9 hoàn tất 4/4; tổng tạm thời 38/40. Log screen40-defaults/task-9.log; chưa phải official SR.


## Screening defaults hoàn tất: 38/40 =95.0%

libero_object task0–9 ×4episode, seed42 môi trường, n_action_steps1, noise server ngẫu nhiên mặc định. Không gộp smoke2; không phải official200. Không abort/missing; cả40MP4 mở được frame đầu, frame_count=steps+1. Artifact audit lưu screen40-artifact-audit.json.

| Task | Success | Steps |
|---|---:|---|
| 0 | 4/4 | 198, 175, 188, 174 |
| 1 | 4/4 | 127, 157, 138, 148 |
| 2 | 4/4 | 128, 125, 147, 120 |
| 3 | 3/4 | 500, 143, 138, 128 |
| 4 | 4/4 | 156, 157, 168, 156 |
| 5 | 3/4 | 220, 500, 222, 142 |
| 6 | 4/4 | 163, 161, 156, 145 |
| 7 | 4/4 | 133, 153, 146, 156 |
| 8 | 4/4 | 184, 159, 173, 162 |
| 9 | 4/4 | 133, 127, 142, 136 |

Hai failure ở task3 và5 đều chạm500step, is_success=False, tính vào mẫu số40. Chưa phân tích video để quy nguyên nhân cho model/quantization/kernel. Server screening đã dừng trước khi benchmark GPU tiếp theo. Baseline full-runtime-pair đo lại defaults và flash-mm theo3process/config,3warmup+20reps, interleave và telemetry riêng.


## Kết quả GPU bổ sung sau screening defaults

Full runtime pair3process/config,3warmup+20reps: defaults chọnmean113.9,p50113.9,p90115.4ms; flash+mm-default mean91.5,p5091.5,p9091.8ms, mean giảm19.67%. Trace allocations flash3process peak1040.268383MiB vs defaults1912.469456MiB, giảm45.61%tracked allocations, không phải toànresident VRAM. Telemetry lượt pair60–75°C, SM210–1980MHz, power11.32–108.15W, gồmload/idle. Không suy ra throttle chỉ từ các số này.

ModelGGUF CUDA fixed-input checksđãđạt cho cả9export. Full vla-bench candidate3process/config,3warmup+20reps, input2×512²/16token; memory nsys captureprocess-tree30childprocess (control+9model mỗi3process), auditPIDđủ và khôngfree thiếu/live âm/lost-drop. Controlpeak khớptraceđộc lập chínhxác0MiBchênh.

| Model | p50/p90 ms | Mean ms | Vision ms | Peak allocation MiB | Max abs real7 |
|---|---:|---:|---:|---:|---:|
| defaults | 113.7/114.6 | 113.7 | 56.1 | 1912.47 | 0 |
| smolvla-coupled-tail-30 | 110.3/110.9 | 110.4 | 56.0 | 1814.40 | 0.063353531 |
| smolvla-q4_0-lm | 110.2/111.0 | 110.0 | 56.1 | 865.07 | 0.08243826 |
| smolvla-q4_0-vision | 109.9/110.4 | 109.7 | 56.2 | 729.75 | 0.17617321 |
| smolvla-q5_0-lm | 110.5/111.2 | 110.6 | 56.0 | 903.98 | 0.033897966 |
| smolvla-q5_0-vision | 111.0/111.9 | 111.4 | 57.4 | 778.78 | 0.10021222 |
| smolvla-q8_0-lm | 109.3/110.3 | 109.5 | 56.0 | 1020.70 | 0.0084213912 |
| smolvla-q8_0-vision | 110.3/111.0 | 110.2 | 56.3 | 925.88 | 0.041252255 |
| smolvla-vision-spaced-10 | 104.5/105.7 | 104.7 | 47.0 | 1885.39 | 0.094964884 |
| smolvla-vision-tail-10 | 106.6/108.1 | 106.4 | 47.4 | 1885.39 | 1.7565101 |

NhữngmodelnàychưaquaLIBEROsmokeriêngtạithờiđiểmghi. Finitepredict khôngchứngminhSR. Benchvisionbao gồmconnector/preprocess/copy, khôngpureViTkernel. Variantquant-lm theo scriptbao gồmconnector, AEXgiữfloat.

Phase suite3process/case,3warmup+20reps đãthu; bestwhole-processmeanTotal: baseline192.79ms, flash176.34ms, visiontail211.71ms, visionspaced205.52ms, coupled203.10ms. Phasewallcóbiếnđộnglớn/overhead, khôngthaylatencyvla-bench. CóGPU nodetracesNONE20predictsau3warmupcho defaults/flash, bước tiếp theoPHASEnodetracesđểtáchGPUgraphspans.

Nsight NONE defaults: 160020kernel events; spanfirst-lasttargetactivity2541.99ms; targetactivitygap10.54%, maxgap16.75ms. KhôngphảiphầntrămtoànGPUidle; khôngsystem-wide, Pythonpacking/preprocess/driver có thể đóng góp.

Nsight NONE flash-mm: 194100kernel events; spanfirst-lasttargetactivity1921.82ms; targetactivitygap9.13%, maxgap9.26ms. KhôngphảiphầntrămtoànGPUidle; khôngsystem-wide, Pythonpacking/preprocess/driver có thể đóng góp.

QueryCUDA_GRAPH_EVENTS trên nodetrace ban đầu báo no such table; kiểm tra schema xác nhận node mode không cần table whole-graph. Chuyển sang kernel.graphId+correlationId, xác nhậnNONE60launches=20×[vision,vision,inference]. summarize_graph_modules.py dùng pattern/cardinalityguard và báoGPU first-lastkernelspan, khônggọiđólàCPU+GPUphasewall. PHASEdựkiến80launches=20×[vision,vision,prefill,denoise]; nếu khôngkhớp sẽkhônggánmodule.


Điều kiện GPU cần bổ sung: kiểm tra host hiện tại thấy server cũ PID23222 ở127.0.0.1:5555, --warmup0 --timing-detail phase, tồn tại hơn5giờ. Giữ nguyên vì không phải server do các runner mới quản lý. Truy vấn NVML trước từng thấyPIDnày. Không khẳng định GPU hoàn toàn cô lập; các runner chỉ bảo đảm không tự chạy đồng thời các workload do mình tạo. Process allocation theoCUPTI đã táchPID nên không cộng VRAMservercũ; timing có giới hạn vì chưa xác minh hoạt động request của servercũ trong từng lượt. Smoke mới dùng15558.


2026-11-04T14:22:51.535885+07:00: người dùng đã dừng server cũ5555; kiểm tra host xác nhận PID23222 không còn, chỉ còn smoke server15558 PID97266 tại lúc kiểm tra. Flash-mm smoke episode1 thành công155step, episode2 đang chạy. Không thay điều kiện các benchmark đã đo trước mốc này; các lần đo sau phải ghi riêng/control mới nếu so sánh thời gian.

Smoke flash-mm: complete, successes=1/2, steps=[155, 500]; logs candidate-smokes/flash-mm. Không phải screening hoặc official.


## Nsight PHASE attribution đã khoanh vùng và sửa

Bộgánmodule theo80graphlaunch bịtừchối cả4trace. Defaultchỉ54launch thayvì80; có98033kernel graphIdNULL. Khôngphải mấttrace: 98033=13×7541 (13predict main trực tiếp), cộng7predict main bằnggraph và40visionlaunch. Sourceggml-cuda.cu4466–4501 xác nhận khi properties thayđổi graphwarmupreset và chạytrựctiếp đếnkhiổnđịnh. Chưa khẳngđịnh thuộc tính cụ thể gâyđổi, nhưng đãcóbằngchứng modePHASE táidựnggraph vàmixdirect/graph; khôngép gán sai.

Đã dùngmốcKVcopytừsource: prefix145token=2×64+16+1, head64×KVheads5×float4bytes ->185600bytes/copy, 2copy/layer. Tracemỗi32layer có1280D2Dcopies=20predict×64; coupled30có1200=20×60. Vision40launch được xácnhận; chiaGPUkernels theo biênKVcopy và kiểmtra khôngtrùng/thiếu bất kỳkernel nào. Script summarize_phase_boundaries.py.

| Config | Vision+connector GPU active ms | KV prefill GPU active ms | Denoise+crossKV GPU active ms |
|---|---:|---:|---:|
| defaults | 52.708 | 14.956 | 36.318 |
| flash-mm | 40.520 | 11.687 | 33.475 |
| smolvla-vision-spaced-10 | 43.740 | 14.920 | 36.591 |
| smolvla-coupled-tail-30 | 52.722 | 14.095 | 34.531 |

Mỗitrace20predict sau3warmup. Đây là unionthờigian thực thi kernel GPU processđích theo từngmoduleproxy, loại khoảngCPUdispatch/graphbuild/copy; khôngphải walllatency hay phânrã chínhxácbaseline vìPHASE dùng16tokenthực thay48slotpadded. GPUspan cũngđược lưu riêng và cóthể baogồm khoảngchờlaunchtrựctiếp. Denoise gồm crossKVreprojection/actionprojections; vision gồm connector nhưng khôngCPUpreprocess/inter-viewcopy. Kết quả phùhợpvới vla-bench: visionprunegần8%end2end, coupledgần3%, chưa chứngminhSRđượcgiữ.

Cácsmoke mớiđangchạy: flash-mm complete1/2 (155,500steps), khôngabort. Chưa đủ mẫuđể kếtluậnSRgiảm, khôngchọntốt nhấtchỉtheolatency.

Smoke smolvla-coupled-tail-30: complete, successes=0/2, steps=[500, 500]; logs candidate-smokes/smolvla-coupled-tail-30. Không phải screening hoặc official.

Smoke smolvla-q4_0-lm: complete, successes=1/2, steps=[148, 500]; logs candidate-smokes/smolvla-q4_0-lm. Không phải screening hoặc official.

Smoke smolvla-q4_0-vision: complete, successes=0/2, steps=[500, 500]; logs candidate-smokes/smolvla-q4_0-vision. Không phải screening hoặc official.

Smoke smolvla-q5_0-lm: complete, successes=1/2, steps=[159, 500]; logs candidate-smokes/smolvla-q5_0-lm. Không phải screening hoặc official.

Smoke smolvla-q5_0-vision: complete, successes=0/2, steps=[500, 500]; logs candidate-smokes/smolvla-q5_0-vision. Không phải screening hoặc official.

Smoke smolvla-q8_0-lm: complete, successes=2/2, steps=[193, 147]; logs candidate-smokes/smolvla-q8_0-lm. Không phải screening hoặc official.

Smoke smolvla-q8_0-vision: complete, successes=2/2, steps=[146, 236]; logs candidate-smokes/smolvla-q8_0-vision. Không phải screening hoặc official.

Đã khởi chạy queue_official200.py: tự chờ smoke complete và port15558 đóng rồi chạy baseline 10task×20episode trên port15559. Giữ repo defaults, seed42 môi trường, server noise ngẫu nhiên. Log official200-runner.log; protocol/progress/result riêng official200-defaults. Không chạy GPU song song với smoke.

Smoke smolvla-vision-spaced-10: complete, successes=0/2, steps=[500, 500]; logs candidate-smokes/smolvla-vision-spaced-10. Không phải screening hoặc official.

Smoke smolvla-vision-tail-10: complete, successes=0/2, steps=[500, 500]; logs candidate-smokes/smolvla-vision-tail-10. Không phải screening hoặc official.

Official200 defaults: task 0 hoàn tất 18/20; tổng tạm thời 18/20. Log official200-defaults/task-0.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 1 hoàn tất 20/20; tổng tạm thời 38/40. Log official200-defaults/task-1.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 2 hoàn tất 17/20; tổng tạm thời 55/60. Log official200-defaults/task-2.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 3 hoàn tất 17/20; tổng tạm thời 72/80. Log official200-defaults/task-3.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 4 hoàn tất 19/20; tổng tạm thời 91/100. Log official200-defaults/task-4.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 5 hoàn tất 13/20; tổng tạm thời 104/120. Log official200-defaults/task-5.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 6 hoàn tất 16/20; tổng tạm thời 120/140. Log official200-defaults/task-6.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 7 hoàn tất 20/20; tổng tạm thời 140/160. Log official200-defaults/task-7.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Official200 defaults: task 8 hoàn tất 20/20; tổng tạm thời 160/180. Log official200-defaults/task-8.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Recovery official200: original task9 interrupted at episode4 by zmq.error.Again after SIGSTOP/SIGCONT; exact cause not proven. Original task0–8 retained (160/180). Restart task9 all20 in official200-defaults/recovery-task9; exclude original partial3 successes from denominator. Explicit --recv-timeout-ms -1 disables receive deadline; requires manual intervention if server hangs. Client now checkpoints episodes.csv atomically after each completed episode. Original logs/videos preserved. Recovery runner appends official200-runner.log and publishes result.json only after all200 counted episodes.

Official200 defaults: task 9 hoàn tất 17/20; tổng tạm thời 177/200. Log official200-defaults/task-9.log; SR tạm thời; chỉ chốt sau đủ200 episode.

Phần3 hoàn tất bổ sung: run_part3.sh exit0; latency30process, memory30process, audit không thiếu PID/không unmatched free/không lost-drop theo diagnostic. Report PART3_OPTIMIZATION.md và raw part3-runtime/. Output delta tái sử dụng fixed harness đã lưu, không benchmark lẫn input.

Khởi chạy run_screen_variants.py theo yêu cầu người dùng: flash-mm, q8-lm, q8-vision tuần tự, mỗi40episode; port15560; no VLA env inherited; offline cached tokenizer. Không ghi đè baseline và không đổi file GGUF.

Screen40 flash-mm: task 0 hoàn tất 4/4; tổng tạm thời 4/4. Log screen40-variants/flash-mm/task-0.log; chưa phải official SR.

Screen40 flash-mm: task 1 hoàn tất 4/4; tổng tạm thời 8/8. Log screen40-variants/flash-mm/task-1.log; chưa phải official SR.

Screen40 flash-mm: task 2 hoàn tất 3/4; tổng tạm thời 11/12. Log screen40-variants/flash-mm/task-2.log; chưa phải official SR.

Screen40 flash-mm: task 3 hoàn tất 4/4; tổng tạm thời 15/16. Log screen40-variants/flash-mm/task-3.log; chưa phải official SR.

Screen40 flash-mm: task 4 hoàn tất 4/4; tổng tạm thời 19/20. Log screen40-variants/flash-mm/task-4.log; chưa phải official SR.

Screen40 flash-mm: task 5 hoàn tất 1/4; tổng tạm thời 20/24. Log screen40-variants/flash-mm/task-5.log; chưa phải official SR.

Screen40 flash-mm: task 6 hoàn tất 4/4; tổng tạm thời 24/28. Log screen40-variants/flash-mm/task-6.log; chưa phải official SR.

Screen40 flash-mm: task 7 hoàn tất 4/4; tổng tạm thời 28/32. Log screen40-variants/flash-mm/task-7.log; chưa phải official SR.

Screen40 flash-mm: task 8 hoàn tất 4/4; tổng tạm thời 32/36. Log screen40-variants/flash-mm/task-8.log; chưa phải official SR.

Screen40 flash-mm: task 9 hoàn tất 3/4; tổng tạm thời 35/40. Log screen40-variants/flash-mm/task-9.log; chưa phải official SR.

Screen40 q8-lm: task 0 hoàn tất 4/4; tổng tạm thời 4/4. Log screen40-variants/q8-lm/task-0.log; chưa phải official SR.

Screen40 q8-lm: task 1 hoàn tất 4/4; tổng tạm thời 8/8. Log screen40-variants/q8-lm/task-1.log; chưa phải official SR.

Screen40 q8-lm: task 2 hoàn tất 3/4; tổng tạm thời 11/12. Log screen40-variants/q8-lm/task-2.log; chưa phải official SR.

Screen40 q8-lm: task 3 hoàn tất 4/4; tổng tạm thời 15/16. Log screen40-variants/q8-lm/task-3.log; chưa phải official SR.

Screen40 q8-lm: task 4 hoàn tất 4/4; tổng tạm thời 19/20. Log screen40-variants/q8-lm/task-4.log; chưa phải official SR.

Screen40 q8-lm: task 5 hoàn tất 3/4; tổng tạm thời 22/24. Log screen40-variants/q8-lm/task-5.log; chưa phải official SR.

Screen40 q8-lm: task 6 hoàn tất 4/4; tổng tạm thời 26/28. Log screen40-variants/q8-lm/task-6.log; chưa phải official SR.

Screen40 q8-lm: task 7 hoàn tất 4/4; tổng tạm thời 30/32. Log screen40-variants/q8-lm/task-7.log; chưa phải official SR.

Screen40 q8-lm: task 8 hoàn tất 4/4; tổng tạm thời 34/36. Log screen40-variants/q8-lm/task-8.log; chưa phải official SR.

Screen40 q8-lm: task 9 hoàn tất 2/4; tổng tạm thời 36/40. Log screen40-variants/q8-lm/task-9.log; chưa phải official SR.

Screen40 q8-vision: task 0 hoàn tất 4/4; tổng tạm thời 4/4. Log screen40-variants/q8-vision/task-0.log; chưa phải official SR.

## Bổ sung theo yêu cầu điền đủ6task và commit/push

Người dùng cho phép bổ sung tính toán ở phần thiếu và commit/push GitHub. Đã tạo CALCULATIONS_SMOLVLA.md và script complete_calculations.py; cập nhật REPORT/SUBMISSION/PART3. Q8-lm result mới xác nhận36/40, thay snapshot34/36. Q8-vision đang chạy, không dừng hoặc can thiệp runner. Số liệu dự toán official flash-mm175/200 và Q8-lm180/200 chỉ là200×SRscreening, không phải episode đã chạy. Không suy residentVRAM, SR từ synthetic error hoặc tự ghi testpass. Lệnh tái lập: python3 research-notes/smolvla-followup-20261104/complete_calculations.py. Lệnh `python` không có trên PATH, dùng `python3` thành công. Giữ nguyên on.txt/off.txt và dữ liệu runner đang ghi; chỉ stage tài liệu/script và snapshot result Q8-lm hoàn tất.

Screen40 q8-vision: task 1 hoàn tất 4/4; tổng tạm thời 8/8. Log screen40-variants/q8-vision/task-1.log; chưa phải official SR.

Screen40 q8-vision: task 2 hoàn tất 4/4; tổng tạm thời 12/12. Log screen40-variants/q8-vision/task-2.log; chưa phải official SR.

Screen40 q8-vision: task 3 hoàn tất 4/4; tổng tạm thời 16/16. Log screen40-variants/q8-vision/task-3.log; chưa phải official SR.

Screen40 q8-vision: task 4 hoàn tất 3/4; tổng tạm thời 19/20. Log screen40-variants/q8-vision/task-4.log; chưa phải official SR.

Screen40 q8-vision: task 5 hoàn tất 3/4; tổng tạm thời 22/24. Log screen40-variants/q8-vision/task-5.log; chưa phải official SR.

Screen40 q8-vision: task 6 hoàn tất 4/4; tổng tạm thời 26/28. Log screen40-variants/q8-vision/task-6.log; chưa phải official SR.

Screen40 q8-vision: task 7 hoàn tất 3/4; tổng tạm thời 29/32. Log screen40-variants/q8-vision/task-7.log; chưa phải official SR.

Screen40 q8-vision: task 8 hoàn tất 3/4; tổng tạm thời 32/36. Log screen40-variants/q8-vision/task-8.log; chưa phải official SR.

Screen40 q8-vision: task 9 hoàn tất 4/4; tổng tạm thời 36/40. Log screen40-variants/q8-vision/task-9.log; chưa phải official SR.

## Chốt tài liệu sau screening — 05/11/2026 (UTC+07)

Batch ba biến thể đã complete, 120/120 episode, cả ba exit 0: flash-mm 35/40, Q8-lm 36/40, Q8-vision 36/40. Q8-vision task 9 đạt 4/4, không skip. Trạng thái chạy trong các dòng nhật ký trước là lịch sử.

Đã cập nhật REPORT_SMOLVLA.md, SUBMISSION_SMOLVLA.md, PART3_OPTIMIZATION.md, CALCULATIONS_SMOLVLA.md, MENTOR_REVIEW.md và PAPER_READING_SMOLVLA_CLP.md. Phụ lục có thêm Q8-vision, Wilson 95% 76,9–96,0%; 180/200 vẫn chỉ là ngoại suy, chưa phải official. Báo cáo có bảng tiêu chí đóng các thực nghiệm còn mở.

Thêm audit_screening.py: đối chiếu result.json với 40 CSV của defaults và ba biến thể (160 episode), kiểm tra ID episode/task, steps, success, aborted, SR và exit batch. Kết quả khớp: defaults 38/40; flash-mm 35/40; Q8-lm 36/40; Q8-vision 36/40; tất cả aborted=0. screening-final-audit.json lưu số đo từng episode và SHA256 CSV; không kiểm tra video. submission-progress.json được cập nhật từ result cuối.

Kiểm tra: audit_screening.py pass; complete_calculations.py sinh lại phụ lục giống nhau; sáu tài liệu không có local link mất target hoặc trạng thái Q8-vision cũ; hai script parse thành công; git diff --check sạch. Không chạy lại benchmark hoặc CUDA tests trong lượt chốt tài liệu này. Hai CUDA tests vẫn fail theo log đã có; official biến thể, repeated-seed, CKA/finetuning, connector/attention riêng, lock sạch và resident VRAM vẫn mở. Giữ nguyên on.txt/off.txt và dữ liệu thí nghiệm.


## Bổ sung rà soát tính hoàn chỉnh — 05/11/2026

Đã thay checklist đầu nhật ký đã lỗi thời bằng trạng thái cuối, bổ sung bảng theo task và audit tái lập cho cả official lẫn screening. `audit_submission.py` kiểm tra 360 episode từ manifest cuối, gồm task 9 recovery; lưu SHA256 từng CSV. Đây là xác minh dữ liệu đã có, không phải chạy thêm rollout. Các giới hạn thực nghiệm vẫn giữ nguyên theo phạm vi chốt báo cáo.

Screen40 smolvla-vision-tail-10: task 0 hoàn tất 0/4; tổng tạm thời 0/4. Log screen40-variants/smolvla-vision-tail-10/task-0.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 1 hoàn tất 0/4; tổng tạm thời 0/8. Log screen40-variants/smolvla-vision-tail-10/task-1.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 2 hoàn tất 0/4; tổng tạm thời 0/12. Log screen40-variants/smolvla-vision-tail-10/task-2.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 3 hoàn tất 0/4; tổng tạm thời 0/16. Log screen40-variants/smolvla-vision-tail-10/task-3.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 4 hoàn tất 0/4; tổng tạm thời 0/20. Log screen40-variants/smolvla-vision-tail-10/task-4.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 5 hoàn tất 0/4; tổng tạm thời 0/24. Log screen40-variants/smolvla-vision-tail-10/task-5.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 6 hoàn tất 0/4; tổng tạm thời 0/28. Log screen40-variants/smolvla-vision-tail-10/task-6.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 7 hoàn tất 0/4; tổng tạm thời 0/32. Log screen40-variants/smolvla-vision-tail-10/task-7.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 8 hoàn tất 0/4; tổng tạm thời 0/36. Log screen40-variants/smolvla-vision-tail-10/task-8.log; chưa phải official SR.

Screen40 smolvla-vision-tail-10: task 9 hoàn tất 0/4; tổng tạm thời 0/40. Log screen40-variants/smolvla-vision-tail-10/task-9.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 0 hoàn tất 0/4; tổng tạm thời 0/4. Log screen40-variants/smolvla-vision-spaced-10/task-0.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 1 hoàn tất 0/4; tổng tạm thời 0/8. Log screen40-variants/smolvla-vision-spaced-10/task-1.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 2 hoàn tất 0/4; tổng tạm thời 0/12. Log screen40-variants/smolvla-vision-spaced-10/task-2.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 3 hoàn tất 0/4; tổng tạm thời 0/16. Log screen40-variants/smolvla-vision-spaced-10/task-3.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 4 hoàn tất 0/4; tổng tạm thời 0/20. Log screen40-variants/smolvla-vision-spaced-10/task-4.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 5 hoàn tất 0/4; tổng tạm thời 0/24. Log screen40-variants/smolvla-vision-spaced-10/task-5.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 6 hoàn tất 0/4; tổng tạm thời 0/28. Log screen40-variants/smolvla-vision-spaced-10/task-6.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 7 hoàn tất 0/4; tổng tạm thời 0/32. Log screen40-variants/smolvla-vision-spaced-10/task-7.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 8 hoàn tất 0/4; tổng tạm thời 0/36. Log screen40-variants/smolvla-vision-spaced-10/task-8.log; chưa phải official SR.

Screen40 smolvla-vision-spaced-10: task 9 hoàn tất 0/4; tổng tạm thời 0/40. Log screen40-variants/smolvla-vision-spaced-10/task-9.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 0 hoàn tất 0/4; tổng tạm thời 0/4. Log screen40-variants/smolvla-coupled-tail-30/task-0.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 1 hoàn tất 0/4; tổng tạm thời 0/8. Log screen40-variants/smolvla-coupled-tail-30/task-1.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 2 hoàn tất 3/4; tổng tạm thời 3/12. Log screen40-variants/smolvla-coupled-tail-30/task-2.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 3 hoàn tất 3/4; tổng tạm thời 6/16. Log screen40-variants/smolvla-coupled-tail-30/task-3.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 4 hoàn tất 2/4; tổng tạm thời 8/20. Log screen40-variants/smolvla-coupled-tail-30/task-4.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 5 hoàn tất 2/4; tổng tạm thời 10/24. Log screen40-variants/smolvla-coupled-tail-30/task-5.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 6 hoàn tất 4/4; tổng tạm thời 14/28. Log screen40-variants/smolvla-coupled-tail-30/task-6.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 7 hoàn tất 2/4; tổng tạm thời 16/32. Log screen40-variants/smolvla-coupled-tail-30/task-7.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 8 hoàn tất 4/4; tổng tạm thời 20/36. Log screen40-variants/smolvla-coupled-tail-30/task-8.log; chưa phải official SR.

Screen40 smolvla-coupled-tail-30: task 9 hoàn tất 3/4; tổng tạm thời 23/40. Log screen40-variants/smolvla-coupled-tail-30/task-9.log; chưa phải official SR.

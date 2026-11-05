# Timer và phạm vi phép đo

Source `src/models/smolvla.cpp:1307` dùng `std::chrono::steady_clock`: đồng hồ CPU wall time, không phải CUDA event.

| Field | Biên source | Thực sự bao gồm |
|---|---|---|
| ms_vision |1357→1398|Graph setup/allocation, preprocess từng view, upload, vision+connector compute, download embedding; không tách connector|
| ms_prefill |1538→1543|Compute graph tạo prefix K/V, gồm launch/wait; chỉ khi PHASE|
| ms_denoise |1554→1561|Compute expert/10 Euler steps và dependency cross-KV projection; không riêng action expert|
| ms_inference |1562|prefill+compute còn lại; default chứa cả prefix và denoise trong một graph|
| ms_total |1308→1574|Toàn predict_impl đến trước return; thêm graph setup, host state/noise/language/masks,copies,denormalize|
| bench latency |src/serving/vla-bench.cpp194→202|Wall time quanh `vla::predict`; không model loading, network, simulator hoặc robot execution|

**Có đồng bộ**: `build-cuda/_deps/llama-src/ggml/src/ggml-backend.cpp:455–458`: `ggml_backend_graph_compute_async` rồi `ggml_backend_synchronize`. CUDA timer CPU vì vậy chờ hoàn thành graph; thời gian còn có CPU dispatch, chờ và inter-kernel gaps, không phải pure GPU active. Source dependency trong build đã đọc; chưa chứng minh binary khớp chính source dependency này bằng rebuild.

PHASE không chỉ bật đồng hồ:1403 chọn `in.n_lang`16 thay vì cfg48;1438 tạo graph với KV leaves,1545–1547 copy K/V sang leaves,1549 dựng graph denoise. Default dùng cached main graph1417–1430. Graph setup/copy giữa hai timer không được phân bổ vào prefill/denoise. Vì vậy tổng phase không bằng total và không lấy phase measurements để chia baseline default114,1ms.

`--timing-detail none|phase` là flag server (`src/serving/server.cpp:282–288`), không có trong vla-bench hiện tại. Python `predict(timing=TIMING_PHASE)` đi qua C API (`src/vla_c_api.cpp:154`) và Inputs. Không tự thêm flag này vào bench. API định nghĩa `include/vla.h:129`; field zero khi không đo không có nghĩa module miễn phí.

Protocol phase lịch sử: `profile_phase.py`:2×512,IDs1..16,state/noise cố định,3warmup+20reps;3process mỗi cấu hình. Means của60 samples:

| CPU wall phase ms | Defaults | Flash-mm |
|---|---:|---:|
| Vision+connector+host/copies |57.535|43.985|
| Prefill |20.891|15.131|
| Denoise+crossKV |108.414|84.584|
| Total |235.682|187.677|
| Total−vision−inference |48.842|43.977|

Các outlier phase lớn và path khác default: không dùng để tuyên bố tốc độ module chính xác cho production. Dữ liệu sample nằm trong `../smolvla-followup-20261104/phase-suite/{defaults,flash-mm}-*/result.json`. Mean nhiều biến có thể cộng; p50/p90 của từng biến không nhất thiết cộng thành percentile total.

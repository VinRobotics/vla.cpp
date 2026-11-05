# Nsight Systems: tái phân tích SQLite ngày04/11

Scope:20predict sau3warmup,capture cudaProfilerApi,node trace,default path;không load/warmup. [Commands gốc](../smolvla-followup-20261104/kernel-traces/commands.json), [.nsys-rep baseline](../smolvla-followup-20261104/kernel-traces/defaults.nsys-rep), [.nsys-rep flash-mm](../smolvla-followup-20261104/kernel-traces/flash-mm.nsys-rep).

Audit mở SQLite mode=ro,tính lại union intervals và tổng duration;khớp số cũ. **Cả hai trace cảnh báo có thể không thu đủ CUDA events**,vì vậy kết luận chỉ có điều kiện trên dữ liệu lưu. Không có proof zero dropped events.

| Metric (20predict) | Defaults | Flash-mm |
|---|---:|---:|
|Kernel events|160020.000|194100.000|
|Summed kernel duration ms|2260.227|1733.185|
|Activity span ms|2541.987|1921.821|
|Activity union ms|2274.033|1746.418|
|Gap ms|267.954|175.403|
|Max gap ms|16.749|9.262|

|Name category,sum ms|Defaults|Flash-mm|
|---|---:|---:|
|other|326.800|503.440|
|GEMM_including_attention_products|1317.749|898.702|
|softmax_or_RoPE|446.448|39.605|
|norm|39.288|58.461|
|attention_named|129.942|232.978|

GEMM bao gồm attention products; attention_named bao gồm fused/layout;softmax/RoPE không toàn bộ attention;other có activation/add/cast/layout chưa map chính xác. Không coi các category là module latency. Ví dụ baseline top softmax_rows_loop402,82ms;flash top attention kernel flash_attn_ext_f16 khoảng214,35ms. Source311–335 xác nhận vision đổi explicit→flash,còn VLM/expert dùng flash op trong attn_mlp674–686 ở cả hai cấu hình. Không gán tất cả flash kernels riêng cho vision.

CUDA API:360cudaStreamSynchronize mỗi trace,sum2096,295/1582,760ms;60cudaGraphLaunch,sum227,511/227,927ms;300cudaMemcpyAsync,sum31,281/33,053ms. CPU API duration overlap GPU execution,không cộng vào kernel duration và không gọi synchronize duration là overhead độc lập. Chưa đo riêng pure launch overhead/launch-to-start latency.

GPU memcpy:340events,279408480bytes mỗi trace;sum17,340/16,394ms. API calls và GPU memcpy event count khác không tự chứng minh mất copies (graph có node/mapping riêng). Copy sum không tương đương critical-path cost.

Gaps=span−union(kernel+memcpy+memset),từ activity đầu tới cuối của process trong capture;không gồm edge gaps.10,54→9,13%;max16,749→9,262ms. Đây là khoảng trống **recorded process activity**,không hardware utilization,toàn GPU idle hay proof CPU bottleneck. Thiếu event/instrumentation/process khác có thể ảnh hưởng. Không chỉ cộng kernel durations để tính busy nếu overlap.

Phase attribution được chạy lại dùng vision graphID/correlation và D2D K/V copy boundary:prefix145,64copies/prediction,1280copies/20;bytes/copy145×64×5×4=185600. Analyzer đảm bảo recorded kernels không thiếu/trùng khi chia vision,prefill,denoise. Điều này xác minh cách phân bổ dữ liệu đã lưu,không chứng minh toàn execution đã được capture đầy đủ. Đọc phase timing methodology trước khi đối chiếu default throughput.

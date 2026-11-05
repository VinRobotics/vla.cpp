# Defaults vs ứng viên runtime flash-mm

Best ở đây là ứng viên runtime giữ10steps đã có đo lặp, không best quality hoặc mọi cấu hình khả dĩ. Precision arithmetic khác có chủ đích; model/input/steps giữ nguyên. Không gộp số từ các population khác nhau.

| Metric | Baseline | Flash-mm | Difference best−base |
|---|---:|---:|---:|
|E2E predict mean (median 3 runs), ms|114.100|92.400|-21.700|
|p50 (median 3 run p50), ms|114.100|92.500|-21.600|
|p90 (median 3 run p90), ms|115.400|93.000|-22.400|
|Vision+connector wall (median 3 means), ms|56.100|43.600|-12.500|
|VLM-only wall default|N/A|N/A|Default không tách|
|Action-expert-only wall|N/A|N/A|Denoise gồm crossKV/solver|
|Denoising-only wall default|N/A|N/A|Không tách default|
|Peak process resident VRAM|N/A|N/A|NVML process memory N/A|
|Exact full attention time|N/A|N/A|Kernel-name không đủ semantic mapping|
|Whole-GPU idle fraction|N/A|N/A|Target trace không bao phủ mọi process/events|

Mean đại diện giảm21,7ms =19,02%,speedup1,235×. Vision giảm12,5ms; đây là chênh median thống kê,không phải phân rã cặp sample chính xác.

Các phép đo riêng (không cộng vào bảng default wall):

| Metric / population | Defaults | Flash-mm |
|---|---:|---:|
|PHASE CPU prefill mean60samples,ms|20.891|15.131|
|PHASE CPU denoise+crossKV mean60samples,ms|108.414|84.584|
|PHASE trace vision+connector GPU active mean20,ms|52.708|40.520|
|PHASE trace prefill GPU active mean20,ms|14.956|11.687|
|PHASE trace denoise+crossKV GPU active mean20,ms|36.318|33.475|
|Default-path trace GEMM group summed20,ms|1317.749|898.702|
|Default-path trace attention-named group summed20,ms|129.942|232.978|
|Default-path trace gaps recorded activity,%|10.541|9.127|
|Peak traced CUDA allocation,MiB (3memory traces each)|1912.469|1040.268|
|LIBERO screening success,40episodes|38|35|

Allocation metric: Device+Device Static tracked allocations,không resident VRAM. Nguồn `../smolvla-vram-20261104/memory-summary.json`, `../smolvla-followup-20261104/memory-flash-mm/memory-summary.json`;không tính lại memory lifecycle trong audit mới. GPU phase active được kiểm tra lại với boundary analyzer; không tính host/copy/gaps. Kernel categories có completeness warning.

Kernel mix thay đổi:attention-named tăng không đồng nghĩa attention tổng chậm hơn;explicit softmax/matmul có thể chuyển sang fused attention. Chưa semantic attribution toàn bộ nên không kết luận bottleneck chuyển module nào. GEMM vẫn là nhóm tên có tổng lớn nhất trong recorded events; chưa chứng minh compute-bound hay tensor-core saturation.

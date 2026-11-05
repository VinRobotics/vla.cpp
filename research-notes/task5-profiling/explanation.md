# Giải thích toàn bộ bằng ví dụ

## Từ camera tới hành động

**Inference** là dùng trọng số đã học để tính output, không cập nhật trọng số. Trong bài này một inference nhận2ảnh, instruction và state rồi trả chunk50actions. Latency chunk không đồng nghĩa thời gian chạy robot50bước.

**Vision encoder** biến ảnh thành feature: ảnh512 chia patch16 tạo1024patch. SigLIP dùng attention/FFN để mỗi patch mang thông tin ngữ cảnh, ví dụ nhận ra vật cần gắp.1024tokens khiến attention lớn; đây là phần tiêu tốn đáng kể trong số đo.

**Connector** nối không gian feature ảnh với VLM. Gom nhóm4×4patch thành feature12288 rồi project960, còn64tokens mỗi ảnh. Ví dụ2ảnh cho128visual tokens. Token ít hơn làm prefix attention nhỏ hơn; connector không được đo riêng trong ms_vision.

**VLM** là mạng vision-language: ghép visual tokens, instruction embeddings và state token, tạo K/V context cho expert. Với16token ngôn ngữ, default pad48 nên prefix128+48+1=177;PHASE chỉ145. Do đó chỉ bật timing cũng có thể thay workload.

**Attention** cho mỗi query lấy tổng có trọng số của values theo mức phù hợp với keys: softmax(QKᵀ/√d)V. Ví dụ token lệnh “cup” đối chiếu vùng ảnh chứa cốc. Self-attention xử lý quan hệ trong chuỗi; cross-attention expert đọc context prefix. Chi phí có matrix multiply, softmax, layout và memory traffic; không chỉ một kernel tên attention.

**Action expert** chuyển context và action đang ước lượng thành velocity. Ở checkpoint này hidden480,32layers,SA/CA xen kẽ; output50×32, chỉ7cột thực. Nó chạy lại mỗi bước denoise nên giảm số bước tác động trực tiếp số lần tính expert.

**Denoising / flow matching** bắt đầu từ noise rồi tích phân trường velocity đã học. Mỗi bước x←x−0.1v;10bước để ra chunk. Đây là suy luận flow matching, không phải lọc nhiễu ảnh và không phải sampling text. Giảm10→3bước có thể nhanh nhưng đổi độ chính xác, không phải tối ưu kernel tương đương.

**Prefill/decode:** prefill ở repo là tính context K/V một lần. Không có vòng decode instruction từng token; expert xử lý toàn chunk ở từng solver step. Cache được dùng lại nên không chạy lại vision/VLM10lần.

## GPU làm việc thế nào

**GEMM** là phép nhân ma trận tổng quát, ví dụ token feature X nhân weight W trong linear projector. Cả FFN lẫn QKᵀ/AV của attention đều có thể dùng matmul, nên GEMM≠FFN/VLM. Đây là nhóm lớn nhất trong thời gian kernel được trace, chưa chứng minh giới hạn FLOPs.

**CUDA kernel** là hàm được CPU gửi để nhiều GPU threads thực thi; ví dụ softmax1024phần tử mỗi hàng. Một layer có nhiều kernel; nhiều kernel hơn không tự động chậm hơn: flash-mm có194100events so với160020 nhưng tổng duration thấp hơn.

**Kernel launch overhead** là chi phí gửi/lập lịch công việc;ví dụ CPU gọi cudaGraphLaunch để replay nhiều node. API duration không phải latency từ CPU tới GPU bắt đầu, cũng không phải pure launch overhead. Trace này ghi60graph launches/20predictions; tổng API khoảng227,5/227,9ms dưới profiler. Không quy toàn số này thành chi phí loại bỏ được.

**CPU/GPU synchronization** là host chờ công việc GPU xong hoặc đảm bảo thứ tự. `graph_compute` đồng bộ trước khi trả, nên CPU timer thấy cả thời gian chờ.360StreamSynchronize API calls có tổng2096,3/1582,8ms nhưng overlap kernel execution; không cộng thêm vào kernel time.

**GPU idle gap** là khoảng giữa activity intervals. Ta lấy hợp kernel+memcpy+memset để không đếm overlap hai lần.10,54%/9,13% là khoảng không có activity của tiến trình trong trace, không chứng minh GPU toàn máy nhàn vì có thể có process khác hoặc event thiếu.

**Latency p50/p90** là phân vị: p50 trung vị, p90 có khoảng90%samples không vượt giá trị đó (tùy nội suy). Ví dụ p90 cao hơn mean nhiều báo tail spikes. Median của3run p90 không phải p90 gộp60samples. Phase số lớn/outliers không được thay cho steady-state default benchmark.

**NVTX** là annotation CPU code, ví dụ range “vision”, giúp nối timeline với module. Nó không tự đo toàn bộ kernel GPU, cần correlation và xử lý bất đồng bộ/graph. Không thêm NVTX ở lượt này vì đã có source/cache boundary để phân tích thận trọng; nếu cần connector riêng thì phải instrument và kiểm chứng overhead.

**Nsight Systems** ghi timeline CPU API, GPU kernels, transfers và annotations để xem công việc nối tiếp/chồng lấp/chờ. Nó giúp tìm nơi đáng điều tra; hardware counters chi tiết/occupancy không được suy ra từ bảng duration. **WSL2 profiling** chạy qua Windows driver và Linux guest; CLI/CUPTI hoạt động trong artifact này nhưng NVML không đầy đủ, hardware tracing unavailable và có completeness warning. Không kết luận mọi tính năng WSL đều không dùng được.

## Đọc kết quả đúng

Quan sát: runtime flash-mm nhanh hơn khoảng19%;vision wall56,1→43,6ms;GEMM recorded1317,75→898,70ms/20predict;softmax/RoPE446,45→39,60ms. Giải thích hợp lý: flash path thay phép tính vision attention, mm precision đổi kernel/weights. Chưa chứng minh từng phần đóng góp bao nhiêu bằng thí nghiệm đơn biến trong cặp này.

Chất lượng:screening defaults38/40,flash-mm35/40;không đủ để kết luận quan hệ nhân quả hoặc ý nghĩa thống kê. Best runtime là tiêu chí khác với best robot success. Giữ defaults làm mốc chất lượng và báo tradeoff rõ ràng.

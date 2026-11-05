# Giới hạn và mentor review

## Observed facts

Audit CPU mới khớp6log benchmark, hai tổng kernel/gap từ SQLite. Model SHA256 khớp. Đã chạy lại phase-boundary analyzer trên symlink tới SQLite cũ, kết quả exit0; kiểm tra count/size K/V copies và kernel coverage của sự kiện lưu. Không chỉnh source/instrumentation, không rebuild, không chạy GPU workload mới.

`nsys status --environment`: process-tree CPU profiling OK; system-wide Fail; paranoid2; LBR unavailable. Đây là check CPU, không chứng minh GPU hardware counters dùng được. Installed nsys help có CUDA graph node tracing; node tracing tăng overhead. Không thu counters/SM utilization/occupancy/bandwidth, không khẳng định compute-bound hoặc memory-bound chỉ từ tên kernel.

Diagnostic của cả hai kernel traces: `Not all CUDA events might have been collected.`; `CUDA hardware tracing is not supported on this system. A legacy (software instrumented) trace was collected instead.`; Unified Memory không trace được vì peer-to-peer không hỗ trợ. Không có diagnostic chứa lost/drop không đủ chứng minh complete. Mọi tỷ lệ/tổng/gap là conditional trên events đã ghi nhận; thiếu activity có thể phóng đại gaps hoặc đổi tỷ lệ nhóm.

[NVIDIA CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) xác nhận hỗ trợ Nsight Systems CLI/CUPTI và giới hạn NVML. Trang đọc ngày05/11 hiện là13.5; máy dùng13.3. Không coi trang docs mới là phép xác minh tính năng chính xác mọi phiên bản. Nsys log thực tế là bằng chứng quan trọng hơn cho trace cụ thể.

## Tests và blockers

`ctest -N` liệt kê foldquant_cuda_op, foldquant_cuda_op_ref, fused_cuda_ops. `build-cuda/Testing/Temporary/LastTestsFailed.log` mtime04/11 10:05 +07 vẫn ghi11:foldquant_cuda_op và13:fused_cuda_ops. LastTest.log mtime05/11 10:52 không có nội dung match hai tên trong lần đọc. CTest log là lịch sử; chưa xác minh fail còn tái hiện, cũng chưa có bằng chứng đã sửa. Không báo test pass.

Không chạy lại vì GPU đang được server dùng. Không tự kill server. Chưa thu fresh benchmark/trace/tests trên GPU rảnh; chưa xác minh Windows build; chưa rebuild chứng minh binary/source khớp. Không có pure connector/action-expert-only timing, exact semantic attention time hay peak process resident VRAM.

## Hypotheses và recommendations

Vision softmax giảm cùng chuyển sang flash path gợi ý giảm materialization/traffic hữu ích; chưa có hardware counters chứng minh bandwidth bottleneck. Gap có thể liên quan host copies/setup/sync hoặc instrumentation/scheduling; chưa correlation đầy đủ từng gap→CPU event để kết luận CPU bottleneck. Cần GPU rảnh và trace mới xử lý/công bố completeness warning; dùng source/NVTX correlation nếu cần exact module attribution, đo instrumentation overhead.

## Mentor review

| Yêu cầu mentor | Trạng thái |
|---|---|
|Kiến trúc paper + source, shapes/denoise|Có; cấu hình checkpoint được phân biệt paper|
|Phase latency và giới hạn|Có dữ liệu cũ được tính lại, timer/synchronization/padding đã giải thích|
|Default vs optimized runtime|Có3run/cấu hình, giữ10steps; chưa chứng minh upstream original hoặc best quality|
|Nsys GEMM/attention/other|Có phân nhóm tên trên trace lưu; exact attention N/A; completeness có cảnh báo|
|Khoảng trống giữa kernels|Có interval union; không chứng minh toàn GPU idle/nguyên nhân CPU|
|WSL2 limitations|Có log cụ thể và docs chính thức|
|Fresh reproducibility, tests|Chưa: GPU bận; test cũ fail chưa rerun|

**Chưa tuyên bố Task5 hoàn tất vô điều kiện.** Phần phân tích và báo cáo có thể review; phần xác nhận đo mới, provenance original/best và trace completeness còn mở. Không che phần thiếu bằng số liệu tự tạo.

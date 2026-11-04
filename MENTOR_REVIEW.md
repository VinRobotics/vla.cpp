# Các điểm nhờ mentor review

Cập nhật ngày 04/11/2026 (UTC+07). Tài liệu chính: [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md), [bảng phần3](PART3_OPTIMIZATION.md).

## Các vấn đề thực nghiệm còn mở

| Vấn đề | Bằng chứng / giới hạn | Câu hỏi review |
|---|---|---|
| Hai CUDA tests fail | ctest-build-cuda.log: foldquant_cuda_op, fused_cuda_ops; 13/15 pass | Có cần sửa/đổi baseline trước khi dùng SR để kết luận không? |
| Fused/unfused lệch | max abs 0,000545874 trên harness cố định; chưa xác định nguyên nhân | Mức kiểm tra numerics nào cần thêm? |
| VRAM trên WSL | NVML process N/A; trace ledger peak không bao hết driver/context | Có chấp nhận báo peak traced allocation cùng giới hạn không? |
| Noise | Seed 42 cho môi trường, server random RNG | Có cần repeated seeds hoặc protocol fixed noise riêng cho final? |
| Batch latency | Control khác nhau: 112,0 / 113,9 / 104,0 ms theo batch; không gộp | Có cần final rerun trong điều kiện nguồn/nhiệt chặt hơn? |
| Synthetic output | Input 6 token, 50×7 chiều action thực; benchmark 16 token | Nên thêm calibration/real observation output checks nào? |
| LIBERO recovery | Task 9 official chạy lại 20 episode; toàn bộ phần dở bị loại | Policy recovery này có phù hợp không? |
| Dependencies | Bảy metadata conflicts; client import/render/rollout thành công | Có yêu cầu môi trường lock sạch để tái lập không? |
| Pruning | Tail/spaced; cả 3 bản smoke đạt 0/2; chưa CKA/finetune | Chọn calibration, bảo toàn parity SA/CA và coupling VLM/expert thế nào? |
| Profile | GPU active khác wall; denoise gồm crossKV; vision gồm connector | Cần thêm instrumentation để tách action expert/connector không? |

## Dữ liệu và trạng thái đánh giá

Kết quả và script tái lập được lưu trong nhánh này. Screening các cấu hình tối ưu đang tiếp tục; kết quả hiện tại là tạm thời, chưa dùng để chọn cấu hình đánh giá chính thức.

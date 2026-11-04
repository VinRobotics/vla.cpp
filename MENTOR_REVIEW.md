# Các điểm nhờ mentor review

Snapshot 04/11/2026 (UTC+07). Tài liệu chính: [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md), [bảng phần3](PART3_OPTIMIZATION.md).

## Các vấn đề thực nghiệm còn mở

| Vấn đề | Bằng chứng / giới hạn | Câu hỏi review |
|---|---|---|
| Hai CUDA tests fail | ctest-build-cuda.log: foldquant_cuda_op, fused_cuda_ops; 13/15pass | Có cần sửa/đổi baseline trước khi dùng SR để kết luận không? |
| Fused/unfused lệch | maxabs0,000545874 trên harness cố định; chưa xác định nguyên nhân | Mức kiểm tra numerics nào cần thêm? |
| VRAM trên WSL | NVML process N/A; trace ledger peak không bao hết driver/context | Có chấp nhận báo peak traced allocation cùng giới hạn không? |
| Noise | Seed42 cho môi trường, server random RNG | Có cần repeated seeds hoặc protocol fixed noise riêng cho final? |
| Batch latency | Control khác nhau112,0 /113,9 /104,0ms theo batch; không gộp | Có cần final rerun trong điều kiện nguồn/nhiệt chặt hơn? |
| Synthetic output | Input6token,50×7real action; benchmark16token | Nên thêm calibration/real observation output checks nào? |
| LIBERO recovery | Task9 official rerun20episode; toàn bộ phần dở bị loại | Policy recovery này có phù hợp không? |
| Dependencies | Bảy metadata conflicts; client import/render/rollout thành công | Có yêu cầu môi trường lock sạch để tái lập không? |
| Pruning | Tail/spaced; cả3smoke0/2; chưa CKA/finetune | Chọn calibration, bảo toàn parity SA/CA và coupling VLM/expert thế nào? |
| Profile | GPU active khác wall; denoise gồm crossKV; vision gồm connector | Cần thêm instrumentation để tách action expert/connector không? |

## Rà soát Markdown

Đã đọc các tài liệu thực nghiệm và kiểm tra link file cục bộ trên56Markdown của repo/tài liệu mới. Đây không phải xác minh mọi thông số benchmark upstream trên mọi phần cứng. Kết quả máy đọc: research-notes/review-20261104/markdown-audit.json.

- Đã thay report tổng bị cũ bằng snapshot hiện trạng; bản cũ giữ ở REPORT_HISTORY.md và gắn nhãn archive.
- Đã sửa tài liệu kiến trúc đang ghi chưa có profile/pruning, bổ sung số đo và giới hạn attribution.
- NOTES_SMOLVLA.md là nhật ký lịch sử: các đoạn chưa chạy/đang cài cũ không phải hiện trạng.
- PART3_OPTIMIZATION.md phân biệt batch, input benchmark và input output-check, cũng như allocation với resident VRAM.
- docs/backend/cuda.md mô tả fast path bit-exact; kết quả tại máy này chưa xác nhận được do test fail và fused/unfused khác. Không tự sửa claim upstream thành kết luận chung chỉ từ một máy.
- docs/EVAL.md trỏ tới thư mục SimplerEnv chưa cài tại máy; eval/README.md trỏ eval/reports chưa có. Đây là đường dẫn tùy môi trường, không thuộc kết quả LIBERO hiện tại.
- Chưa kiểm tra tự động mọi anchor/URL ngoài mạng hoặc render Mermaid bằng trình duyệt.

## Những gì đưa lên GitHub

Scripts và dữ liệu nhỏ (JSON/CSV/log cần thiết/output checks), report và bằng chứng lỗi. Không đưa GGUF, wheel, venv, video hoặc nsys/sqlite lớn lên Git. Đường dẫn tuyệt đối trong command/protocol là provenance máy đo; khi tái lập phải thay root/model/config theo máy mới.

Screening đang chạy: dữ liệu commit là snapshot, chưa phải kết quả cuối. Không chạy official biến thể cho đến khi tổng hợp screening và chọn ứng viên.

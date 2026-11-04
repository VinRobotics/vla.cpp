# Các điểm nhờ mentor review

Cập nhật ngày 05/11/2026 (UTC+07). Tài liệu chính: [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md), [bảng phần3](PART3_OPTIMIZATION.md).

## Các vấn đề thực nghiệm còn mở

Đợt báo cáo chốt theo dữ liệu hiện có; không chạy thêm official cho biến thể hoặc thí nghiệm mới. Hai lỗi CUDA tạm hoãn xử lý, vẫn ghi nhận 13/15 tests pass. Các câu hỏi dưới đây phục vụ nhận xét kết quả và định hướng nghiên cứu sau đợt này.

| Vấn đề | Bằng chứng / giới hạn | Câu hỏi review |
|---|---|---|
| Hai CUDA tests fail | ctest-build-cuda.log: foldquant_cuda_op, fused_cuda_ops; 13/15 pass; tạm hoãn xử lý | Giới hạn này ảnh hưởng thế nào đến phạm vi kết luận từ kết quả đã đo? |
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

Kết quả và script tái lập được lưu trong nhánh này. Screening đã hoàn tất 120/120 episode: flash-mm 35/40, Q8-lm 36/40, Q8-vision 36/40, so với defaults 38/40. Official biến thể và repeated-seed chưa chạy; chưa kết luận giữ SR. Kết quả đã đối chiếu CSV bằng `research-notes/smolvla-followup-20261104/audit_screening.py`.

## Rà soát phần thiếu và tiêu chí hoàn tất

“Đủ sáu phần trong báo cáo” chỉ nói về phạm vi trình bày. Đối chiếu yêu cầu mentor ban đầu, phần 1, 2, 4, 5 và 6 còn các tiêu chí thực nghiệm chưa đạt đầy đủ. Phần 3 có đủ bảng đo yêu cầu, nhưng chưa chứng minh cấu hình tối ưu giữ chất lượng.

| Phần | Phần đã bổ sung hoặc xác minh trong lượt rà soát | Điều kiện còn thiếu để đạt trọn yêu cầu |
|---|---|---|
| 1. Cài đặt | Thống nhất trạng thái 13/15 giữa checklist và báo cáo | Sửa hai test CUDA, chạy lại theo hợp đồng hiện tại; chưa thực hiện vì đang tạm hoãn |
| 2. Baseline | Giữ rõ allocation và resident VRAM; công bố điều kiện đo và giới hạn telemetry | Đo process resident memory trên môi trường hỗ trợ nếu yêu cầu đúng chỉ số này |
| 3. Tối ưu | Đã có bảng bốn chỉ tiêu, control riêng và tính speedup cùng batch | Kiểm tra chất lượng cho cấu hình định chọn; không suy SR từ synthetic delta |
| 4. LIBERO | Audit cả official và screening; thêm bảng thành công từng task | Official cho cấu hình được chọn chưa có; không chạy thêm trong đợt chốt này |
| 5. Profile | Phân biệt module GPU-active, phase wall và default path | Connector riêng, attention riêng và phân tích nguyên nhân gap chưa hoàn tất |
| 6. Pruning | Có ba depth ablation và kết quả âm; phân biệt rõ với CLP | Nếu yêu cầu tái hiện phương pháp paper: cần activation calibration, CKA, mapping hợp lệ và quy trình finetuning/đánh giá tương ứng |

## Kiểm tra dữ liệu có thể chạy lại

Từ thư mục gốc repo, không cần GPU hoặc simulator:

```bash
python3 research-notes/smolvla-followup-20261104/audit_submission.py
python3 research-notes/smolvla-followup-20261104/complete_calculations.py
```

Lệnh đầu đối chiếu 50 CSV của 360 episode với manifest cuối: task/episode ID, số bước, success, aborted và SR; lưu hash từng CSV trong [submission-audit.json](research-notes/smolvla-followup-20261104/submission-audit.json), sinh [bảng từng task](research-notes/smolvla-followup-20261104/TASK_RESULTS.md). Chỉ dùng task 9 official từ đường dẫn recovery đã được chọn, không cộng lượt dở. Lệnh thứ hai sinh lại phụ lục tính toán. Cần có các CSV cục bộ theo đường dẫn manifest; lệnh không tự tải model, tạo môi trường hoặc sinh episode thiếu.

Audit này không xác nhận video, đúng nội dung instruction, kernel correctness hoặc môi trường portable. Cùng seed môi trường và episode ID chưa bảo đảm cùng noise; bảng từng task chỉ mô tả kết quả quan sát, chưa xác định nguyên nhân thất bại hay chứng minh non-inferiority.

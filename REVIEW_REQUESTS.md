# Các phần xin review

Cập nhật ngày 05/11/2026 (UTC+07). Tài liệu chính: [REPORT_SMOLVLA.md](REPORT_SMOLVLA.md), [bảng phần3](PART3_OPTIMIZATION.md).

## Các vấn đề thực nghiệm còn mở

Nhờ người review góp ý về phương pháp đo, cách diễn giải kết quả và hướng nghiên cứu trong các phần dưới đây. Hai lỗi CUDA vẫn được công bố: 13/15 tests pass. Official flash-mm đã được khởi chạy theo yêu cầu bổ sung; chưa đưa SR cuối vào tài liệu này.

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
| Pruning | Screening vision-tail 0/40, vision-spaced 0/40, coupled-tail 23/40; có tiện ích CKA nhưng chưa calibration thực/finetune | Chọn calibration, bảo toàn parity SA/CA và coupling VLM/expert thế nào? |
| Profile | GPU active khác wall; denoise gồm crossKV; vision gồm connector | Cần thêm instrumentation để tách action expert/connector không? |

## Dữ liệu và trạng thái đánh giá

Kết quả và script tái lập được lưu trong nhánh này. Screening đã hoàn tất 120/120 episode: flash-mm 35/40, Q8-lm 36/40, Q8-vision 36/40, so với defaults 38/40. Official flash-mm đã khởi chạy; official hai bản Q8 và repeated-seed chưa có. Chưa kết luận giữ SR. Kết quả đã đối chiếu CSV bằng `research-notes/smolvla-followup-20261104/audit_screening.py`.

## Rà soát phần thiếu và tiêu chí hoàn tất

“Đủ sáu phần trong báo cáo” chỉ nói về phạm vi trình bày. Đối chiếu yêu cầu mentor ban đầu, phần 1, 2, 4, 5 và 6 còn các tiêu chí thực nghiệm chưa đạt đầy đủ. Phần 3 có đủ bảng đo yêu cầu, nhưng chưa chứng minh cấu hình tối ưu giữ chất lượng.

| Phần | Phần đã bổ sung hoặc xác minh trong lượt rà soát | Điều kiện còn thiếu để đạt trọn yêu cầu |
|---|---|---|
| 1. Cài đặt | Thống nhất trạng thái 13/15 giữa checklist và báo cáo | Sửa hai test CUDA, chạy lại theo hợp đồng hiện tại; chưa thực hiện vì đang tạm hoãn |
| 2. Baseline | Giữ rõ allocation và resident VRAM; công bố điều kiện đo và giới hạn telemetry | Đo process resident memory trên môi trường hỗ trợ nếu yêu cầu đúng chỉ số này |
| 3. Tối ưu | Đã có bảng bốn chỉ tiêu, control riêng và tính speedup cùng batch | Kiểm tra chất lượng cho cấu hình định chọn; không suy SR từ synthetic delta |
| 4. LIBERO | Audit cả official và screening; thêm bảng thành công từng task | Defaults đã có official 177/200; flash-mm đang được đánh giá bổ sung, chưa có kết quả cuối trong tài liệu này |
| 5. Profile | Phân biệt module GPU-active, phase wall và default path | Connector riêng, attention riêng và phân tích nguyên nhân gap chưa hoàn tất |
| 6. Pruning | Có ba depth ablation và kết quả âm; phân biệt rõ với CLP | Nếu yêu cầu tái hiện phương pháp paper: cần activation calibration, CKA, mapping hợp lệ và quy trình finetuning/đánh giá tương ứng |

## Task 5 — kết quả audit và điều kiện kết luận

Audit CPU ngày 05/11 đã parse khớp sáu raw latency log, tính lại tổng kernel/gap từ hai SQLite trace và kiểm tra phase-boundary attribution trên các kernel đã lưu. Trung vị ba run means là defaults 114,1 ms và flash-mm 92,4 ms (giảm 19,02%); không dùng riêng minimum run mean 113,9/91,5 ms làm kết quả đại diện mới.

Điểm cần review thêm: cả hai trace báo `Not all CUDA events might have been collected.`; không thể chứng minh completeness chỉ vì không có chuỗi lost/drop. Trace dùng legacy software instrumentation; Unified Memory không trace được. Gap 10,54%/9,13% chỉ thuộc recorded target-process activity, không whole-GPU idle hoặc CPU bottleneck đã xác nhận. Timer wall có backend synchronization; PHASE đổi graph/padding nên không chia trực tiếp baseline mặc định thành module times.

GPU đang có server dùng tại lúc audit nên chưa đo lại hoặc rerun hai CUDA tests. File LastTestsFailed lịch sử vẫn liệt kê foldquant_cuda_op và fused_cuda_ops; không đồng nghĩa đã tái hiện fail hôm nay. Chưa xác minh source/binary provenance bằng rebuild, defaults được chọn theo SR quan sát, chưa có official so sánh cho biến thể, exact connector/action-expert-only/attention timing hoặc resident process VRAM.

[Review Task 5 chi tiết](research-notes/task5-profiling/limitations.md) và [script audit CPU](research-notes/task5-profiling/audit_evidence.py). Đây là báo cáo có thể review trên bằng chứng hiện có, chưa xác nhận hoàn tất toàn bộ yêu cầu thực nghiệm.

## Kiểm tra dữ liệu có thể chạy lại

Từ thư mục gốc repo, không cần GPU hoặc simulator:

```bash
python3 research-notes/smolvla-followup-20261104/audit_submission.py
python3 research-notes/smolvla-followup-20261104/complete_calculations.py
```

Lệnh đầu đối chiếu 50 CSV của 360 episode với manifest cuối: task/episode ID, số bước, success, aborted và SR; lưu hash từng CSV trong [submission-audit.json](research-notes/smolvla-followup-20261104/submission-audit.json), sinh [bảng từng task](research-notes/smolvla-followup-20261104/TASK_RESULTS.md). Chỉ dùng task 9 official từ đường dẫn recovery đã được chọn, không cộng lượt dở. Lệnh thứ hai sinh lại phụ lục tính toán. Cần có các CSV cục bộ theo đường dẫn manifest; lệnh không tự tải model, tạo môi trường hoặc sinh episode thiếu.

Audit này không xác nhận video, đúng nội dung instruction, kernel correctness hoặc môi trường portable. Cùng seed môi trường và episode ID chưa bảo đảm cùng noise; bảng từng task chỉ mô tả kết quả quan sát, chưa xác định nguyên nhân thất bại hay chứng minh non-inferiority.

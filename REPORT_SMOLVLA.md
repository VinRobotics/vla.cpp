# Báo cáo thực nghiệm SmolVLA

Snapshot để mentor review, ngày 04/11/2026 (UTC+07). Repo gốc tại commit `4a542ed3bfb8f330388aceb8926294a951156172`; các kết quả là số đo cục bộ trên RTX 3060 Laptop 6 GiB qua WSL2. Snapshot screening có thể cũ hơn tiến trình đang chạy.

## Trạng thái sáu phần

| Phần | Kết quả hiện có | Chưa hoàn tất |
|---|---|---|
| 1. Cài đặt | CUDA arch86; CLI, server, binding, predict_check đã chạy | Tests13/15; hai lỗi bit-exact còn mở |
| 2. Baseline | Có p50/p90, traced allocation và điều kiện đo | Không đo được toàn bộ resident VRAM riêng process trên WSL |
| 3. Không train | 10 runtime configs và Q8/Q5/Q4 có/không vision; đủ4chỉ tiêu | Chưa kết luận giữ chất lượng điều khiển |
| 4. LIBERO | Baseline screening38/40; official177/200 | Screening biến thể đang chạy; official biến thể chưa chạy |
| 5. Profile | Kiến trúc, phase, nsys kernel/module traces | Chưa chọn bản tốt nhất theo SR; chưa tách connector khỏi vision |
| 6. Pruning | 3depth ablations; module latency và smoke | Smoke cả3bản0/2; chưa implement CKA/finetuning |

## Protocol và kết quả baseline

- Checkpoint gốc: `vrfai/smolvla-libero-gguf`; hash/metadata lưu trong research-notes.
- vla-bench:2views512×512,16tokens,10solver steps;3process/config,3warmup+20reps. Chọn lượt có mean thấp nhất và p50/p90 cùng lượt; bộ nhớ lấy peak lớn nhất.
- LIBERO: libero_object, task0–9, n_action_steps1, seed môi trường42; server noise mặc định ngẫu nhiên. Seed42 không đảm bảo output bitwise hoặc cùng noise giữa các cấu hình.
- Defaults screening38/40=95%; official177/200=88,5%. Không so screening của biến thể trực tiếp với official baseline để kết luận giảm SR.
- Task9 official từng timeout. Chạy lại toàn bộ20episode task9, loại toàn bộ phần dở; giữ180episode task0–8. Lần recovery đầu lỗi config trước episode; retry thành công. Kết quả cuối đã đối chiếu10CSV đủ200episode, không abort.

| Task | Thành công /20 |
|---|---:|
| 0 | 18 |
| 1 | 20 |
| 2 | 17 |
| 3 | 17 |
| 4 | 19 |
| 5 | 13 |
| 6 | 16 |
| 7 | 20 |
| 8 | 20 |
| 9 | 17 |

## Tối ưu không train

[Bảng phần3](PART3_OPTIMIZATION.md) có kích thước file, mean/p50/p90, peak allocation và max abs/RMSE real7 cho từng cấu hình.

Ba batch latency phải đọc riêng: baseline ban đầu mean112,0ms; runtime-pair defaults113,9ms/flash+mm91,5ms; runtime bổ sung defaults104,0ms/flash+mm85,5ms. Đây là các lượt ở thời điểm/điều kiện khác nhau, không phải một control chung. Không chọn số thấp nhất xuyên batch để tính speedup. Quantization có control riêng mean113,7ms.

Sai lệch output dùng input harness6token và noise cố định; benchmark dùng16token. Metric lấy50×7 action thực, loại padding. Không suy ra SR từ sai số synthetic.

## Screening biến thể — snapshot

| Cấu hình | Episode đã chốt | Thành công | Trạng thái |
|---|---:|---:|---|
| flash-mm | 40/40 | 35 | complete |
| q8-lm | 16/40 | 15 | running task 4 |
| q8-vision | 0/40 | 0 | chưa bắt đầu |

Đếm theo task đã hoàn tất. Chạy tuần tự flash+mm → Q8 không vision → Q8 có vision, tổng120episode. Raw progress: `research-notes/smolvla-followup-20261104/screen40-variants/`. Snapshot không phải kết quả cuối nếu batch còn running.

## Profile và pruning

[Kiến trúc và profile](research-notes/smolvla-followup-20261104/ARCHITECTURE_AND_PROFILE.md) giải thích phạm vi phase, module GPU-active và giới hạn nsys/WSL. [Ghi chú paper](PAPER_READING_SMOLVLA_CLP.md) nêu hướng CKA. Pruning hiện chỉ bỏ layer tail/spaced, không gọi là tái hiện CLP.

## Vấn đề cần mentor review

1. Hai CUDA tests fail và fused/unfused maxabs0,000545874: chưa tìm ra nguyên nhân; giữ defaults theo quyết định baseline của người dùng.
2. Peak allocation1912,47MiB chưa bao hết context/driver. NVML process memory trả N/A; không gọi là toàn bộ process resident VRAM.
3. Bảy xung đột dependency metadata còn tồn tại dù client chạy được. Môi trường chưa có lock portable sạch.
4. Random server noise và screening40 nhỏ làm khác biệt SR có độ bất định. Chưa có interval hoặc repeated-seed evaluation.
5. Batch latency có khác biệt điều kiện nhiệt/tải; đề nghị thống nhất protocol final trước kết luận triển khai.
6. Chọn calibration riêng và kiểm tra ràng buộc layer VLM/expert, parity SA/CA trước CKA; không dùng episode đánh giá để chọn layer.

## Tài liệu và bằng chứng

- [Nhật ký](NOTES_SMOLVLA.md) là lịch sử, không dùng các dòng trạng thái cũ làm hiện trạng.
- [Report lịch sử](research-notes/review-20261104/REPORT_HISTORY.md) giữ nguyên snapshot cũ, có nội dung đã hết hiệu lực.
- [Review Markdown](MENTOR_REVIEW.md) liệt kê các vấn đề tài liệu và phương pháp.
- Dữ liệu nhỏ/scripts được commit; GGUF, video, trace nsys/sqlite, wheel và môi trường cài đặt không đưa lên Git.

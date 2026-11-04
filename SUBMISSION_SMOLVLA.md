# Báo cáo tiến độ SmolVLA — bản gửi review

Tổng hợp đủ 6 task; cập nhật kết quả screening cuối ngày 05/11/2026. Số đo thật, dự toán và tiêu chí chưa đạt được tách rõ.

Phạm vi chốt báo cáo: sử dụng kết quả đã đo, không chạy thêm official cho biến thể hoặc thí nghiệm mới trong đợt này. Hai lỗi CUDA được giữ nguyên và công bố với trạng thái 13/15 tests pass.

## Kết quả đã hoàn tất

- CUDA build arch86; các giao diện CLI/server/binding/predict_check hoạt động. Tests 13/15 pass, hai lỗi còn mở.
- Baseline LIBERO chính thức: **177/200 = 88,5%**. Task9 đã chạy lại đầy đủ20episode sau timeout, loại phần dở; kết quả cuối kiểm tra đủ200episode không abort.
- Phần3:10cấu hình runtime,6GGUF Q8/Q5/Q4 có/không vision; có bảng kích thước, latency, peak allocation và sai lệch action.
- Flash+mm screening: **35/40 = 87,5%**. Baseline screening đối chứng:38/40. Chưa kết luận khác biệt có ý nghĩa thống kê.

## Screening đã hoàn tất

| Cấu hình | Episode đã chốt | Thành công | Trạng thái |
|---|---:|---:|---|
| flash-mm | 40/40 | 35 | complete: 87,5% |
| q8-lm | 40/40 | 36 | complete: 90% |
| q8-vision | 40/40 | 36 | complete: 90% |

Batch đã hoàn tất 120/120 episode, cả ba cấu hình exit 0. Official200 cho biến thể chưa chạy; screening chưa đủ để kết luận giữ chất lượng. Q8-vision giảm allocation còn 925,88 MiB, thấp hơn Q8-lm 1020,70 MiB; cả hai đạt 36/40 nhưng chưa chứng minh tương đương. Flash-mm ưu tiên latency; defaults vẫn là bản có SR official đã đo.

## Tài liệu chi tiết

- [Phụ lục tính toán đủ 6 task](CALCULATIONS_SMOLVLA.md) — speedup, allocation, Wilson SR, dự toán official và pruning.
- [Báo cáo tổng thể](REPORT_SMOLVLA.md)
- [Bảng tối ưu không train](PART3_OPTIMIZATION.md)
- [Kiến trúc và profiling](research-notes/smolvla-followup-20261104/ARCHITECTURE_AND_PROFILE.md)
- [Các câu hỏi cần góp ý](MENTOR_REVIEW.md)
- [Ghi chú hai paper](PAPER_READING_SMOLVLA_CLP.md)

## Giới hạn cần đọc cùng kết quả

VRAM là peak traced allocation, chưa phải toàn bộ resident VRAM riêng process trên WSL. Latency chỉ so với control cùng batch. Output-check dùng input synthetic khác benchmark và không thay được SR. Server noise chưa cố định. Pruning hiện là depth ablation, chưa tái hiện CLP; cả3bản smoke0/2. Các chi tiết test fail, môi trường và recovery được công bố trong báo cáo tổng thể.

Dự toán, chưa đo: flash-mm175/200, Q8-lm180/200 và Q8-vision180/200 bằng200×SRscreening; không dùng làm kết quả official. Kết quả âm pruning vẫn được báo đầy đủ: ba bản giảm depth đều smoke0/2. Mục tiêu triển khai giữSR chưa đạt; defaults177/200 vẫn là mốc chính thức.

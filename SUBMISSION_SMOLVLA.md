# Báo cáo tiến độ SmolVLA — bản gửi review

Cập nhật 04/11/2026 23:41 UTC+07. Đây là báo cáo giữa kỳ; các giá trị còn thiếu được ghi rõ, không phải kết quả cuối của toàn bộ bài.

## Kết quả đã hoàn tất

- CUDA build arch86; các giao diện CLI/server/binding/predict_check hoạt động. Tests 13/15 pass, hai lỗi còn mở.
- Baseline LIBERO chính thức: **177/200 = 88,5%**. Task9 đã chạy lại đầy đủ20episode sau timeout, loại phần dở; kết quả cuối kiểm tra đủ200episode không abort.
- Phần3:10cấu hình runtime,6GGUF Q8/Q5/Q4 có/không vision; có bảng kích thước, latency, peak allocation và sai lệch action.
- Flash+mm screening: **35/40 = 87,5%**. Baseline screening đối chứng:38/40. Chưa kết luận khác biệt có ý nghĩa thống kê.

## Đánh giá đang tiếp tục

| Cấu hình | Episode đã chốt | Thành công | Trạng thái |
|---|---:|---:|---|
| flash-mm | 40/40 | 35 | complete |
| q8-lm | 36/40 | 34 | running task 9 |
| q8-vision | 0/40 | — | chưa bắt đầu; chưa có số đo |

Kết quả Q8 chưa đầy đủ, không trình bày tỷ lệ tạm thời như SR của cả40episode. Official200 cho biến thể chưa chạy. Không chọn cấu hình cuối cùng khi chưa tổng hợp đánh giá.

## Tài liệu chi tiết

- [Báo cáo tổng thể](REPORT_SMOLVLA.md)
- [Bảng tối ưu không train](PART3_OPTIMIZATION.md)
- [Kiến trúc và profiling](research-notes/smolvla-followup-20261104/ARCHITECTURE_AND_PROFILE.md)
- [Các câu hỏi cần góp ý](MENTOR_REVIEW.md)
- [Ghi chú hai paper](PAPER_READING_SMOLVLA_CLP.md)

## Giới hạn cần đọc cùng kết quả

VRAM là peak traced allocation, chưa phải toàn bộ resident VRAM riêng process trên WSL. Latency chỉ so với control cùng batch. Output-check dùng input synthetic khác benchmark và không thay được SR. Server noise chưa cố định. Pruning hiện là depth ablation, chưa tái hiện CLP; cả3bản smoke0/2. Các chi tiết test fail, môi trường và recovery được công bố trong báo cáo tổng thể.

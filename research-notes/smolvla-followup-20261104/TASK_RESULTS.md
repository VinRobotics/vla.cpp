# Đối chiếu kết quả LIBERO theo task

Sinh từ CSV bằng `python3 research-notes/smolvla-followup-20261104/audit_submission.py` tại gốc repo. Kiểm tra 360 episode: 200 official defaults, 40 screening defaults và 120 screening biến thể. Kiểm tra task/episode ID, số bước, success, aborted và SR so với manifest; lưu SHA256 từng CSV trong [submission-audit.json](submission-audit.json). Không chạy inference mới hoặc kiểm tra lại video.

| Task | Defaults /4 | Flash-mm /4 | Q8-lm /4 | Q8-vision /4 | Official defaults /20 |
|---|---:|---:|---:|---:|---:|
| 0 | 4 | 4 | 4 | 4 | 18 |
| 1 | 4 | 4 | 4 | 4 | 20 |
| 2 | 4 | 3 | 3 | 4 | 17 |
| 3 | 3 | 4 | 4 | 4 | 17 |
| 4 | 4 | 4 | 4 | 3 | 19 |
| 5 | 3 | 1 | 3 | 3 | 13 |
| 6 | 4 | 4 | 4 | 4 | 16 |
| 7 | 4 | 4 | 4 | 3 | 20 |
| 8 | 4 | 4 | 4 | 3 | 20 |
| 9 | 4 | 3 | 2 | 4 | 17 |
| Tổng | 38 | 35 | 36 | 36 | 177 |

Mọi CSV được chọn có aborted=0. Official task 9 lấy đường dẫn recovery trong result.json; không cộng episode của lượt bị gián đoạn. Hash chỉ xác nhận nội dung artifact tại lúc audit, không chứng minh môi trường hoặc noise có thể tái hiện từng bit.

Bảng giúp xác định task cần xem lại video. Chênh lệch thành công không xác định nguyên nhân do quantization, attention hay noise. Cùng episode ID và seed môi trường chưa tạo thành cặp đối chứng cùng noise; không dùng bảng này để kiểm định ghép cặp hoặc chọn cấu hình trên tập đánh giá.

Chỉ so các cột screening với nhau khi mô tả chênh lệch quan sát. Cột official có số episode khác và là một lượt chạy riêng; không lấy 177/200 làm control cho screening biến thể.

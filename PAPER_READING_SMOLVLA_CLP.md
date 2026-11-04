# Ghi chú đọc hai paper mentor cung cấp

## SmolVLA — kiến trúc và inference

Nguồn: https://arxiv.org/html/2506.01844v1 — mục3.1,3.3 và4.7.

Paper kết hợp VLM nhỏ với action expert flow matching. Các hướng giảm chi phí gồm giảm visual token, dùng đặc trưng từ các layer VLM sớm và attention xen kẽ. Asynchronous execution tách dự đoán khỏi thực thi action. Không đồng nhất cải thiện điều khiển bất đồng bộ với giảm latency một forward.

## CLP — chọn layer bằng dữ liệu

Nguồn: https://arxiv.org/html/2606.20246v1 — mục3,4 và5.

CLP dùng activation calibration để tính CKA giữa layer liên tiếp. Gom vùng tương tự cao, giữ layer đầu làm anchor, chọn các layer dư thừa theo ngân sách rồi loại bỏ. Bước chọn/cắt không cần training; kết quả chất lượng phải đọc cùng downstream finetuning. CKA cao chưa đảm bảo bỏ layer an toàn.

## Áp dụng vào repo này — đề xuất thực nghiệm của mình

Các bản tail/spaced hiện có là đối chứng depth ablation, chưa phải tái hiện CLP. Bước tiếp theo cần thu activation trên dữ liệu calibration tách khỏi episode đánh giá, chọn layer theo điểm số và xuất GGUF. Loader hiện có ràng buộc ghép layer VLM/expert và parity SA/CA; phải kiểm tra mapping trước khi áp dụng lựa chọn độc lập. Có thể cần sửa loader nếu muốn đúng phạm vi pruning của phương pháp.

Cần ghi rõ bản không finetune là adaptation training-free của ý tưởng, không gán kết quả paper cho checkpoint này. So sánh với tail/spaced cùng số layer bị bỏ, chạy smoke rồi screening40. Đánh giá latency module bằng cả benchmark và nsys; official200 chỉ dành cấu hình được chọn. Không sử dụng dữ liệu episode đánh giá làm calibration.

Paper mô tả kiến trúc tổng quát; bảng số layer/hidden size phải lấy metadata checkpoint đang dùng. Kết quả đang chạy trên GPU không bị thay đổi bởi việc đọc tài liệu này.

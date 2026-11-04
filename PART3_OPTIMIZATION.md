# Phần 3 — Tối ưu SmolVLA không cần train

## Phạm vi và protocol

Không thay GGUF gốc. Runtime flags dùng cùng checkpoint; quantization xuất sáu file riêng, không train, không pruning.
Latency: vla-bench, 2 ảnh512×512,16token,3warmup+20reps,3process/config. Chọn cả lượt có mean thấp nhất; p50/p90 thuộc cùng lượt. Bộ nhớ lấy peak lớn nhất trong3process/config, đo bằng nsys riêng với cùng workload.
Output: vla_predict_check,2ảnh512,6token theo harness, noise cố định; so với defaults. Max abs và RMSE trên50×7 action thực, loại padding32chiều. Đây là input khác benchmark; không suy ra SR từ độ lệch synthetic.

## Runtime flags

| Cấu hình | File MiB | Mean / p50 / p90 ms | Peak allocation MiB | Max abs real7 | RMSE real7 |
|---|---:|---:|---:|---:|---:|
| bf16-flash | 1074.18 | 111.3 / 111.2 / 112.0 | 1044.27 | 0.00078821182 | 0.00028052564 |
| bf16 | 1074.18 | 123.1 / 123.0 / 123.7 | 1134.34 | 0 | 0 |
| defaults | 1074.18 | 104.0 / 103.7 / 105.0 | 1912.47 | 0 | 0 |
| f16-flash | 1074.18 | 109.1 / 109.1 / 109.9 | 1044.27 | 0.0012158751 | 0.0002933432 |
| f16 | 1074.18 | 125.9 / 126.0 / 126.5 | 1119.39 | 0.00097492337 | 0.00026844391 |
| flash-mm | 1074.18 | 85.5 / 85.4 / 85.9 | 1040.27 | 0.0027671456 | 0.00061441981 |
| flash | 1074.18 | 91.8 / 91.8 / 92.3 | 1818.39 | 0.00078821182 | 0.00028052564 |
| mm-default | 1074.18 | 97.4 / 97.4 / 97.9 | 1134.34 | 0.0020601451 | 0.00066080622 |
| steps-3 | 1074.18 | 79.2 / 79.0 / 80.0 | 1911.83 | 0.30797094 | 0.042685641 |
| steps-5 | 1074.18 | 86.2 / 86.1 / 87.1 | 1912.01 | 0.059932411 | 0.018432219 |

Flags chính xác của mỗi cấu hình nằm trong `part3-runtime/latency-runs.json`. bf16/f16 chỉ đổi weight dtype; activation giữ mặc định F32. Không thử act-dtype BF16 cho SmolVLA vì docs giới hạn flag này cho các kiến trúc khác. steps-5/3 đổi số bước solver; các cấu hình khác giữ10bước.

## Quantization — runtime defaults

| Model | File MiB | Mean / p50 / p90 ms | Peak allocation MiB | Max abs real7 | RMSE real7 |
|---|---:|---:|---:|---:|---:|
| defaults | 1074.18 | 113.7 / 113.7 / 114.6 | 1912.47 | 0 | 0 |
| smolvla-q4_0-lm | 626.76 | 110.0 / 110.2 / 111.0 | 865.07 | 0.08243826 | 0.025425385 |
| smolvla-q4_0-vision | 510.32 | 109.7 / 109.9 / 110.4 | 729.75 | 0.17617321 | 0.050421203 |
| smolvla-q5_0-lm | 665.67 | 110.6 / 110.5 / 111.2 | 903.98 | 0.033897966 | 0.0089555682 |
| smolvla-q5_0-vision | 559.35 | 111.4 / 111.0 / 111.9 | 778.78 | 0.10021222 | 0.029256382 |
| smolvla-q8_0-lm | 782.38 | 109.5 / 109.3 / 110.3 | 1020.70 | 0.0084213912 | 0.0021052691 |
| smolvla-q8_0-vision | 706.45 | 110.2 / 110.3 / 111.0 | 925.88 | 0.041252255 | 0.011150829 |

Quantization theo scripts/quantize_gguf.py trong docs/MODELS.md. `lm` gồm225tensor (VLM và connector mm.fc1); `vision` thêm72tensor. Action expert289tensor giữ float. Chi tiết kích thước byte, SHA256 và command ở `quantization/manifest.json`.

## Điều kiện đo và giới hạn

RTX3060Laptop6GiB, WSL2; laptop cắm sạc, chế độ nguồn bình thường, không tản nhiệt ngoài theo người dùng. Đợt runtime bổ sung từ chối chạy nếu còn vla-server; không khẳng định loại hết ứng dụng Windows. Số đo quantization là batch trước, có control cùng batch; không ghép batch để suy ra chênh lệch nhỏ.
WSL báo N/A cho process resident VRAM. Cột bộ nhớ là peak traced device allocation (Device+DeviceStatic), không bao hết context/driver hoặc allocation không được ghi nhận. Không coi đây là toàn bộ peak resident VRAM.
Hai CUDA tests bit-exact còn fail; baseline giữ defaults theo lựa chọn người dùng. Fused/unfused có sai lệch đã ghi riêng. Không gán đảm bảo bit-exact trong docs thành kết quả đã xác nhận trên máy này.

## Nguồn dữ liệu và tái lập

- Runtime: part3_runtime_children.py, run_part3.sh, part3-runtime/*, audit_part3_memory.py.
- Quantization: prepare_quantized.py, quantization/manifest.json, candidate-bench/selected.json, candidate-checks/summary.json, memory-all/summary.json.
- Docs repo: docs/USAGE.md, docs/backend/cuda.md, docs/MODELS.md.
- Không kết luận SR trong phần này. Đánh giá LIBERO thuộc phần4.

## Nhận xét từ số đo
- Runtime nhanh nhất trong tập thử: steps-3, mean 79.2ms; giảm 23.85% so với control cùng batch. Cần đọc kèm độ lệch action; không xem giảm solver steps là miễn phí chất lượng.
- File runtime không đổi kích thước vì dùng cùng GGUF; resident weight dtype thay đổi trong bộ nhớ.
- Quantization giảm rõ kích thước và allocation; latency cần so với defaults của chính candidate-bench. Không suy ra tốc độ chỉ từ số bit.
- Output synthetic không chứng minh giữ khả năng điều khiển. Phần3 hoàn tất phép đo, chưa chốt cấu hình triển khai theo SR.

Telemetry bổ sung: 120 mẫu, nhiệt độ ghi nhận 53–72°C; không phủ toàn bộ batch và không đủ để kết luận có/không throttle. Raw: `part3-runtime/telemetry.jsonl`.

## Phép tính bổ sung và liên hệ 6 task

[Bảng tính tái lập](CALCULATIONS_SMOLVLA.md#task-3--mức-giảm-so-control-cùng-batch) bổ sung phần trăm giảm latency/allocation và speedup cho từng cấu hình. Flash-mm giảm mean17,79%, speedup1,216× và giảm allocation45,61% so control runtime104,0ms/1912,47MiB. Q8-lm và Q8-vision screening đều đã hoàn tất 36/40; official của biến thể chưa đo. Q8-vision giảm allocation 51,59% so với control quantization, Q8-lm giảm 46,63%; screening bằng nhau chưa chứng minh tương đương chất lượng. Xem [báo cáo đủ6task](REPORT_SMOLVLA.md) và [pruning](CALCULATIONS_SMOLVLA.md#task-6--ba-pruning-đã-implement-và-kiểm-tra).

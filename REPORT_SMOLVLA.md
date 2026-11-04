# Báo cáo thực nghiệm SmolVLA

Bản tổng hợp đủ nội dung 6 task, bổ sung tính toán theo yêu cầu người dùng ngày 04/11/2026. Các thí nghiệm chưa chạy vẫn được ghi rõ. Repo gốc tại commit `4a542ed3bfb8f330388aceb8926294a951156172`; các kết quả là số đo cục bộ trên RTX 3060 Laptop 6 GiB qua WSL2. Cập nhật ngày 05/11/2026 (UTC+07): screening ba biến thể đã hoàn tất 120/120 episode; phần dự toán không phải số đo.

## Trạng thái sáu phần

| Phần | Kết quả hiện có | Chưa hoàn tất |
|---|---|---|
| 1. Cài đặt | CUDA arch86; CLI, server, binding, predict_check đã chạy | Tests13/15; hai lỗi bit-exact còn mở |
| 2. Baseline | Có p50/p90, traced allocation và điều kiện đo | Không đo được toàn bộ resident VRAM riêng process trên WSL |
| 3. Không train | 10 runtime configs và Q8/Q5/Q4 có/không vision; đủ4chỉ tiêu | Flash-mm và hai bản Q8 vẫn thực hiện được tác vụ; SR screening thấp hơn defaults, chưa xác nhận giữ nguyên chất lượng |
| 4. LIBERO | Baseline screening38/40; official177/200 | Flash-mm35/40, Q8-lm36/40; Q8-vision36/40; official biến thể chưa đo |
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

## Screening biến thể — kết quả hoàn tất

| Cấu hình | Episode đã chốt | Thành công | Trạng thái |
|---|---:|---:|---|
| flash-mm | 40/40 | 35 | complete: 87,5% |
| q8-lm | 40/40 | 36 | complete: 90% |
| q8-vision | 40/40 | 36 | complete: 90% |

Đã chạy tuần tự flash+mm → Q8 không vision → Q8 có vision, tổng 120/120 episode; cả ba runner exit 0, batch complete. Kết quả cuối và CSV từng task: `research-notes/smolvla-followup-20261104/screen40-variants/`. Đối chiếu CSV bằng `audit_screening.py` trong cùng thư mục cha; báo cáo kiểm tra: `screening-final-audit.json`. Screening defaults 38/40, flash-mm 35/40 và hai bản Q8 36/40 chưa đủ để kết luận giữ SR hoặc xếp hạng chất lượng.

## Profile và pruning

[Kiến trúc và profile](research-notes/smolvla-followup-20261104/ARCHITECTURE_AND_PROFILE.md) giải thích phạm vi phase, module GPU-active và giới hạn nsys/WSL. [Ghi chú paper](PAPER_READING_SMOLVLA_CLP.md) nêu hướng CKA. Pruning hiện chỉ bỏ layer tail/spaced, không gọi là tái hiện CLP.

## Vấn đề cần mentor review

1. Hai CUDA tests fail và fused/unfused maxabs0,000545874: chưa tìm ra nguyên nhân; giữ defaults theo quyết định baseline của người dùng.
2. Peak allocation1912,47MiB chưa bao hết context/driver. NVML process memory trả N/A; không gọi là toàn bộ process resident VRAM.
3. Bảy xung đột dependency metadata còn tồn tại dù client chạy được. Môi trường chưa có lock portable sạch.
4. Random server noise và screening40 nhỏ làm khác biệt SR có độ bất định. Wilson interval tính trong phụ lục; chưa có repeated-seed evaluation.
5. Batch latency có khác biệt điều kiện nhiệt/tải; đề nghị thống nhất protocol final trước kết luận triển khai.
6. Chọn calibration riêng và kiểm tra ràng buộc layer VLM/expert, parity SA/CA trước CKA; không dùng episode đánh giá để chọn layer.

## Tài liệu và bằng chứng

- [Bảng kết quả theo task](research-notes/smolvla-followup-20261104/TASK_RESULTS.md) và [audit 360 episode](research-notes/smolvla-followup-20261104/submission-audit.json): kiểm tra lại 200 official + 160 screening từ CSV, có hash và loại lượt task 9 bị gián đoạn. Chạy lại bằng `python3 research-notes/smolvla-followup-20261104/audit_submission.py`; không cần inference mới.
- [Nhật ký](NOTES_SMOLVLA.md) là lịch sử, không dùng các dòng trạng thái cũ làm hiện trạng.
- [Report lịch sử](research-notes/review-20261104/REPORT_HISTORY.md) giữ nguyên snapshot cũ, có nội dung đã hết hiệu lực.
- [Review Markdown](MENTOR_REVIEW.md) liệt kê các vấn đề tài liệu và phương pháp.
- Dữ liệu nhỏ/scripts được commit; GGUF, video, trace nsys/sqlite, wheel và môi trường cài đặt không đưa lên Git.

## Task 1 — cài đặt, công cụ và tiêu chí đạt

CUDA architecture 86 khớp compute capability 8.6 của RTX 3060 Laptop; Release, `GGML_CUDA=ON`, `VLA_BUILD_TESTS=ON`, CUDA graphs bật. Incremental build exit0; đây chưa phải clean rebuild. Ctest 13/15 pass; `foldquant_cuda_op` và `fused_cuda_ops` fail hợp đồng bit-exact. Không nới tolerance để gọi pass. FoldQuant quan sát -3.4419589 so -3.44195867; chưa xác định nguyên nhân.

| Công cụ | Dùng để hiểu/kiểm tra | Kết quả |
|---|---|---|
| CLI | Inference one-shot qua terminal | CUDA trả chunk50×32 với ảnh/token synthetic |
| Server | Phục vụ request qua protobuf/transport | Request thật trả output hữu hạn, khớp bit harness khi cùng noise/input |
| Python binding | Gọi C ABI/libvla.so từ Python | Shape50×32, hữu hạn, khớp bit predict_check |
| vla_predict_check | Kiểm tra kỹ thuật với input/noise cố định | Real action7, padded32, chunk50; không đo hiểu instruction |
| ctest | Kiểm thử từng phần runtime/kernel | 13/15; còn2fail |

Lệnh tái hiện và log: [NOTES_SMOLVLA.md](NOTES_SMOLVLA.md#lệnh-đã-chạy-và-ý-nghĩa), `research-notes/smolvla-audit-20261104/ctest-build-cuda.log`, `interfaces.log`, `check_interfaces.py`. Môi trường LIBERO đã chạy thật dù còn7xung đột metadata dependency; chưa có lock portable sạch.

## Task 2 — bảng baseline độc lập

| Thông số | Giá trị đo / cấu hình |
|---|---|
| Mean / p50 / p90 |112,0 /111,8 /112,8 ms (cùng process được chọn)|
| Peak traced device allocation |1912,47 MiB; lấy peak lớn nhất3process|
| Input |2ảnh512×512;16language tokens;10denoise steps; chunk50×32|
| Protocol |3warmup+20reps/process;3process tuần tự; gap0|
| Phạm vi |Engine predict; không network/simulator|
| Điều kiện laptop |WSL2;6GiB; cắm sạc, nguồn bình thường, không tản nhiệt ngoài theo người dùng|

Telemetry runtime bổ sung ghi nhận53–72°C trong120mẫu, không phủ toàn bộ baseline; chưa đủ kết luận throttle. NVML process resident memory N/A. Có server cũ được phát hiện trong giai đoạn đo trước, chưa xác minh hoạt động request từng lượt; không khẳng định toàn bộ timing được cô lập. Lượt runtime bổ sung có guard từ chối khi còn server. Không suy nhiệt độ/throttle hoặc resident VRAM từ tính toán danh nghĩa.

## Đọc đủ Task 3–6 và phần chưa có số liệu

- Task3: [bảng đầy đủ runtime và sáu quantization](PART3_OPTIMIZATION.md), kèm input, sai lệch real7 và giới hạn memory.
- Task4: baseline official177/200; screening defaults38/40, flash-mm35/40, Q8-lm36/40, Q8-vision36/40. [Phụ lục tính toán](CALCULATIONS_SMOLVLA.md#task-4--sr-độ-bất-định-và-dự-toán-official) có Wilson95% và dự toán flash-mm175/200, Q8-lm180/200 và Q8-vision180/200, giả định SR không đổi. Các dự toán này chưa được chạy/xác nhận.
- Task5: [sơ đồ, phase timing và nsys](research-notes/smolvla-followup-20261104/ARCHITECTURE_AND_PROFILE.md); [tính tỷ trọng và dự báo theo depth](CALCULATIONS_SMOLVLA.md#task-5--phân-bổ-gpu-và-mô-hình-dự-báo). Flash-mm là ứng viên latency đã profile; chưa chốt best theo officialSR.
- Task6: [bảng ba pruning với file, allocation, p50/p90, sai lệch và smoke](CALCULATIONS_SMOLVLA.md#task-6--ba-pruning-đã-implement-và-kiểm-tra), implementation [prune_depth.py](research-notes/smolvla-followup-20261104/prune_depth.py). Đây là kết quả âm về chất lượng của depth pruning chưa finetune; chưa gọi là tái hiện CLP.

Tài liệu đã điền đủ phần giải thích, bằng chứng và tính toán cho6task; không đồng nghĩa mọi tiêu chí thực nghiệm đều đạt. Chưa thể thay bằng tính toán: hai testfail, residentVRAM, SRofficialbiến thể, attention/connector timing riêng, CKA/finetuning và repeated-seed. Đề xuất giữ defaults làm bản chất lượng đã được đo chính thức; ưu tiên đánh giá Q8-lm/Q8-vision cho memory và flash-mm cho latency, không chọn pruning0/2 để triển khai.


## Kết luận và giới hạn của đợt báo cáo

Các cấu hình flash-mm, Q8-lm và Q8-vision vẫn thực hiện được tác vụ điều khiển trong screening, nhưng tỷ lệ thành công quan sát thấp hơn defaults: flash-mm 35/40 (87,5%), hai bản Q8 cùng 36/40 (90%), so với defaults 38/40 (95%). Mức giảm quan sát tương ứng là 7,5 và 5 điểm phần trăm. Vì vậy, chưa xác nhận cấu hình nào giữ nguyên chất lượng điều khiển. Hai bản Q8 có kết quả screening cao hơn flash-mm, nhưng mẫu 40 episode chưa đủ để khẳng định khác biệt chất lượng. Kết luận này chỉ áp dụng cho ba cấu hình đã screening, không mở rộng sang các cấu hình runtime hoặc quantization khác chỉ có kiểm tra output synthetic.

Đợt báo cáo này chốt theo dữ liệu đã có: baseline official 200 episode và screening 40 episode cho mỗi cấu hình. Theo phạm vi đã thống nhất, không chạy thêm official cho biến thể hoặc thí nghiệm mới; tạm bỏ qua xử lý hai lỗi CUDA. Trạng thái kiểm thử vẫn là 13/15 pass, hai lỗi chưa sửa.

Screening hoàn tất ngày 05/11/2026. Defaults vẫn là mốc chất lượng chính thức 177/200. Flash-mm là ứng viên latency với mean 85,5 ms so với control 104,0 ms cùng batch. Q8-vision dùng allocation 925,88 MiB so với Q8-lm 1020,70 MiB và control quantization 1912,47 MiB; cả hai Q8 đạt 36/40. Chưa có kết quả cấu hình kết hợp quantization + flash-mm, nên không cộng các mức cải thiện.

Các mục dưới đây là giới hạn và hướng nghiên cứu tiếp theo, không phải công việc đang chạy trong đợt báo cáo này.

| Hạng mục chưa xác minh | Thực nghiệm cần có nếu mở rộng nghiên cứu |
|---|---|
| Hai lỗi CUDA bit-exact | Xác định nguyên nhân, sửa và chạy lại tests với hợp đồng hiện tại; không nới tolerance để đổi trạng thái |
| Chất lượng biến thể chính thức | Chạy đủ 200 episode/biến thể theo protocol đã công bố, lưu CSV và policy recovery; tối đa 600 episode cho ba biến thể |
| Độ ổn định theo seed | Định nghĩa noise/seed protocol và chạy lặp; không chọn layer hoặc chỉnh cấu hình trên episode đánh giá |
| Profile connector/attention | Thêm instrumentation tách module và đo lại, giữ rõ GPU-active và wall time |
| CLP/CKA | Calibration riêng, activation/CKA, chọn layer hợp lệ, finetuning và đánh giá mới; ba depth ablation hiện tại chưa đạt chất lượng |
| Môi trường tái lập | Giải quyết bảy dependency conflicts và tạo lock đã kiểm tra trên môi trường sạch |
| Resident VRAM | Đo trên backend hỗ trợ process memory; WSL hiện trả N/A nên chỉ báo traced allocation |

Bản báo cáo hoàn tất trong phạm vi dữ liệu hiện có. Chưa xác nhận giữ SR cho biến thể hoặc hiệu quả CLP/CKA; phép ngoại suy SR và mô hình depth chỉ là tính toán có giả định.

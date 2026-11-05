# Baseline và protocol lịch sử

Raw logs: `../smolvla-followup-20261104/runtime-full-pair/{defaults,flash-mm}-{0,1,2}.log`; JSON commands: [runs.json](../smolvla-followup-20261104/runtime-full-pair/runs.json). Audit mới parse cả6log và khớp các thống kê đã lưu.

Cùng model gốc,2views512×512,16token,10steps,3warmup,20reps/process;3lượt xen kẽ AB/BA/AB. State/noise synthetic cố định từ `src/serving/vla-bench.cpp:165–171`. Không dùng ảnh robot thật để suy ra latency end-to-end ứng dụng.

```bash
build-cuda/vla-bench --ckpt /home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf --images 2 --size 512 --tokens 16 --warmup 3 --reps 20
# Cặp optimized: cùng command, thêm
# --flash-attn --mm-prec default
```

`--images` số view;`--size` cạnh ảnh;`--tokens` số token synthetic;`--warmup` loại3lượt đầu khỏi summary;`--reps`20lượt đo;`--flash-attn` bật vision flash path;`--mm-prec default` bỏ ép accumulation F32 tại các LM matmul được `mm_w` điều khiển. Không phải cùng mọi lựa chọn precision: checkpoint giữ nguyên, runtime arithmetic thay đổi có chủ đích và output có sai khác.

| Trial | Defaults mean/p50/p90 ms | Flash-mm mean/p50/p90 ms |
|---|---|---|
|0|119.0 /119.1 /120.6|92.4 /92.5 /93.0|
|1|113.9 /113.9 /115.4|91.5 /91.5 /91.8|
|2|114.1 /114.1 /115.0|93.3 /93.3 /94.2|

Báo cáo dùng median của3run means=114,1/92,4; median của3run p50=114,1/92,5; median của3run p90=115,4/93,0. **Không phải percentile gộp60samples** (bench raw từng sample không được lưu trong log này). Không chọn minimum mean trong selected.json làm kết quả đại diện. Đây là baseline runtime default trên working tree lịch sử, không khẳng định là upstream original commit chưa sửa.

Trace có overhead: default-path node trace `ms_total` mean126,639/95,672; không thay thế benchmark không trace. Điều kiện GPU lịch sử có telemetry JSONL nhưng không đảm bảo toàn máy không có nhiễu mọi thời điểm; audit này chưa xác minh toàn bộ telemetry hoặc Windows processes.

Model hash kiểm tra lại khớp. Default profile log ghi BF16 tower;416 LM GEMM weights widened F32;weight buffer1763,1MiB. Flash-mm weight buffer985,0MiB. Đây là lượng allocation buffer được log, không peak resident VRAM.

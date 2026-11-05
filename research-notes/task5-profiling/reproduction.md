# Tái lập

## Audit CPU an toàn khi GPU đang bận

Từ repo root:

```bash
python3 research-notes/task5-profiling/audit_evidence.py
```

Không cần third-party Python packages; không chạy GPU. Mỗi lần tạo logs/timestamp/audit.json mới, kiểm tra6log, tính lại node trace và lưu SHA256. Phase boundary audit đã chạy với script gốc `summarize_phase_boundaries.py` trên symlink đặt trong thư mục log mới, truyền `--workload` tới kernel-phase-traces/{variant}-workload/result.json; không chạy script gốc trực tiếp trên path cũ vì nó ghi `.boundaries.json` cạnh SQLite.

## Khi GPU đã rảnh

Kiểm tra `git status`, `nvidia-smi`, không chạy chung server/simulator. Lưu commit,binary/model hashes, environment VLA_ overrides (kiểm tra cục bộ, tránh in secret),GPU telemetry và command vào thư mục mới. Không tự dừng tiến trình khác. Dùng command trong baseline-results.md và full_runtime_pair.py làm protocol tham khảo; không chạy script nguyên trạng vì output dir cũ đã tồn tại. AB/BA/AB,3warmup20reps mỗi process;giữ model,input,10steps và weight dtype,ghi runtime arithmetic khác biệt.

Chỉ sau khi GPU rảnh chạy serial:

```bash
ctest --test-dir build-cuda --output-on-failure -R '^(foldquant_cuda_op|foldquant_cuda_op_ref|fused_cuda_ops)$'
```

Lệnh này thực thi tests, có thể cập nhật CTest logs; sao lưu bằng chứng cũ trước. Fail phải giữ log,không giả định sửa xong.

## Command trace tham khảo (chưa chạy mới)

Chọn PREFIX mới chưa tồn tại; workload tự tạo OUTPUT_DIR mới. Dùng `.venv/bin/python` hiện có và script đã được đọc:

```bash
nsys profile --trace=cuda --cuda-graph-trace=node   --capture-range=cudaProfilerApi --capture-range-end=stop   --sample=none --cpuctxsw=none --output=NEW_PREFIX   .venv/bin/python research-notes/smolvla-followup-20261104/profile_phase.py   --variant defaults --mode none --capture --output NEW_OUTPUT_DIR
```

Đổi variant flash-mm và dùng prefix/output khác cho cặp. `--trace=cuda`ghi API/GPU;`node`ghi từng graph node có overhead;capture chỉ sau3warmup trước20predict;stop dừng capture;sample/ctxsw none tắt CPU sampling/context switches. `mode none`đo default path;đổi phase chỉ cho thí nghiệm phân rã riêng và công bố khác graph/padding. Script đặt flash-mm qua runtime config. Bỏ VLA_ overrides không mong muốn ở môi trường child; script `trace_kernels.py` cũ có ví dụ nhưng output dir cố định nên không chạy trực tiếp.

```bash
nsys export --type sqlite --output NEW_PREFIX.sqlite NEW_PREFIX.nsys-rep
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum NEW_PREFIX.nsys-rep
```

Kiểm tra diagnostics cả report/SQLite trước attribution; không bỏ qua “Not all CUDA events…”. Nsys stats có thể tạo artifact bên cạnh report nên chỉ dùng prefix mới. Để xem timeline mở `.nsys-rep` bằng Nsight Systems GUI tương thích. Chưa xác minh hardware metric devices hoặc counters trong lượt này,không thêm gpu-metrics flags tùy tiện.

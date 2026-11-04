#!/bin/bash
set -euo pipefail
cd /home/doris/projects/vla.cpp
B=research-notes/smolvla-followup-20261104
if pgrep -x vla-server >/dev/null; then echo 'Server still running; refusing concurrent GPU benchmarks'; exit 1; fi
python3 -u "$B/part3_runtime_children.py" latency
nsys profile --trace=cuda --cuda-memory-usage=true --cuda-trace-all-apis=true --sample=none --cpuctxsw=none --output="$B/part3-runtime/runtime" python3 -u "$B/part3_runtime_children.py" memory
nsys export --type sqlite --output="$B/part3-runtime/runtime.sqlite" "$B/part3-runtime/runtime.nsys-rep"
python3 "$B/audit_part3_memory.py"

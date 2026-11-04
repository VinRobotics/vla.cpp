"""Run serial baseline processes; retain raw NVML evidence, including N/A."""
from pathlib import Path
import json
import subprocess
import time

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
command = [str(root/'build-cuda/vla-bench'), '--ckpt', '/home/doris/.cache/vla/vrfai/smolvla-libero-gguf/smolvla-libero.gguf', '--images', '2', '--size', '512', '--tokens', '16', '--warmup', '3', '--reps', '20']
(out/'baseline-command.json').write_text(json.dumps(command, indent=2)+'\n')
for trial in range(1, 4):
    with (out/f'baseline-{trial}.log').open('w') as log, (out/f'telemetry-{trial}.jsonl').open('w') as telemetry:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            record = {'time': time.time(), 'pid': process.pid}
            for key, fields in [('gpu', '--query-gpu=temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,memory.used'), ('processes', '--query-compute-apps=pid,process_name,used_gpu_memory')]:
                result = subprocess.run(['nvidia-smi', fields, '--format=csv'], capture_output=True, text=True, timeout=10)
                record[key] = result.stdout + result.stderr
            telemetry.write(json.dumps(record)+'\n')
            telemetry.flush()
            time.sleep(.5)
        print('trial', trial, 'exit', process.returncode, flush=True)
        if process.returncode:
            raise SystemExit(process.returncode)

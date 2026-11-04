from pathlib import Path
import json,time
base=Path(__file__).resolve().parent
last=-1
for _ in range(900):
    try:rows=json.loads((base/'memory-all/children.json').read_text())
    except (OSError,ValueError):rows=[]
    if len(rows)!=last:
        last=len(rows)
        label=rows[-1]['name'] if rows else 'starting'
        print(f'MEMORY: {last}/30 process completed; latest={label}',flush=True)
    if (base/'memory-all/summary.json').exists():
        print('MEMORY: allocation audit complete; see memory-all/summary.json',flush=True);break
    try:log=(base/'memory-all-runner.log').read_text()
    except OSError:log=''
    if 'Traceback' in log:
        print('MEMORY: audit/profile reported an error; inspect memory-all-runner.log',flush=True);break
    time.sleep(2)

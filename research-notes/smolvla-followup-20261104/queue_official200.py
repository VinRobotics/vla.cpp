"""Wait for current smoke batch and its owned server to exit, then run official baseline."""
from pathlib import Path
import json,subprocess,sys,time,socket,os
b=Path(__file__).resolve().parent
print('QUEUED official200 defaults: waiting for smoke batch completion; no parallel GPU evaluation.',flush=True)
while True:
    try:p=json.loads((b/'candidate-smokes/progress.json').read_text())
    except (OSError,ValueError):time.sleep(5);continue
    if p.get('state')=='complete':
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',15558))!=0:break
    time.sleep(5)
print('START official200 defaults: 10 tasks x 20 episodes, repo defaults, seed42, random server noise.',flush=True)
result=subprocess.run([sys.executable,str(b/'run_official200.py')],cwd=b.parents[1])
print('Official runner exit:',result.returncode,flush=True)
sys.exit(result.returncode)

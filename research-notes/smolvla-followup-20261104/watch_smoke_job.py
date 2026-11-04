"""Read-only per-episode/request progress for existing VS Code dashboard."""
from pathlib import Path
import json,time
base=Path(__file__).resolve().parent;folder=base/'candidate-smokes'
last=''
for _ in range(1440):
    try:progress=json.loads((folder/'progress.json').read_text())
    except (OSError,ValueError):time.sleep(5);continue
    if progress.get('state')=='complete':
        print(f"SMOKE batch complete: {progress['completed']}/{progress['expected']} configurations; inspect results.json for successes/failures",flush=True);break
    name=progress.get('current','?');current=folder/name
    try:client=(current/'client.log').read_text(errors='replace').splitlines()
    except OSError:client=[]
    episodes=[line for line in client if line.startswith('*** Episode')]
    try:server=(current/'server.log').read_text(errors='replace').splitlines()
    except OSError:server=[]
    requests=[line for line in server if 'vla-server: rid=' in line]
    status=f"SMOKE {progress.get('completed',0)}/{progress.get('expected',10)} configs completed; current={name}; {episodes[-1] if episodes else 'loading'}; {requests[-1] if requests else 'waiting for server/client'}"
    if status!=last:print(status,flush=True);last=status
    time.sleep(5)

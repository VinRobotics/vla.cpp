"""Refresh managed report when smoke results/progress change; no GPU work."""
from pathlib import Path
import subprocess,sys,time,json
base=Path(__file__).resolve().parent
previous=None
for _ in range(1440):
    paths=[base/'candidate-smokes'/name for name in ('results.json','progress.json')]
    current=tuple(p.stat().st_mtime_ns if p.exists() else None for p in paths)
    if current!=previous:
        subprocess.run([sys.executable,str(base/'update_live_report.py')],check=True)
        previous=current
    try:
        if json.loads(paths[1].read_text()).get('state')=='complete':break
    except (OSError,ValueError):pass
    time.sleep(5)

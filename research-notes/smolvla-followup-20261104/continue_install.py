import os
from pathlib import Path
import subprocess
import time
out=Path(__file__).resolve().parent
pid=42602
print('Waiting for requirements installer PID',pid,flush=True)
while True:
 try: os.kill(pid,0)
 except ProcessLookupError: break
 time.sleep(3)
log=(out/'libero-resume.log').read_text()
if 'Installed ' not in log:
 raise SystemExit('Requirements installer ended without Installed marker; inspect libero-resume.log')
print('Requirements completed; continuing remaining repo setup steps',flush=True)
with (out/'libero-finish.log').open('w') as output:
 r=subprocess.run(['bash',str(out/'finish_libero.sh')],stdout=output,stderr=subprocess.STDOUT)
print('Remaining install exit',r.returncode,flush=True)
raise SystemExit(r.returncode)

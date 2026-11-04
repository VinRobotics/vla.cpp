"""Follow official evaluation logs across tasks; Ctrl+C stops viewer only."""
from pathlib import Path
import time
base=Path(__file__).resolve().parent
positions={}
print('Theo dõi LIBERO trực tiếp — Ctrl+C chỉ đóng bộ xem log.',flush=True)
try:
    while True:
        paths=[base/'official200-runner.log']
        paths+=sorted((base/'official200-defaults').glob('task-*.log'),key=lambda p:int(p.stem.split('-')[1]))
        for p in paths:
            try:
                offset=positions.get(p,0)
                if p.stat().st_size<offset:offset=0
                with p.open('rb') as f:
                    f.seek(offset);data=f.read();positions[p]=f.tell()
                if data:
                    print(f'\n[{p.name}]',flush=True)
                    print(data.decode(errors='replace'),end='',flush=True)
            except OSError:pass
        time.sleep(.5)
except KeyboardInterrupt:
    print('\nĐóng bộ xem; đánh giá vẫn tiếp tục.',flush=True)

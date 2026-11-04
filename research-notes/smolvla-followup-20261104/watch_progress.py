"""Read-only live dashboard for the VS Code integrated terminal. Ctrl+C exits viewer only."""
from pathlib import Path
import datetime,json,time
root=Path(__file__).resolve().parents[2]
base=Path(__file__).resolve().parent

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def count(folder,file,total):
    value=read(base/folder/file)
    if value is None:return f'Chưa có kết quả / {total}'
    if isinstance(value,list):n=len(value)
    else:return 'Hoàn tất' if value else 'Đang chạy'
    return f'{n}/{total}' + (' — hoàn tất' if n>=total else ' — đang chạy')
try:
    while True:
        now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
        lines=['SMOLVLA — TIẾN TRÌNH TRỰC TIẾP',now.strftime('%d/%m/%Y %H:%M:%S UTC+07'),'', 'Chỉ đọc log. Ctrl+C chỉ đóng bảng này, không dừng phép đo.','']
        progress=read(base/'screen40-defaults/progress.json',{})
        lines.append(f"LIBERO screening: {progress.get('completed_episodes',0)}/40 episode; success {progress.get('successes',0)}; {progress.get('state','chưa chạy')}")
        official=read(base/'official200-defaults/progress.json',{})
        lines.append(f"LIBERO official baseline: {official.get('completed_episodes',0)}/200 episode; success {official.get('successes',0)}; {official.get('state','đã xếp hàng, chờ smoke nhả GPU')}")
        task_logs=list((base/'official200-defaults').glob('task-*.log'))
        if task_logs:
            latest=max(task_logs,key=lambda p:p.stat().st_mtime)
            tail=latest.read_text(errors='replace').splitlines()
            episodes=[line for line in tail if line.startswith('*** Episode')]
            lines.append(f"Đang chạy {latest.stem}: {episodes[-1] if episodes else 'đang khởi tạo'}")
            lines.extend(tail[-5:])
        jobs=[('Runtime full, defaults/flash','runtime-full-pair','runs.json',6),('CUDA predict, model mới','candidate-checks','summary.json',11),('Memory flash+mm','memory-flash-mm','memory-summary.json',3),('Phase profiling','phase-suite','runs.json',15),('Benchmark GGUF/control','candidate-bench','runs.json',30),('Memory workload GGUF/control','memory-all','children.json',30)]
        for label,folder,file,total in jobs:lines.append(f'{label}: {count(folder,file,total)}')
        traces=sum((base/'kernel-traces'/f'{name}.summary.json').exists() for name in ['defaults','flash-mm'])
        lines.append(f'Kernel trace + phân tích: {traces}/2')
        runs=read(base/'candidate-bench/runs.json',[])
        if runs:
            last=runs[-1]
            lines.extend(['',f"Benchmark vừa xong: {last['name']}, lượt {last['trial']+1}/3, {last['status']}, mean={last.get('mean_ms','N/A')} ms"])
        logs=sorted(base.glob('*runner*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:2]
        for log in logs:
            lines.extend(['',str(log.relative_to(root))])
            try:lines.extend(log.read_text(errors='replace').replace('\r','\n').splitlines()[-10:])
            except OSError:lines.append('Đang đợi log...')
        lines.extend(['','Chi tiết kết quả: REPORT_SMOLVLA.md và NOTES_SMOLVLA.md'])
        print('\033[2J\033[H'+'\n'.join(lines),flush=True)
        time.sleep(2)
except KeyboardInterrupt:
    print('\nĐã đóng bảng theo dõi. Các phép đo vẫn chạy trong tiến trình riêng.')

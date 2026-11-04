"""Audit finalized episode CSVs and render a task-level review appendix; no inference."""
import json

from audit_screening import DATA, audit, read, require


def main():
    batch = read(DATA / 'screen40-variants/batch.json')
    names = ['flash-mm', 'q8-lm', 'q8-vision']
    require(batch['state'] == 'complete' and batch['current'] is None, 'unfinished batch')
    require([r['name'] for r in batch['results']] == names, 'batch configurations')
    require(all(r['exit'] == 0 for r in batch['results']), 'runner failure')
    results = {'defaults': audit('defaults', DATA / 'screen40-defaults')}
    for name in names:
        results[name] = audit(name, DATA / 'screen40-variants' / name)
    results['official-defaults'] = audit(
        'official-defaults', DATA / 'official200-defaults', 20, 'official_sr')
    # The manifest selects the recovered task 9, excluding all interrupted attempts.
    require(sum(r['episodes'] for r in results.values()) == 360, 'episode total')
    report = {
        'state': 'verified',
        'scope': 'Final manifest CSVs only; no video, kernel, environment or model-quality certification',
        'episodes': 360,
        'results': results,
    }
    (DATA / 'submission-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = [
        '# Đối chiếu kết quả LIBERO theo task',
        '',
        'Sinh từ CSV bằng `python3 research-notes/smolvla-followup-20261104/audit_submission.py` tại gốc repo. '
        'Kiểm tra 360 episode: 200 official defaults, 40 screening defaults và 120 screening biến thể. '
        'Kiểm tra task/episode ID, số bước, success, aborted và SR so với manifest; lưu SHA256 từng CSV trong '
        '[submission-audit.json](submission-audit.json). Không chạy inference mới hoặc kiểm tra lại video.',
        '',
        '| Task | Defaults /4 | Flash-mm /4 | Q8-lm /4 | Q8-vision /4 | Official defaults /20 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for task_id in range(10):
        values = [str(r['tasks'][task_id]['successes']) for r in results.values()]
        lines.append('| ' + ' | '.join([str(task_id)] + values) + ' |')
    lines.extend([
        '| Tổng | ' + ' | '.join(str(r['successes']) for r in results.values()) + ' |',
        '',
        'Mọi CSV được chọn có aborted=0. Official task 9 lấy đường dẫn recovery trong result.json; '
        'không cộng episode của lượt bị gián đoạn. Hash chỉ xác nhận nội dung artifact tại lúc audit, '
        'không chứng minh môi trường hoặc noise có thể tái hiện từng bit.',
        '',
        'Bảng giúp xác định task cần xem lại video. Chênh lệch thành công không xác định nguyên nhân '
        'do quantization, attention hay noise. Cùng episode ID và seed môi trường chưa tạo thành cặp '
        'đối chứng cùng noise; không dùng bảng này để kiểm định ghép cặp hoặc chọn cấu hình trên tập đánh giá.',
        '',
        'Chỉ so các cột screening với nhau khi mô tả chênh lệch quan sát. Cột official có số episode '
        'khác và là một lượt chạy riêng; không lấy 177/200 làm control cho screening biến thể.',
        '',
    ])
    (DATA / 'TASK_RESULTS.md').write_text('\n'.join(lines))
    for name, result in results.items():
        print(f"{name}: {result['successes']}/{result['episodes']}, aborted={result['aborted']}")
    print('Verified 360 episodes; wrote submission-audit.json and TASK_RESULTS.md.')


if __name__ == '__main__':
    main()

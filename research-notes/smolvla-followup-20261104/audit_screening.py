"""Validate final screening results against every per-episode CSV."""
import csv
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(name, directory, episodes_per_task=4, sr_key='screening_sr'):
    result = read(directory / 'result.json')
    require(result['state'] == 'complete', f'{name}: unfinished result')
    tasks = result['tasks']
    require(sorted(t['task_id'] for t in tasks) == list(range(10)), f'{name}: task IDs')
    episodes = successes = aborted = 0
    records = []
    for task in tasks:
        path = Path(task['csv'])
        if not path.is_absolute():
            path = ROOT / path
        with path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == task['episodes'] == episodes_per_task, f'{name}: task episode count')
        require(sorted(int(r['episode']) for r in rows) == list(range(episodes_per_task)), f'{name}: episode IDs')
        require(all(r['success'] in ('0', '1') and r['aborted'] in ('0', '1') for r in rows), f'{name}: flags')
        count = sum(int(r['success']) for r in rows)
        skips = sum(int(r['aborted']) for r in rows)
        require(count == task['successes'], f'{name}: success mismatch')
        require([int(r['steps']) for r in rows] == task['steps'], f'{name}: steps mismatch')
        require(all(int(r['steps']) > 0 for r in rows), f'{name}: invalid steps')
        episodes += len(rows)
        successes += count
        aborted += skips
        records.append({'task_id': task['task_id'], 'episodes': len(rows), 'successes': count,
                        'aborted': skips, 'csv': str(path.relative_to(ROOT)),
                        'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'episode_records': [{key: int(row[key]) for key in ('episode', 'success', 'steps', 'aborted')} for row in rows]})
    require(episodes == result['completed_episodes'] == result['expected_episodes'] == 10 * episodes_per_task, f'{name}: total episodes')
    require(result['completed_tasks'] == 10, f'{name}: total tasks')
    require(successes == result['successes'] and aborted == 0, f'{name}: final totals')
    require(abs(result[sr_key] - successes / episodes) < 1e-12, f'{name}: SR')
    return {'episodes': episodes, 'successes': successes, 'aborted': aborted,
            'screening_sr': successes / episodes, 'tasks': records}


def main():
    batch = read(DATA / 'screen40-variants/batch.json')
    names = ['flash-mm', 'q8-lm', 'q8-vision']
    require(batch['state'] == 'complete' and batch['current'] is None, 'unfinished batch')
    require([r['name'] for r in batch['results']] == names, 'batch configurations')
    require(all(r['exit'] == 0 for r in batch['results']), 'runner failure')
    require(batch['expected_configurations'] == 3 and batch['expected_episodes'] == 120, 'batch counts')
    results = {'defaults': audit('defaults', DATA / 'screen40-defaults')}
    for name in names:
        results[name] = audit(name, DATA / 'screen40-variants' / name)
    updated = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    output = {'updated_at': updated, 'state': 'verified', 'scope': 'CSV counts, IDs, steps, SR and runner exits; no video audit',
              'variant_episodes': sum(results[n]['episodes'] for n in names), 'results': results}
    (DATA / 'screening-final-audit.json').write_text(json.dumps(output, indent=2) + '\n')
    snapshot = {'updated_at': updated, 'classification': 'screening complete; variant official200 not measured',
                'variants': {n: read(DATA / 'screen40-variants' / n / 'result.json') for n in names}}
    (ROOT / 'research-notes/review-20261104/submission-progress.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    for name, result in results.items():
        print(f"{name}: {result['successes']}/{result['episodes']}, aborted={result['aborted']}")
    print('Verified 120 variant episodes and 40 baseline screening episodes.')


if __name__ == '__main__':
    main()

"""Mentor task 4/6: evaluate existing depth pruning, 10 tasks x 4 episodes each.

Uses the established protocol runner; preserves all earlier evaluation artifacts.
"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
base = root / 'research-notes/smolvla-followup-20261104'
output = Path(__file__).resolve().parent / 'screening-batch.json'
names = ['smolvla-vision-tail-10', 'smolvla-vision-spaced-10', 'smolvla-coupled-tail-30']

def main():
    if output.exists():
        raise SystemExit('Batch exists; inspect results before resuming')
    if subprocess.run(['pgrep', '-x', 'vla-server'], stdout=subprocess.DEVNULL).returncode == 0:
        raise SystemExit('Existing server; refuse concurrent evaluation')
    models = [root / 'research-notes/generated-models' / (name + '.gguf') for name in names]
    if not all(model.is_file() for model in models):
        raise SystemExit('Missing pruned GGUF')
    records = []
    for name, model in zip(names, models):
        output.write_text(json.dumps(dict(state='running', current=name, results=records), indent=2)+'\n')
        command = [sys.executable, '-u', str(base / 'run_variant40.py'), name, str(model)]
        result = subprocess.run(command, cwd=root)
        records.append(dict(name=name, command=command, exit=result.returncode))
        if result.returncode:
            output.write_text(json.dumps(dict(state='failed', results=records), indent=2)+'\n')
            raise SystemExit(result.returncode)
    output.write_text(json.dumps(dict(state='complete', results=records), indent=2)+'\n')

if __name__ == '__main__':
    main()

"""Offline CKA selection for activation exports; never uses evaluation outcomes.

NPZ keys: vit.0 ... vit.11, or vlm.0 ... vlm.31 and aex.0 ... aex.31.
Each array is [aligned observations/tokens, hidden features], not pooled means.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def linear_cka(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 3:
        raise ValueError('Expected aligned matrices with at least three rows')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Non-finite activations')
    x, y = x - x.mean(0), y - y.mean(0)
    # Choose smaller feature or sample formulation; both are exact linear CKA.
    if x.shape[0] < max(x.shape[1], y.shape[1]):
        a, b = x @ x.T, y @ y.T
        numerator = np.sum(a * b)
    else:
        a, b = x.T @ x, y.T @ y
        numerator = np.sum((x.T @ y) ** 2)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError('Degenerate activations')
    return float(np.clip(numerator / denominator, 0, 1))


def select(scores, budget, threshold, period=1):
    """Maximum-score constrained selection; keep plateau anchors and block types.

    DP state counts dropped layers so old/new attention parity remains valid.
    Coupled scores use the minimum VLM/expert adjacent CKA, a conservative
    adaptation to this loader, rather than independent CLP selections.
    """
    n = len(scores)
    if not 0 <= budget < n or period < 1 or not 0 <= threshold <= 1:
        raise ValueError('Invalid budget, threshold or attention period')
    states = {0: (0., [])}
    for old in range(n):
        following = {}
        for dropped, (value, indices) in states.items():
            choices = []
            if (old % period == 0) == ((old - dropped) % period == 0):
                choices.append((dropped, value, indices))
            if old > 0 and scores[old] >= threshold and dropped < budget:
                choices.append((dropped + 1, value + scores[old], indices + [old]))
            for count, score, chosen in choices:
                if count not in following or score > following[count][0]:
                    following[count] = (score, chosen)
        states = following
    if budget not in states:
        raise ValueError('No feasible selection: lower threshold or change budget; no fallback to tail')
    return states[budget][1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--activations', type=Path, required=True)
    p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--module', choices=['vit', 'coupled'], required=True)
    p.add_argument('--budget', type=int, required=True)
    p.add_argument('--threshold', type=float, default=.95)
    p.add_argument('--attention-period', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Refuse overwrite')
    provenance = json.loads(args.provenance.read_text())
    for field in ['checkpoint_sha256', 'dataset', 'calibration_episode_ids', 'evaluation_episode_ids', 'row_order', 'capture_description']:
        if not provenance.get(field):
            p.error('Missing provenance: ' + field)
    if set(provenance['calibration_episode_ids']) & set(provenance['evaluation_episode_ids']):
        p.error('Calibration/evaluation episode overlap')
    modules = ['vit'] if args.module == 'vit' else ['vlm', 'aex']
    with np.load(args.activations, allow_pickle=False) as data:
        profiles = {}
        counts = []
        rows = set()
        for module in modules:
            keys = sorted(int(k.split('.')[1]) for k in data.files if k.startswith(module + '.'))
            if keys != list(range(len(keys))) or len(keys) < 2:
                p.error('Missing/invalid contiguous layer exports for ' + module)
            counts.append(len(keys))
            rows.update(data[f'{module}.{i}'].shape[0] for i in keys)
            profiles[module] = [0.] + [linear_cka(data[f'{module}.{i-1}'], data[f'{module}.{i}']) for i in keys[1:]]
        if len(set(counts)) != 1:
            p.error('Coupled layer counts differ')
        # VLM and expert token counts may differ; alignment is within each module.
        scores = np.min([profiles[m] for m in modules], axis=0).tolist()
    drops = select(scores, args.budget, args.threshold, 1 if args.module == 'vit' else args.attention_period)
    flag = '--vision-drop' if args.module == 'vit' else '--coupled-drop'
    report = dict(method='CKA-guided adaptation; no finetuning', module=args.module,
                  budget=args.budget, threshold=args.threshold, profiles=profiles, drop=drops,
                  exporter_args=[flag, ','.join(map(str, drops))], provenance=provenance,
                  activation_sha256=hashlib.file_digest(args.activations.open('rb'), 'sha256').hexdigest(),
                  inference_validated=False, sr_measured=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

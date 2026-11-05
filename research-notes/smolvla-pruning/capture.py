"""PyTorch hook utility for a checkpoint-specific calibration adapter.

The caller supplies actual transformer blocks and runs real training observations.
For expert blocks called repeatedly, select one fixed solver step using occurrence.
No automatic module-name guessing or checkpoint conversion is performed.
"""
import numpy as np


class ActivationCapture:
    def __init__(self, blocks, occurrence=0, max_rows_per_example=128):
        if occurrence < 0 or max_rows_per_example < 3:
            raise ValueError('Invalid capture settings')
        self.blocks = blocks  # {'vit.0': torch_module, ...}
        self.occurrence = occurrence
        self.max_rows = max_rows_per_example
        self.chunks = {key: [] for key in blocks}
        self.handles = []
        self.counts = {}
        self.current = {}

    def __enter__(self):
        for key, block in self.blocks.items():
            def hook(module, inputs, output, key=key):
                count = self.counts.get(key, 0)
                self.counts[key] = count + 1
                if count != self.occurrence:
                    return
                if isinstance(output, tuple):
                    output = output[0]
                # Adapter must ensure features are last and exclude padding.
                values = output.detach().float().cpu().numpy()
                values = values.reshape(-1, values.shape[-1])
                index = np.linspace(0, len(values)-1, min(len(values), self.max_rows), dtype=int)
                self.current[key] = values[index].copy()
            self.handles.append(block.register_forward_hook(hook))
        return self

    def begin_example(self):
        self.counts.clear()
        self.current.clear()

    def end_example(self):
        if set(self.current) != set(self.blocks):
            raise ValueError('Not all blocks captured at selected occurrence')
        for module in {key.split('.')[0] for key in self.blocks}:
            shapes = {value.shape[0] for key, value in self.current.items() if key.startswith(module + '.')}
            if len(shapes) != 1:
                raise ValueError('Unaligned layer token rows')
        for key, value in self.current.items():
            self.chunks[key].append(value)

    def save(self, path):
        from pathlib import Path
        if Path(path).exists():
            raise ValueError('Refuse overwrite')
        if not all(self.chunks.values()):
            raise ValueError('No completed calibration examples')
        np.savez(path, **{key: np.concatenate(value) for key, value in self.chunks.items()})

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

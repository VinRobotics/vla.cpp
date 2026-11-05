import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from capture import ActivationCapture


class CaptureTests(unittest.TestCase):
    def test_occurrence_alignment_and_hook_cleanup(self):
        a, b = torch.nn.Identity(), torch.nn.Identity()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'features.npz'
            with ActivationCapture({'vit.0': a, 'vit.1': b}, occurrence=1) as capture:
                for example in range(2):
                    capture.begin_example()
                    for step in range(2):
                        x = torch.arange(24).reshape(1, 8, 3).float() + step + example
                        b(a(x))
                    capture.end_example()
                capture.save(path)
            with np.load(path) as data:
                self.assertEqual(data['vit.0'].shape, (16, 3))
                np.testing.assert_array_equal(data['vit.0'], data['vit.1'])
                self.assertEqual(data['vit.0'][0, 0], 1)
            self.assertFalse(a._forward_hooks)
            self.assertFalse(b._forward_hooks)

    def test_missing_layer_does_not_commit_example(self):
        a, b = torch.nn.Identity(), torch.nn.Identity()
        with ActivationCapture({'vit.0': a, 'vit.1': b}) as capture:
            capture.begin_example()
            a(torch.ones(5, 3))
            with self.assertRaises(ValueError):
                capture.end_example()
            self.assertFalse(capture.chunks['vit.0'])

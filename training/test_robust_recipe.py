"""Focused regressions without downloading data or running the notebook at import."""
import ast
from pathlib import Path
import unittest

import numpy as np
import scipy.signal
import torch


def recipe():
    tree = ast.parse(Path(__file__).with_name("wesep_retrain.py").read_text())
    names = {"rms", "discovered_style", "environmental_noise", "channel_augment", "frame_energy_loss", "tse_loss"}
    functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name in names], type_ignores=[])
    scope = {"np": np, "scipy": scipy, "torch": torch, "RATE": 16000, "ROBUST": True, "TAU": 1e-3}
    exec(compile(functions, "recipe-functions", "exec"), scope)
    return scope


class NoGaps:
    def uniform(self, low, high):
        return (low + high) / 2

    def integers(self, *args):
        return 0


class RobustRecipeTests(unittest.TestCase):
    def setUp(self):
        self.scope = recipe()

    def test_four_second_leak_reaches_five_second_enrollment(self):
        signal = np.ones(80000, dtype="float32") * 0.1
        leak = np.ones(64000, dtype="float32") * 0.05
        augment = self.scope["discovered_style"]
        clean = augment(NoGaps(), signal, None)
        mixed = augment(NoGaps(), signal, leak)
        self.assertEqual(mixed.shape, signal.shape)
        self.assertGreater(float(np.linalg.norm(mixed[:64000] - clean[:64000])), 1)
        np.testing.assert_allclose(mixed[64000:], clean[64000:])
        self.scope["ROBUST"] = False
        np.testing.assert_allclose(augment(NoGaps(), signal, leak), clean)

    def test_environment_and_channel_preserve_shape_and_are_finite(self):
        for seed in range(12):
            noise = self.scope["environmental_noise"](np.random.default_rng(seed), 64000)
            self.assertEqual(noise.shape, (64000,))
            self.assertTrue(np.isfinite(noise).all())
            self.assertGreater(float(np.linalg.norm(noise)), 0)
            filtered = self.scope["channel_augment"](np.random.default_rng(seed), noise)
            self.assertEqual(filtered.shape, noise.shape)
            self.assertTrue(np.isfinite(filtered).all())

    def test_energy_term_preserves_natural_loudness_and_penalizes_attenuation(self):
        target = torch.cat([torch.ones(1600) * 0.01, torch.ones(1600) * 0.1])[None]
        energy = self.scope["frame_energy_loss"]
        self.assertAlmostEqual(float(energy(target, target)), 0)
        self.assertAlmostEqual(float(energy(target * 0.5, target)), 6.0206, places=3)

    def test_absent_and_present_targets_have_finite_loss_and_gradients(self):
        for present in (False, True):
            generator = torch.Generator().manual_seed(2026)
            mix = torch.randn(2, 6400, generator=generator) * 0.05
            target = mix * 0.5 if present else torch.zeros_like(mix)
            estimate = (mix * 0.1).requires_grad_()
            loss = self.scope["tse_loss"](estimate, target, mix)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(estimate.grad).all())


if __name__ == "__main__":
    unittest.main()

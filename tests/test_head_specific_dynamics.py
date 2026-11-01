"""Small CPU smoke test for opt-in head-specific NextLat dynamics.

Run at repository root:
    python -m unittest tests.test_head_specific_dynamics
"""
import unittest

import torch

from models.model_nextlat import (
    HeadSpecificDynamicsModel,
    NextLatConfig,
    NextLatDynamicsModel,
    NextLatTransformer,
)


class HeadSpecificDynamicsTests(unittest.TestCase):
    def test_shape_and_gradients(self):
        torch.manual_seed(0)
        cfg = NextLatConfig(
            vocab_size=64, block_size=12, n_layer=2, n_head=4, n_embd=32,
            head_specific_dynamics=True,
        )
        model = NextLatTransformer(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 8))
        embeddings, hidden, heads = model(
            tokens, return_hidden_states=True, return_head_outputs=True
        )
        self.assertEqual(heads.shape, (2, 8, 4, 8))
        prediction = model.dynamics_model(
            hidden[:, :-1], embeddings[:, 1:], head_features=heads[:, :-1]
        )
        self.assertEqual(prediction.shape, (2, 7, 32))
        second_step = model.dynamics_model(prediction[:, :-1], embeddings[:, 2:])
        self.assertEqual(second_step.shape, (2, 6, 32))
        second_step.square().mean().backward()
        self.assertIsNotNone(model.transformer.blocks[-1].attn.c_attn.weight.grad)
        self.assertIsNotNone(model.dynamics_model.gate_logits.weight.grad)

    def test_baseline_and_uniform_ablation(self):
        cfg = NextLatConfig(
            vocab_size=64, block_size=12, n_layer=2, n_head=4, n_embd=32,
            head_specific_dynamics=False,
        )
        baseline = NextLatTransformer(cfg)
        self.assertIsInstance(baseline.dynamics_model, NextLatDynamicsModel)
        cfg.head_specific_dynamics = True
        cfg.head_uniform_gate = True
        ablation = NextLatTransformer(cfg)
        self.assertIsInstance(ablation.dynamics_model, HeadSpecificDynamicsModel)
        prediction = ablation.dynamics_model(
            torch.randn(2, 32), torch.randn(2, 32)
        )
        self.assertEqual(prediction.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()

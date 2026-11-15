"""Small CPU smoke test for opt-in head-specific NextLat dynamics.

Run at repository root:
    python -m unittest tests.test_head_specific_dynamics
"""
import unittest

import torch

from models.model_nextlat import (
    HeadSpecificDynamicsModel,
    NextLat,
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

    def _latent_metrics(self, head_specific, uniform_gate=False, horizon=3):
        torch.manual_seed(0)
        cfg = NextLatConfig(
            vocab_size=64, block_size=12, n_layer=2, n_head=4, n_embd=32,
            mtp_horizon=horizon, eos_token_id=63,
            head_specific_dynamics=head_specific, head_uniform_gate=uniform_gate,
        )
        nextlat = NextLat(cfg)
        tokens = torch.randint(0, 63, (2, 10))
        outputs = nextlat.model(
            tokens, return_hidden_states=True, return_head_outputs=head_specific
        )
        embeds, hidden = outputs[0], outputs[1]
        heads = outputs[2] if head_specific else None
        eos = tokens == cfg.eos_token_id
        losses = nextlat._nextlat_compute_losses(
            hidden, embeds, tokens[:, 1:], ~eos[:, :-1], eos, head_outputs_det=heads
        )
        self.assertEqual(len(losses), 5)
        return losses[-1]

    def test_latent_metrics_learned_gate(self):
        metrics = self._latent_metrics(head_specific=True)
        for k in range(1, 4):
            self.assertIn(f"mse_step_{k}", metrics)
            self.assertIn(f"nsl1_step_{k}", metrics)
            self.assertIn(f"gate/entropy_step_{k}", metrics)
            self.assertTrue(0.0 <= metrics[f"gate/entropy_step_{k}"].item() <= 1.0 + 1e-5)
        self.assertIn("nsl1", metrics)
        self.assertIn("gate/entropy", metrics)
        head_weights = sum(metrics[f"gate/w_head{h}"] for h in range(4))
        self.assertAlmostEqual(head_weights.item(), 1.0, places=5)
        self.assertTrue(all(not v.requires_grad for v in metrics.values()))

    def test_latent_metrics_baseline_and_uniform(self):
        for head_specific, uniform in [(False, False), (True, True)]:
            metrics = self._latent_metrics(head_specific, uniform, horizon=2)
            self.assertIn("nsl1_step_2", metrics)
            self.assertFalse(any(k.startswith("gate/") for k in metrics))


if __name__ == "__main__":
    unittest.main()

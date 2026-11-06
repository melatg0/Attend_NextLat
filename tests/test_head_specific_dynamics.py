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
            vocab_size=64, block_size=12, n_layer=2, n_head=4, n_embd=64,
            head_specific_dynamics=True,
        )
        model = NextLatTransformer(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, 8))
        embeddings, hidden, heads = model(
            tokens, return_hidden_states=True, return_head_outputs=True
        )
        self.assertEqual(heads.shape, (2, 8, 4, 16))
        prediction = model.dynamics_model(
            hidden[:, :-1], embeddings[:, 1:], head_features=heads[:, :-1]
        )
        self.assertEqual(prediction.shape, (2, 7, 64))
        second_step = model.dynamics_model(prediction[:, :-1], embeddings[:, 2:])
        self.assertEqual(second_step.shape, (2, 6, 64))
        second_step.square().mean().backward()
        self.assertIsNotNone(model.transformer.blocks[-1].attn.c_attn.weight.grad)
        self.assertIsNotNone(model.dynamics_model.gate_logits.weight.grad)

    def test_baseline_and_uniform_ablation(self):
        cfg = NextLatConfig(
            vocab_size=64, block_size=12, n_layer=2, n_head=4, n_embd=64,
            head_specific_dynamics=False,
        )
        baseline = NextLatTransformer(cfg)
        self.assertIsInstance(baseline.dynamics_model, NextLatDynamicsModel)
        cfg.head_specific_dynamics = True
        cfg.head_uniform_gate = True
        ablation = NextLatTransformer(cfg)
        self.assertIsInstance(ablation.dynamics_model, HeadSpecificDynamicsModel)
        prediction = ablation.dynamics_model(
            torch.randn(2, 64), torch.randn(2, 64)
        )
        self.assertEqual(prediction.shape, (2, 64))


class TrainingRegressionTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = dict(vocab_size=16, block_size=12, n_layer=1, n_head=4,
                      n_embd=64, eos_token_id=15, head_specific_dynamics=True)
        values.update(overrides)
        return NextLatConfig(**values)

    def test_positive_width_and_parameter_budget(self):
        for dim, heads, factor in [(32, 4, 1), (384, 8, 2), (384, 6, .5)]:
            cfg = self.config(n_embd=dim, n_head=heads, proj_factor=factor)
            baseline = NextLatDynamicsModel(cfg)
            feature = HeadSpecificDynamicsModel(cfg)
            self.assertGreater(baseline.mlp[0].out_features, 0)
            self.assertLessEqual(abs(feature.get_num_params() - baseline.get_num_params()), dim)

    def test_returning_heads_preserves_logits(self):
        model = NextLatTransformer(self.config()).eval()
        tokens = torch.tensor([[1, 2, 15, 3, 4]])
        logits = model(tokens)
        _, hidden, _ = model(tokens, return_hidden_states=True, return_head_outputs=True)
        torch.testing.assert_close(logits, model.lm_head(hidden), rtol=0, atol=0)

    def test_empty_auxiliary_targets_are_finite(self):
        from models.model_nextlat import NextLat
        model = NextLat(self.config())
        prediction = torch.randn(2, 4, 64, requires_grad=True)
        losses = model._nextlat_loss_function(
            prediction, torch.randn_like(prediction),
            torch.full((2, 3), -100), torch.randn(2, 3, 16),
            torch.zeros(2, 3, dtype=torch.bool), torch.zeros(2, 4, dtype=torch.bool),
        )
        for loss in losses:
            self.assertTrue(torch.isfinite(loss).all())
            self.assertEqual(loss.item(), 0)
        sum(losses).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_rollout_masks_exclude_all_crossed_boundaries(self):
        from models.model_nextlat import NextLat
        from unittest.mock import patch
        model = NextLat(self.config(mtp_horizon=3))
        tokens = torch.tensor([[1, 2, 15, 3, 4, 5, 6]])
        embeddings, hidden, heads = model.model(
            tokens, return_hidden_states=True, return_head_outputs=True)
        eos = tokens == 15
        with patch.object(model, '_nextlat_loss_function',
                          wraps=model._nextlat_loss_function) as compute:
            model._nextlat_compute_losses(hidden, embeddings, tokens[:, 1:],
                                          ~eos[:, :-1], eos, heads)
        expected = [
            [True, False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, True, True],
        ]
        for call, valid in zip(compute.call_args_list, expected):
            self.assertEqual(call.kwargs['mse_mask'].tolist(), [valid])
            self.assertEqual(call.kwargs['token_pred_mask'].tolist(), [valid[:-1]])

    def test_training_gradients_match_direct_autograd(self):
        import copy
        import contextlib
        import io
        import lightning as L
        from models.model_nextlat import NextLat
        for feature, uniform, horizon in [(False, False, 1), (True, False, 1),
                                           (True, False, 3), (True, True, 3)]:
            with self.subTest(feature=feature, uniform=uniform, horizon=horizon):
                torch.manual_seed(42)
                cfg = self.config(head_specific_dynamics=feature,
                                  head_uniform_gate=uniform, mtp_horizon=horizon,
                                  context_length=1)
                actual = NextLat(cfg)
                reference = copy.deepcopy(actual)
                fabric = L.Fabric(accelerator='cpu', devices=1, precision='32-true')
                with contextlib.redirect_stdout(io.StringIO()):
                    actual.setup_fabric(fabric)
                tokens = torch.tensor([[1, 2, 3, 15, 4, 5, 6, 7],
                                       [15, 1, 2, 3, 4, 5, 6, 7]])
                loss_div = 2 if horizon > 1 else 1
                logged = actual.compute_loss(tokens, backpropagate=True, loss_div=loss_div)
                self.assertTrue(all(torch.isfinite(v).all() for v in logged.values()))
                outputs = reference.model(tokens, return_hidden_states=True,
                                          return_head_outputs=feature)
                embeddings, hidden = outputs[:2]
                heads = outputs[2] if feature else None
                eos = tokens == cfg.eos_token_id
                targets = tokens[:, 1:]
                pos = reference.model.create_position_indices(tokens)
                context = (pos <= cfg.context_length + 1) & (pos != 0)
                targets = targets.masked_fill(context[:, 1:], -100)
                mask = ~eos[:, :-1] & ~context[:, 1:]
                ntp, latent, kl, ce = reference._nextlat_compute_losses(
                    hidden, embeddings, targets, mask, eos, heads)
                loss = ntp + (cfg.lambda_mse * latent + cfg.lambda_kl * kl
                              + cfg.lambda_ce * ce.sum()) / horizon
                loss = loss / loss_div
                loss.backward()
                torch.testing.assert_close(logged['loss'], loss.detach())
                actual_params = dict(actual.model.module.named_parameters())
                for name, param in reference.model.named_parameters():
                    observed = actual_params[name].grad
                    if param.grad is None:
                        self.assertTrue(observed is None or torch.count_nonzero(observed) == 0, name)
                    else:
                        self.assertIsNotNone(observed, name)
                        torch.testing.assert_close(observed, param.grad, rtol=2e-4, atol=2e-6,
                                                   msg=lambda msg: name + ': ' + msg)
                if feature and not uniform:
                    self.assertGreater(actual_params['dynamics_model.gate_logits.weight'].grad.norm(), 0)
                if feature and horizon > 1:
                    self.assertGreater(actual_params['dynamics_model.state_to_heads.weight'].grad.norm(), 0)

    def test_invalid_horizon_is_rejected(self):
        import lightning as L
        from models.model_nextlat import NextLat
        model = NextLat(self.config(mtp_horizon=3))
        model.fabric = L.Fabric(accelerator='cpu')
        with self.assertRaisesRegex(ValueError, 'sequence length minus 2'):
            model.compute_loss(torch.ones(1, 4, dtype=torch.long), backpropagate=False)


if __name__ == "__main__":
    unittest.main()

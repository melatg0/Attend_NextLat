"""Check that planning accuracy excludes the already-known source node."""
import unittest
from types import SimpleNamespace

import torch

from data.stargraph import StarGraphDataset


class StarGraphMetricsTests(unittest.TestCase):
    def test_first_step_is_first_neighbor_not_source(self):
        class Model:
            def eval(self):
                pass

            def generate(self, prefix, **kwargs):
                # Both source copies are correct, but only one arm choice is.
                return torch.cat([prefix, torch.tensor([[1, 2, 3], [1, 9, 3]])], dim=1)

        fabric = SimpleNamespace(device=torch.device('cpu'), world_size=1,
                                 all_reduce=lambda value, **kwargs: value,
                                 print=lambda *args: None)
        dataset = StarGraphDataset([], None, graph_target_len=3, num_arms=2)
        batch = torch.tensor([[8, 8, 1, 2, 3, 15], [8, 8, 1, 2, 3, 15]])
        logs = dataset.evaluate_stargraph(Model(), [batch],
                                         SimpleNamespace(trainer=SimpleNamespace(test_batches=-1)),
                                         fabric, show_progress_bar=False)
        self.assertEqual(logs['val_(2, 3)/token_1'], 1)
        self.assertEqual(logs['val_(2, 3)/first_step_accuracy'], .5)
        self.assertEqual(logs['val_(2, 3)/test_accuracy'], .5)

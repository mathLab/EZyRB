import numpy as np
import torch

from unittest import TestCase
from ezyrb import AE

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T


class TestAE(TestCase):
    def test_constructor_empty(self):
        AE([20, 2], [2, 20], torch.nn.ReLU, torch.nn.ReLU, 20)

    def test_wrong_constructor(self):
        with self.assertRaises(ValueError):
            AE([20, 2], [5, 20], torch.nn.ReLU, torch.nn.ReLU, 20)

    def test_reconstruction(self):
        f = torch.nn.Softplus
        ae = AE([400, 20, 2], [2, 20, 400], f(), f(), 1e-5)
        ae.fit(snapshots)
        snapshots_ = ae.inverse_transform(ae.transform(snapshots))
        rerr = np.linalg.norm(snapshots_ - snapshots)/np.linalg.norm(snapshots)
        assert rerr < 5e-3

    def test_decode_encode(self):
        f = torch.nn.Softplus
        low_dim = 5
        ae = AE([400, low_dim], [low_dim, 400], f(), f(), 20)
        ae.fit(snapshots)
        reduced_snapshots = ae.transform(snapshots)
        assert reduced_snapshots.shape[0] == low_dim
        expanded_snapshots = ae.inverse_transform(reduced_snapshots)
        assert expanded_snapshots.shape[0] == snapshots.shape[0]

    def test_optimizer(self):
        f = torch.nn.Softplus
        low_dim = 5
        ae = AE([400, low_dim], [low_dim, 400], f(), f(), 20)
        ae.fit(snapshots)
        assert ae.optimizer == torch.optim.Adam
        ae = AE([200, 100, 10], [10, 100, 200], torch.nn.Tanh(), torch.nn.Tanh(), 10)
        ae.fit(snapshots)
        assert ae.optimizer == torch.optim.Adam

    def test_optimizer_doublefit(self):
        f = torch.nn.Softplus
        low_dim = 5
        ae = AE([400, low_dim], [low_dim, 400], f(), f(), 20)
        ae.fit(snapshots)
        ae.fit(snapshots)

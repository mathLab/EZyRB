import numpy as np
import torch

from unittest import TestCase
from ezyrb import AE, POD, PODAE

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T


class TestAE(TestCase):
    def test_constructor_empty(self):
        ae = AE([20, 2], [2, 20], torch.nn.ReLU, torch.nn.ReLU, 20)
        pod = POD()
        PODAE(pod, ae)

    def test_reconstruction(self):
        f = torch.nn.Softplus
        ae = AE([20, 20, 2], [2, 20, 20], f(), f(), 1e-9)
        pod = POD(rank=4)
        podae = PODAE(pod, ae)
        podae.fit(snapshots)
        snapshots_ = podae.inverse_transform(podae.transform(snapshots))
        rerr = np.linalg.norm(snapshots_ - snapshots)/np.linalg.norm(snapshots)
        assert rerr < 5e-3

    #TODO
    # def test_decode_encode(self):
    #     f = torch.nn.Softplus
    #     low_dim = 5
    #     ae = AE([400, low_dim], [low_dim, 400], f(), f(), 20)
    #     ae.fit(snapshots)
    #     reduced_snapshots = ae.reduce(snapshots)
    #     assert reduced_snapshots.shape[0] == low_dim
    #     expanded_snapshots = ae.expand(reduced_snapshots)
    #     assert expanded_snapshots.shape[0] == snapshots.shape[0]

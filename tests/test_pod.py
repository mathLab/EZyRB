import numpy as np

from unittest import TestCase
from ezyrb import POD

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
poddb = np.load('tests/test_datasets/p_snapshots_pod.npy')


class TestPOD(TestCase):
    def test_constructor_empty(self):
        a = POD()

    def test_1(self):
        A = POD('svd').reduce(snapshots)
        assert np.allclose(A, poddb, rtol=1e-03, atol=1e-08) or np.allclose(
            A, -1 * poddb, rtol=1e-03, atol=1e-08,)

    def test_2(self):
        A = POD('correlation_matrix').reduce(snapshots)
        assert np.allclose(A, poddb, rtol=1e-03, atol=1e-08) or np.allclose(
            A, -1 * poddb, rtol=1e-03, atol=1e-08,)

    def test_3(self):
        A = POD('randomized_svd').reduce(snapshots)
        assert np.allclose(A, poddb, rtol=1e-03, atol=1e-08) or np.allclose(
            A, -1 * poddb, rtol=1e-03, atol=1e-08,)

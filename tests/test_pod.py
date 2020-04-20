import numpy as np

from unittest import TestCase
from ezyrb import POD

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
poddb = np.load('tests/test_datasets/p_snapshots_pod.npy')
modes = np.load('tests/test_datasets/p_snapshots_pod_modes.npy')


class TestPOD(TestCase):
    def test_constructor_empty(self):
        a = POD()

    def test_numpysvd(self):
        A = POD('svd').reduce(snapshots)
        assert np.allclose(A, poddb, rtol=1e-03, atol=1e-08) or np.allclose(
            A,
            -1 * poddb,
            rtol=1e-03,
            atol=1e-08,
        )

    def test_correlation_matirix(self):
        A = POD('correlation_matrix').reduce(snapshots)
        assert np.allclose(A, poddb, rtol=1e-03, atol=1e-08) or np.allclose(
            A,
            -1 * poddb,
            rtol=1e-03,
            atol=1e-08,
        )

    def test_correlation_matirix_savemem(self):
        A = POD('correlation_matrix', save_memory=True).reduce(snapshots)
        assert np.allclose(A, poddb, rtol=1e-03, atol=1e-08) or np.allclose(
            A,
            -1 * poddb,
            rtol=1e-03,
            atol=1e-08,
        )

    def test_randomized_svd(self):
        A = POD('randomized_svd').reduce(snapshots)
        np.testing.assert_allclose(np.absolute(A),
                                   np.absolute(poddb),
                                   rtol=1e-03,
                                   atol=1e-08)

    def test_singlular_values(self):
        a = POD('svd')
        a.reduce(snapshots)
        np.testing.assert_allclose(
            a.singular_values,
            np.array([887.15704, 183.2508, 84.11757, 26.40448]),
            rtol=1e-6,
            atol=1e-8)

    def test_modes(self):
        a = POD('svd')
        a.reduce(snapshots)
        np.testing.assert_allclose(a.modes, modes)

    def test_truncation_01(self):
        a = POD(method='svd', rank=0)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_02(self):
        a = POD(method='randomized_svd', rank=0)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_03(self):
        a = POD(method='correlation_matrix', rank=0)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 2

    def test_truncation_04(self):
        a = POD(method='svd', rank=3)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 3

    def test_truncation_05(self):
        a = POD(method='randomized_svd', rank=3)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 3

    def test_truncation_06(self):
        a = POD(method='correlation_matrix', rank=4)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 4

    def test_truncation_07(self):
        a = POD(method='svd', rank=0.8)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_08(self):
        a = POD(method='randomized_svd', rank=0.995)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 3

    def test_truncation_09(self):
        a = POD(method='correlation_matrix', rank=0.9999)
        a.reduce(snapshots)
        assert a.singular_values.shape[0] == 2

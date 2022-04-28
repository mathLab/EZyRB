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
        pod = POD('svd').fit(snapshots)
        snapshots_ = pod.inverse_transform(pod.transform(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=4)

    def test_correlation_matrix(self):
        pod = POD('correlation_matrix').fit(snapshots)
        snapshots_ = pod.inverse_transform(pod.transform(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=3)

    def test_correlation_matirix_savemem(self):
        pod = POD('correlation_matrix', save_memory=True).fit(snapshots)
        snapshots_ = pod.inverse_transform(pod.transform(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=3)

    def test_randomized_svd(self):
        pod = POD('randomized_svd', save_memory=False).fit(snapshots)
        snapshots_ = pod.inverse_transform(pod.transform(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=4)

    def test_numpysvd_old(self):
        pod = POD('svd').fit(snapshots)
        snapshots_ = pod.expand(pod.reduce(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=4)

    def test_correlation_matrix_old(self):
        pod = POD('correlation_matrix').fit(snapshots)
        snapshots_ = pod.expand(pod.reduce(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=3)

    def test_correlation_matirix_savemem_old(self):
        pod = POD('correlation_matrix', save_memory=True).fit(snapshots)
        snapshots_ = pod.expand(pod.reduce(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=3)

    def test_randomized_svd_old(self):
        pod = POD('randomized_svd', save_memory=False).fit(snapshots)
        snapshots_ = pod.expand(pod.reduce(snapshots))
        np.testing.assert_array_almost_equal(snapshots, snapshots_, decimal=4)

    def test_singlular_values(self):
        a = POD('svd').fit(snapshots)
        np.testing.assert_allclose(
            a.singular_values,
            np.array([887.15704, 183.2508, 84.11757, 26.40448]),
            rtol=1e-6,
            atol=1e-8)

    def test_modes(self):
        a = POD('svd')
        a.fit(snapshots)
        np.testing.assert_allclose(a.modes, modes)

    def test_truncation_01(self):
        a = POD(method='svd', rank=0)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_02(self):
        a = POD(method='randomized_svd', rank=0)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_03(self):
        a = POD(method='correlation_matrix', rank=0)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_04(self):
        a = POD(method='svd', rank=3)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 3

    def test_truncation_05(self):
        a = POD(method='randomized_svd', rank=3)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 3

    def test_truncation_06(self):
        a = POD(method='correlation_matrix', rank=4)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 4

    def test_truncation_07(self):
        a = POD(method='svd', rank=0.8)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 1

    def test_truncation_08(self):
        a = POD(method='randomized_svd', rank=0.995)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 3

    def test_truncation_09(self):
        a = POD(method='correlation_matrix', rank=0.9999)
        a.fit(snapshots)
        assert a.singular_values.shape[0] == 4

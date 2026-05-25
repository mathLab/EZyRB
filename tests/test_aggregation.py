import copy
import unittest
import numpy as np
from unittest import TestCase
from ezyrb import Database, RBF
from ezyrb.approximation.linear import Linear
from ezyrb.reduction.pod import POD
from ezyrb.reducedordermodel import ReducedOrderModel as ROM
from ezyrb.reducedordermodel import MultiReducedOrderModel as MROM
from ezyrb.plugin.aggregation import Aggregation
from ezyrb.plugin.database_splitter import DatabaseSplitter

class MockROM:
    validation_full_database = None

    def __init__(self, db):
        self.validation_full_database = db

    def predict(self, db):
        return db

class MockMROM:
    train_full_database = None
    validation_full_database = None
    predict_full_database = None
    multi_predict_database = None
    weights_predict = None

    def __init__(self, db, n_roms=2):
        self.roms = {f'rom{i}': MockROM(db) for i in range(n_roms)}
        self.train_full_database = db
        self.validation_full_database = db
        self.predict_full_database = db
        self.multi_predict_database = {f'rom{i}': db for i in range(n_roms)}
        self.weights_predict = {}


def _make_unit_db():
    space = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    params = np.array([[0.5], [1.5]])
    snaps = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    return Database(params, snaps, space=space)


def _make_integration_db(n_params=5, n_space=3):
    mu = np.linspace(0.5, 3.0, n_params)
    x = np.linspace(0, 2 * np.pi, n_space)
    snaps = np.array([np.sin(m * x) for m in mu])
    space = x.reshape(-1, 1)
    return Database(mu.reshape(-1, 1), snaps, space=space)

def _relative_error(predicted, actual):
    norms = np.linalg.norm(actual, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return np.mean(np.linalg.norm(predicted - actual, axis=1) / norms)

class TestAggregation(TestCase):

    def setUp(self):
        self.db = _make_unit_db()

    def test_constructor_default_fit_function_is_none(self):
        agg = Aggregation()
        self.assertIsNone(agg.fit_function)

    def test_constructor_default_predict_function_is_linear(self):
        agg = Aggregation()
        self.assertIsInstance(agg.predict_function, Linear)

    def test_constructor_custom_arguments(self):
        agg = Aggregation(fit_function=RBF(), predict_function=RBF())
        self.assertIsInstance(agg.fit_function, RBF)
        self.assertIsInstance(agg.predict_function, RBF)


    def test_check_sum_gaussians_partial_zeros(self):
        agg = Aggregation()
        mrom = MockMROM(self.db, n_roms=2)
        gaussians = np.array([[0.0, 0.8], [0.0, 0.2]])
        res = agg._check_sum_gaussians(mrom, gaussians.sum(axis=0), gaussians.copy())
        np.testing.assert_array_equal(res[:, 0], [0.5, 0.5])
        np.testing.assert_array_equal(res[:, 1], [0.8, 0.2])

    def test_check_sum_gaussians_no_zeros_unchanged(self):
        agg = Aggregation()
        mrom = MockMROM(self.db, n_roms=2)
        gaussians = np.array([[0.3, 0.7], [0.6, 0.3]])
        original = gaussians.copy()
        res = agg._check_sum_gaussians(mrom, gaussians.sum(axis=0), gaussians.copy())
        np.testing.assert_array_equal(res, original)

    def test_check_sum_gaussians_all_zeros(self):
        agg = Aggregation()
        mrom = MockMROM(self.db, n_roms=2)
        gaussians = np.zeros((2, 3))
        res = agg._check_sum_gaussians(mrom, gaussians.sum(axis=0), gaussians.copy())
        np.testing.assert_array_equal(res, np.full((2, 3), 0.5))

    def test_check_sum_gaussians_equal_weight_matches_n_roms(self):
        n_roms = 4
        agg = Aggregation()
        mrom = MockMROM(self.db, n_roms=n_roms)
        gaussians = np.zeros((n_roms, 2))
        res = agg._check_sum_gaussians(mrom, gaussians.sum(axis=0), gaussians.copy())
        np.testing.assert_array_almost_equal(res, np.full((n_roms, 2), 1.0 / n_roms))


    def test_compute_validation_weights_perfect_prediction_values(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation()
        g = agg._compute_validation_weights(mrom, sigma=1.0, normalized=False)
        np.testing.assert_array_almost_equal(g, np.ones_like(g))

    def test_compute_validation_weights_normalized_sums_to_one(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation()
        g = agg._compute_validation_weights(mrom, sigma=1.0, normalized=True)
        np.testing.assert_array_almost_equal(g.sum(axis=0), np.ones_like(g[0]))

    def test_compute_validation_weights_shape(self):
        mrom = MockMROM(self.db, n_roms=3)
        agg = Aggregation()
        g = agg._compute_validation_weights(mrom, sigma=1.0)
        self.assertEqual(g.shape[0], 3)

    def test_compute_validation_weights_sigma_effect(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation()
        g_large = agg._compute_validation_weights(mrom, sigma=1e6, normalized=False)
        g_small = agg._compute_validation_weights(mrom, sigma=1e-6, normalized=False)
        np.testing.assert_array_almost_equal(g_large, np.ones_like(g_large))
        np.testing.assert_array_almost_equal(g_small, np.ones_like(g_small))


    def test_optimize_sigma_returns_finite_value(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation()
        sigma = agg._optimize_sigma(mrom)
        self.assertTrue(np.isfinite(sigma).all())

    def test_optimize_sigma_within_default_range(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation()
        sigma = agg._optimize_sigma(mrom)
        self.assertGreaterEqual(float(np.squeeze(sigma)), 1e-5)
        self.assertLessEqual(float(np.squeeze(sigma)), 1e-2)

    def test_aggregation_no_fit_function(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation(fit_function=None, predict_function=RBF())
        agg.fit_postprocessing(mrom)
        agg.predict_postprocessing(mrom)
        self.assertIsNotNone(mrom.predict_full_database)
        self.assertEqual(len(agg.predict_functions), 2)

    def test_aggregation_with_fit_function(self):
        mrom = MockMROM(self.db, n_roms=1)
        agg = Aggregation(fit_function=RBF(), predict_function=RBF())
        agg.fit_postprocessing(mrom)
        agg.predict_postprocessing(mrom)
        self.assertIsNotNone(mrom.predict_full_database)

    def test_nan_handling_in_weights(self):
        mrom = MockMROM(self.db, n_roms=2)
        agg = Aggregation(fit_function=None, predict_function=RBF())
        agg._compute_validation_weights = (
            lambda mrom, sigma, normalized=False: np.full((2, 2, 3), np.nan)
        )
        agg._optimize_sigma = lambda mrom: 1e-3
        agg.fit_postprocessing(mrom)
        self.assertEqual(len(agg.predict_functions), 2)


class TestAggregationIntegration(TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.db = _make_integration_db(n_params=5, n_space=3)

    def _make_splitter(self, seed=0):
        return DatabaseSplitter(
            train=2, test=0, validation=2, predict=1, seed=seed
        )

    def _build_and_fit_mrom(self, agg, seed=0):
        splitter = self._make_splitter(seed=seed)
        rom1 = ROM(self.db, POD(rank=1), RBF())
        rom2 = ROM(self.db, POD(rank=1), Linear())
        agg._optimize_sigma = lambda mrom: 1e-3
        mrom = MROM(
            {'rbf': rom1, 'lin': rom2},
            plugins=[splitter, agg],
            rom_plugin=splitter,
        )
        mrom.fit()
        return mrom

    def test_fit_does_not_raise(self):
        agg = Aggregation(fit_function=None, predict_function=RBF())
        self._build_and_fit_mrom(agg)

    def test_fit_regression_path_does_not_raise(self):
        splitter = self._make_splitter()
        rom1 = ROM(self.db, POD(rank=1), RBF())
        agg = Aggregation(fit_function=RBF(), predict_function=RBF())
        mrom = MROM({'rbf': rom1}, plugins=[splitter, agg], rom_plugin=splitter)
        mrom.fit()

    def test_predict_returns_database_instance(self):
        agg = Aggregation(fit_function=None, predict_function=RBF())
        mrom = self._build_and_fit_mrom(agg)
        mrom.predict(mrom.predict_full_database)
        self.assertIsInstance(mrom.predict_full_database, Database)

    def test_predict_snapshot_shape(self):
        agg = Aggregation(fit_function=None, predict_function=RBF())
        mrom = self._build_and_fit_mrom(agg)
        mrom.predict(mrom.predict_full_database)
        self.assertEqual(mrom.predict_full_database.snapshots_matrix.shape[1], 3)

    def test_predict_functions_count_matches_n_roms(self):
        agg = Aggregation(fit_function=None, predict_function=RBF())
        self._build_and_fit_mrom(agg)
        self.assertEqual(len(agg.predict_functions), 2)

    def test_weights_are_finite(self):
        agg = Aggregation(fit_function=None, predict_function=RBF())
        mrom = self._build_and_fit_mrom(agg)
        mrom.predict(mrom.predict_full_database)
        for key, w in mrom.weights_predict.items():
            self.assertTrue(np.isfinite(w).all(),
                            msg=f"Non-finite weight for ROM '{key}'")

    def test_weights_sum_to_one(self):
        agg = Aggregation(fit_function=None, predict_function=RBF())
        mrom = self._build_and_fit_mrom(agg)
        mrom.predict(mrom.predict_full_database)
        weight_sum = np.sum(list(mrom.weights_predict.values()), axis=0)
        np.testing.assert_array_almost_equal(
            weight_sum, np.ones_like(weight_sum), decimal=5
        )

    def test_fit_reproducible_with_same_seed(self):
        agg1 = Aggregation(fit_function=None, predict_function=RBF())
        agg2 = Aggregation(fit_function=None, predict_function=RBF())
        mrom1 = self._build_and_fit_mrom(agg1, seed=7)
        mrom2 = self._build_and_fit_mrom(agg2, seed=7)

        pred_db1 = copy.deepcopy(mrom1.predict_full_database)
        pred_db2 = copy.deepcopy(mrom2.predict_full_database)
        mrom1.predict(pred_db1)
        mrom2.predict(pred_db2)

        np.testing.assert_array_almost_equal(
            mrom1.predict_full_database.snapshots_matrix,
            mrom2.predict_full_database.snapshots_matrix,
            decimal=10,
        )

    def test_fit_different_seeds_produce_different_predictions(self):
        agg1 = Aggregation(fit_function=None, predict_function=RBF())
        agg2 = Aggregation(fit_function=None, predict_function=RBF())
        mrom1 = self._build_and_fit_mrom(agg1, seed=0)
        mrom2 = self._build_and_fit_mrom(agg2, seed=99)

        pred_db1 = copy.deepcopy(mrom1.predict_full_database)
        pred_db2 = copy.deepcopy(mrom2.predict_full_database)
        mrom1.predict(pred_db1)
        mrom2.predict(pred_db2)

        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(
                mrom1.predict_full_database.snapshots_matrix,
                mrom2.predict_full_database.snapshots_matrix,
                decimal=10,
            )

if __name__ == '__main__':
    unittest.main()
import numpy as np
import pytest
from unittest import TestCase

from ezyrb import POD, GPR, RBF, Database
from ezyrb import KNeighborsRegressor, RadiusNeighborsRegressor, Linear
from ezyrb.reducedordermodel import ReducedOrderModel as ROM
from ezyrb.reducedordermodel import MultiReducedOrderModel as MROM

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
pred_sol_tst = np.load('tests/test_datasets/p_predsol.npy').T
pred_sol_gpr = np.load('tests/test_datasets/p_predsol_gpr.npy').T
param = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])

def _make_db():
    return Database(param, snapshots.T)

def _make_rom(rank=None, approx=None):
    pod = POD() if rank is None else POD(rank=rank)
    return ROM(_make_db(), pod, approx or RBF()).fit()

def _make_mrom():
    """Minimal fitted MROM: one db, one POD, one RBF."""
    return MROM({"p": _make_db()}, {"pod": POD()}, {"rbf": RBF()}).fit()

class TestReducedOrderModel(TestCase):
    def test_constructor(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf)

    def test_fit(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()

    def test_save(self):
        fname = 'ezyrb.tmp'
        pod = POD(rank=2)
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf)
        rom.fit()
        rom.save(fname)

    def test_load(self):
        fname = 'ezyrb.tmp'
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf)
        rom.fit()
        rom.save(fname)
        new_rom = ROM.load(fname)
        new_param = [-0.293344, -0.23120537]
        np.testing.assert_array_almost_equal(
            rom.predict(new_param),
            new_rom.predict(new_param)
        )

    def test_load2(self):
        fname = 'ezyrb2.tmp'
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf)
        rom.fit()
        rom.save(fname, save_db=False)
        new_rom = ROM.load(fname)
        new_param = [-0.293344, -0.23120537]
        np.testing.assert_array_almost_equal(
            rom.predict(new_param),
            new_rom.predict(new_param)
        )

    def test_predict_01(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()
        pred_sol = rom.predict([-0.293344, -0.23120537])
        np.testing.assert_allclose(
            pred_sol.flatten(),
            pred_sol_tst, rtol=1e-4, atol=1e-5)

    def test_predict_02(self):
        np.random.seed(117)
        pod = POD(method='svd', rank=4)
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict([-.45, -.45])
        np.testing.assert_allclose(
            pred_sol.flatten(),
            pred_sol_gpr, rtol=1e-4, atol=1e-5)

    def test_predict_03(self):
        pod = POD(method='svd', rank=3)
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict(db.parameters_matrix[2])
        assert pred_sol[0].shape == db.snapshots_matrix[0].shape
        pred_db = rom.predict(db)
        assert pred_db.snapshots_matrix.shape == db.snapshots_matrix.shape

    def test_predict_04(self):
        pod = POD(method='svd', rank=3)
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict(db.parameters_matrix)
        assert pred_sol.shape == db.snapshots_matrix.shape

    # def test_predict_scaler_01(self):
    #     from sklearn.preprocessing import StandardScaler
    #     scaler = StandardScaler()
    #     pod = POD()
    #     rbf = RBF()
    #     db = Database(param, snapshots.T, scaler_snapshots=scaler)
    #     rom = ROM(db, pod, rbf).fit()
    #     pred_sol = rom.predict(db.parameters[0])
    #     np.testing.assert_allclose(pred_sol, db._snapshots[0], rtol=1e-4, atol=1e-5)
    #     pred_sol = rom.predict(db.parameters[0:2])
    #     np.testing.assert_allclose(pred_sol, db._snapshots[0:2], rtol=1e-4, atol=1e-5)

    # def test_predict_scaler_02(self):
    #     from sklearn.preprocessing import StandardScaler
    #     scaler_p = StandardScaler()
    #     scaler_s = StandardScaler()
    #     pod = POD()
    #     rbf = RBF()
    #     db = Database(param, snapshots.T, scaler_parameters=scaler_p, scaler_snapshots=scaler_s)
    #     rom = ROM(db, pod, rbf).fit()
    #     pred_sol = rom.predict(db._parameters[0])
    #     np.testing.assert_allclose(pred_sol, db._snapshots[0], rtol=1e-4, atol=1e-5)
    #     pred_sol = rom.predict(db._parameters[0:2])
    #     np.testing.assert_allclose(pred_sol, db._snapshots[0:2], rtol=1e-4, atol=1e-5)

    # def test_predict_scaling_coeffs(self):
    #     from sklearn.preprocessing import StandardScaler
    #     scaler = StandardScaler()
    #     pod = POD()
    #     rbf = RBF()
    #     db = Database(param, snapshots.T)
    #     rom = ROM(db, pod, rbf, scaler).fit()
    #     pred_sol = rom.predict(db._parameters[0])
    #     np.testing.assert_allclose(pred_sol, db._snapshots[0], rtol=1e-4, atol=1e-5)
    #     pred_sol = rom.predict(db._parameters[0:2])
    #     np.testing.assert_allclose(pred_sol, db._snapshots[0:2], rtol=1e-4, atol=1e-5)

    def test_test_error(self):
        pod = POD(method='svd', rank=-1)
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()
        error = rom.test_error(db)
        np.testing.assert_almost_equal(error, 0, decimal=6)

    def test_kfold_cv_error_01(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        n_splits = len(db)
        rom = ROM(db, pod, rbf)
        err_kfold = rom.kfold_cv_error(n_splits=n_splits)
        err_loo = rom.loo_error()
        np.testing.assert_allclose(err_kfold, err_loo)

    def test_loo_error_01(self):
        pod = POD()
        rbf = RBF()
        gpr = GPR()
        rnr = RadiusNeighborsRegressor()
        knr = KNeighborsRegressor(n_neighbors=1)
        lin = Linear(fill_value=0)
        db = Database(param, snapshots.T)
        exact_len = len(db)
        approximations = [rbf, gpr, knr, rnr]#, lin]
        roms = [ROM(db, pod, app) for app in approximations]
        len_errors = [len(rom.loo_error()) for rom in roms]
        np.testing.assert_allclose(len_errors, exact_len)

    def test_loo_error_02(self):
        pod = POD()
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr)
        err = rom.loo_error()
        np.testing.assert_allclose(
            err[0],
            np.array(0.595857),
            rtol=1e-3)

    def test_loo_error_singular_values(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()
        valid_svalues = rom.reduction.singular_values
        rom.loo_error()
        np.testing.assert_allclose(valid_svalues, rom.reduction.singular_values)

    def test_multi_db(self):
        pod = POD()
        pod2 = POD(rank=1)
        gpr = GPR()
        db1 = Database(param, snapshots.T)
        rom = MROM({'p': db1}, {'a': pod, 'b':pod2}, gpr).fit()
        pred = rom.predict([-.5, -.5])
        assert isinstance(pred, dict)
        assert len(pred) == 2

def test_invariant_pod():
    pod = POD()

    rbf = RBF()
    gpr = GPR()
    rnr = RadiusNeighborsRegressor()
    knr = KNeighborsRegressor(n_neighbors=1)
    lin = Linear(fill_value=0)
    db = Database(param, snapshots.T)

    modal_coeffs = []
    for approx in [rbf, gpr, knr, rnr, lin]:
        rom = ROM(db, pod, approx).fit()
        coeff = rom.reduction.transform(db.snapshots_matrix.T)
        modal_coeffs.append(coeff)

    for i in range(1, len(modal_coeffs)):
        np.testing.assert_allclose(
            modal_coeffs[0],
            modal_coeffs[i],
            rtol=1e-5,
            atol=1e-8
        )

"""
    def test_optimal_mu(self):
        pod = POD()
        rbf = RBF()
        gpr = GPR()
        rnr = RadiusNeighborsRegressor()
        knr = KNeighborsRegressor(n_neighbors=1)
        lin = Linear()
        db = Database(param, snapshots.T)
        exact_len = param.shape[1]
        approximations = [rbf, gpr, knr, rnr, lin]
        for k in [1, 2]:
            roms = [ROM(db, pod, app).fit() for app in approximations]
            len_opt_mu = [rom.optimal_mu(k=k).shape[1] for rom in roms]
            np.testing.assert_allclose(len_opt_mu, exact_len)
            len_k = [rom.optimal_mu(k=k).shape[0] for rom in roms]
            np.testing.assert_allclose(len_k, k)
"""

class TestROMConstructorPlugins(TestCase):

    def test_constructor_with_non_empty_plugins(self):
        class _P:
            def fit_preprocessing(self, rom): pass

        plugin = _P()
        rom = ROM(_make_db(), POD(), RBF(), plugins=[plugin])
        self.assertIn(plugin, rom.plugins)

    def test_execute_plugins_calls_correct_hooks(self):
        stages = []

        class _Tracker:
            def fit_preprocessing(self, rom):
                stages.append("fit_preprocessing")
            def fit_postprocessing(self, rom):
                stages.append("fit_postprocessing")

        ROM(_make_db(), POD(), RBF(), plugins=[_Tracker()]).fit()
        self.assertIn("fit_preprocessing", stages)
        self.assertIn("fit_postprocessing", stages)

    def test_execute_plugins_skips_missing_hooks(self):
        class _NoHooks: pass
        ROM(_make_db(), POD(), RBF(), plugins=[_NoHooks()]).fit()


class TestROMPropertySetters(TestCase):

    def test_database_setter_wrong_type(self):
        with self.assertRaises(TypeError):
            ROM(_make_db(), POD(), RBF()).database = "bad"

    def test_reduction_setter_wrong_type(self):
        with self.assertRaises(TypeError):
            ROM(_make_db(), POD(), RBF()).reduction = "bad"

    def test_approximation_setter_wrong_type(self):
        with self.assertRaises(TypeError):
            ROM(_make_db(), POD(), RBF()).approximation = "bad"


class TestROMPropertyDeleters(TestCase):

    def test_database_deleter(self):
        rom = _make_rom()
        del rom.database
        self.assertFalse(hasattr(rom, "_database"))

    def test_reduction_deleter(self):
        rom = _make_rom()
        del rom.reduction
        self.assertFalse(hasattr(rom, "_reduction"))

    def test_approximation_deleter(self):
        rom = _make_rom()
        del rom.approximation
        self.assertFalse(hasattr(rom, "_approximation"))


class TestROMCountProperties(TestCase):

    def test_n_database_is_one(self):
        self.assertEqual(_make_rom().n_database, 1)

    def test_n_reduction_is_one(self):
        self.assertEqual(_make_rom().n_reduction, 1)


    def test_n_approximation_count(self):
        self.assertEqual(_make_rom().n_approximation, 1)


class TestROMFitRuntimeErrors(TestCase):

    def test_fit_reduction_raises_when_attr_deleted(self):
        rom = ROM(_make_db(), POD(), RBF())
        del rom.train_full_database
        with self.assertRaises(RuntimeError):
            rom.fit_reduction()

    def test_fit_approximation_raises_when_attr_deleted(self):
        rom = ROM(_make_db(), POD(), RBF())
        del rom.train_reduced_database
        with self.assertRaises(RuntimeError):
            rom.fit_approximation()


class TestROMPredict(TestCase):

    def test_predict_database_input_returns_database(self):
        rom = _make_rom()
        result = rom.predict(_make_db())
        self.assertIsInstance(result, Database)

    def test_predict_tuple_input(self):
        result = _make_rom().predict((-0.293344, -0.23120537))
        self.assertEqual(result.shape[0], 1)

    def test_predict_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _make_rom().predict({"bad": "input"})


class TestROMClean(TestCase):

    def test_clean_sets_all_internal_dbs_to_none(self):
        rom = _make_rom()
        rom.clean()
        for attr in (
            "train_full_database", "train_reduced_database",
            "predict_full_database", "predict_reduced_database",
            "test_full_database", "test_reduced_database",
            "validation_full_database", "validation_reduced_database",
        ):
            self.assertIsNone(getattr(rom, attr), f"{attr} should be None")


class TestROMReduceDatabase(TestCase):

    def test_returns_database_with_same_row_count(self):
        rom = _make_rom()
        db = _make_db()
        reduced = rom._reduce_database(db)
        self.assertIsInstance(reduced, Database)
        self.assertEqual(reduced.snapshots_matrix.shape[0],
                         db.snapshots_matrix.shape[0])


class TestROMSavePartialFlags(TestCase):

    def test_save_without_reduction(self):
        rom = _make_rom()
        rom.save("/tmp/rom_no_reduction.pkl", save_reduction=False)
        self.assertFalse(hasattr(ROM.load("/tmp/rom_no_reduction.pkl"), "_reduction"))

    def test_save_without_approx(self):
        rom = _make_rom()
        rom.save("/tmp/rom_no_approx.pkl", save_approx=False)
        self.assertFalse(hasattr(ROM.load("/tmp/rom_no_approx.pkl"), "_approximation"))

    def test_save_all_flags_false(self):
        rom = _make_rom()
        rom.save("/tmp/rom_skeleton.pkl",
                 save_db=False, save_reduction=False, save_approx=False)
        loaded = ROM.load("/tmp/rom_skeleton.pkl")
        for attr in ("_database", "_reduction", "_approximation"):
            self.assertFalse(hasattr(loaded, attr))


class TestROMTestErrorAbsolute(TestCase):

    def test_absolute_error_is_non_negative(self):
        err = _make_rom(rank=-1).test_error(_make_db(), relative=False)
        self.assertGreaterEqual(err, 0.0)

    def test_relative_and_absolute_both_non_negative(self):
        rom = _make_rom()
        db = _make_db()
        self.assertGreaterEqual(rom.test_error(db, relative=True), 0.0)
        self.assertGreaterEqual(rom.test_error(db, relative=False), 0.0)


class TestROMKfoldCvError(TestCase):

    def test_length_equals_n_splits(self):
        errors = ROM(_make_db(), POD(), GPR()).kfold_cv_error(n_splits=2)
        self.assertEqual(len(errors), 2)

    def test_all_errors_non_negative(self):
        errors = ROM(_make_db(), POD(), GPR()).kfold_cv_error(n_splits=2)
        self.assertTrue(np.all(errors >= 0))

    def test_absolute_mode(self):
        errors = ROM(_make_db(), POD(), GPR()).kfold_cv_error(
            n_splits=2, relative=False)
        self.assertEqual(len(errors), 2)
        self.assertTrue(np.all(errors >= 0))


class TestROMOptimalMu(TestCase):

    def test_k1_returns_one_point(self):
        opt = _make_rom().optimal_mu(k=1)
        self.assertEqual(opt.shape, (1, param.shape[1]))

    def test_k2_returns_two_points(self):
        opt = _make_rom().optimal_mu(k=2)
        self.assertEqual(opt.shape[0], 2)

    def test_precomputed_error_gives_same_result(self):
        rom = _make_rom()
        error = rom.loo_error()
        np.testing.assert_allclose(rom.optimal_mu(k=1),
                                   rom.optimal_mu(error=error, k=1))

    def test_simplex_volume_positive(self):
        vol = _make_rom()._simplex_volume(param[:3])
        self.assertGreater(vol, 0.0)


class TestROMReductionError(TestCase):

    def test_default_relative(self):
        err = _make_rom().reduction_error()
        self.assertEqual(err.shape, (1,))
        self.assertGreaterEqual(err[0], 0.0)

    def test_absolute_branch(self):
        err = _make_rom().reduction_error(relative=False)
        self.assertGreaterEqual(err[0], 0.0)

    def test_explicit_db(self):
        err = _make_rom().reduction_error(db=_make_db())
        self.assertEqual(err.shape, (1,))

    def test_full_rank_error_is_small(self):
        err = _make_rom(rank=-1).reduction_error()
        self.assertLess(err[0], 1e-4)


class TestROMApproximationError(TestCase):

    def test_default_relative(self):
        err = _make_rom().approximation_error()
        self.assertEqual(err.shape, (1,))
        self.assertGreaterEqual(err[0], 0.0)

    def test_absolute_branch(self):
        err = _make_rom().approximation_error(relative=False)
        self.assertGreaterEqual(err[0], 0.0)

    def test_explicit_db(self):
        err = _make_rom().approximation_error(db=_make_db())
        self.assertEqual(err.shape, (1,))

    def test_interpolatory_method_near_zero(self):
        np.testing.assert_almost_equal(
            _make_rom().approximation_error()[0], 0.0, decimal=4)


class TestMROMInit(TestCase):

    def test_3arg_cartesian_product(self):
        mrom = MROM({"p": _make_db()},
                    {"pod": POD(), "pod2": POD(rank=1)},
                    {"rbf": RBF()})
        self.assertEqual(len(mrom.roms), 2)

    def test_init_roms_dict_only(self):
        db = _make_db()
        roms = {"a": ROM(db, POD(), RBF()).fit(),
                "b": ROM(db, POD(rank=1), RBF()).fit()}
        mrom = MROM(roms)
        self.assertIn("a", mrom.roms)
        self.assertIn("a", mrom.database)

    def test_init_database_and_roms_dict(self):
        db = _make_db()
        mrom = MROM(db, {"a": ROM(db, POD(), RBF()).fit()})
        self.assertIn(0, mrom.database)

    def test_rom_plugin_appended_to_each_rom(self):
        class _P: pass
        plugin = _P()
        mrom = MROM({"p": _make_db()}, {"pod": POD()}, {"rbf": RBF()},
                    rom_plugin=plugin)
        for rom_ in mrom.roms.values():
            self.assertIn(plugin, rom_.plugins)

    def test_global_plugins_stored_on_mrom(self):
        class _GP: pass
        gp = _GP()
        mrom = MROM({"p": _make_db()}, {"pod": POD()}, {"rbf": RBF()},
                    plugins=[gp])
        self.assertIn(gp, mrom.plugins)


class TestMROMPropertySetters(TestCase):

    def test_database_wrong_type(self):
        with self.assertRaises(TypeError):
            _make_mrom().database = 42

    def test_database_plain_db_wraps_to_dict(self):
        mrom = _make_mrom()
        mrom.database = _make_db()
        self.assertIn(0, mrom._database)

    def test_database_dict_stored_as_is(self):
        d = {"x": _make_db()}
        mrom = _make_mrom()
        mrom.database = d
        self.assertIs(mrom._database, d)

    def test_reduction_wrong_type(self):
        with self.assertRaises(TypeError):
            _make_mrom().reduction = "bad"

    def test_reduction_plain_wraps_to_dict(self):
        mrom = _make_mrom()
        mrom.reduction = POD()
        self.assertIn(0, mrom._reduction)

    def test_approximation_wrong_type(self):
        with self.assertRaises(TypeError):
            _make_mrom().approximation = 3.14

    def test_approximation_plain_wraps_to_dict(self):
        mrom = _make_mrom()
        mrom.approximation = RBF()
        self.assertIn(0, mrom._approximation)


class TestMROMPropertyDeleters(TestCase):

    def test_database_deleter(self):
        mrom = _make_mrom()
        del mrom.database
        self.assertFalse(hasattr(mrom, "_database"))

    def test_reduction_deleter(self):
        mrom = _make_mrom()
        del mrom.reduction
        self.assertFalse(hasattr(mrom, "_reduction"))


class TestMROMCountProperties(TestCase):

    def test_n_database(self):
        mrom = _make_mrom()
        self.assertEqual(mrom.n_database, len(mrom.database))

    def test_n_reduction(self):
        mrom = _make_mrom()
        self.assertEqual(mrom.n_reduction, len(mrom.reduction))

    def test_n_approximation(self):
        mrom = _make_mrom()
        self.assertEqual(mrom.n_approximation, len(mrom.approximation))


class TestMROMPredict(TestCase):

    def test_list_input_returns_dict_of_arrays(self):
        result = _make_mrom().predict([-.5, -.5])
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, np.ndarray)

    def test_database_input_returns_dict_of_databases(self):
        result = _make_mrom().predict(_make_db())
        self.assertIsInstance(result, dict)
        for v in result.values():
            self.assertIsInstance(v, Database)

    def test_invalid_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            _make_mrom().predict({"bad": "input"})

    def test_none_with_no_prior_db_raises_runtime_error(self):
        mrom = _make_mrom()
        mrom.predict_full_database = None
        with self.assertRaises(RuntimeError):
            mrom.predict(None)



class TestMROMFitIdempotent(TestCase):

    def test_second_fit_is_no_op_for_fitted_roms(self):
        mrom = _make_mrom()
        key = list(mrom.roms.keys())[0]
        existing = mrom.roms[key].train_reduced_database
        mrom.fit()
        self.assertIs(mrom.roms[key].train_reduced_database, existing)



class TestMROMSaveLoad(TestCase):

    def test_roundtrip_predictions_match(self):
        mrom = _make_mrom()
        mrom.save("/tmp/mrom_roundtrip.pkl")
        loaded = MROM.load("/tmp/mrom_roundtrip.pkl")
        p = [-.3, -.3]
        for k in mrom.predict(p):
            np.testing.assert_allclose(
                mrom.predict(p)[k], loaded.predict(p)[k], rtol=1e-5)

    def test_save_without_db(self):
        mrom = _make_mrom()
        mrom.save("/tmp/mrom_no_db.pkl", save_db=False)
        self.assertFalse(hasattr(MROM.load("/tmp/mrom_no_db.pkl"), "_database"))

    def test_save_without_reduction(self):
        mrom = _make_mrom()
        mrom.save("/tmp/mrom_no_red.pkl", save_reduction=False)
        self.assertFalse(hasattr(MROM.load("/tmp/mrom_no_red.pkl"), "_reduction"))

    def test_save_without_approx(self):
        _make_mrom().save("/tmp/mrom_no_approx.pkl", save_approx=False)


class TestMROMTestError(TestCase):

    def test_relative(self):
        _make_mrom().test_error(_make_db(), relative=True)

    def test_absolute(self):
        _make_mrom().test_error(_make_db(), relative=False)


class TestMROMKfoldCvError(TestCase):

    def test_kfold_cv_error(self):
        MROM({"p": _make_db()}, {"pod": POD()}, {"gpr": GPR()}).kfold_cv_error(
            n_splits=2)

    def test_kfold_cv_error_absolute(self):
        MROM({"p": _make_db()}, {"pod": POD()}, {"gpr": GPR()}).kfold_cv_error(
            n_splits=2, relative=False)


class TestMROMLooError(TestCase):
    def test_loo_error(self):
        MROM({"p": _make_db()}, {"pod": POD()}, {"rbf": RBF()}).loo_error()


class TestMROMOptimalMu(TestCase):
    def test_optimal_mu_no_error(self):
        _make_mrom().optimal_mu(k=1)

    def test_optimal_mu_precomputed_error(self):
        _make_mrom().optimal_mu(error={('p', 'pod', 'rbf'):np.array([0.1, 0.5, 0.2, 0.8])}, k=1)

    def test_simplex_volume_positive(self):
        self.assertGreater(_make_mrom()._simplex_volume(param[:3]), 0.0)


class TestMROMReductionError(TestCase):

    def test_default(self):
        _make_mrom().reduction_error()

    def test_absolute(self):
        _make_mrom().reduction_error(relative=False)

    def test_explicit_db(self):
        _make_mrom().reduction_error(db=_make_db())


class TestMROMApproximationError(TestCase):

    def test_default(self):
        _make_mrom().approximation_error()

    def test_absolute(self):
        _make_mrom().approximation_error(relative=False)
        
    def test_explicit_db(self):
        _make_mrom().approximation_error(db=_make_db())

if __name__ == "__main__":
    import unittest
    unittest.main()
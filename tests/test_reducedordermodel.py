import numpy as np

from unittest import TestCase
from ezyrb import POD, GPR, RBF, Database
from ezyrb import KNeighborsRegressor, RadiusNeighborsRegressor, Linear
from ezyrb import ReducedOrderModel as ROM

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
pred_sol_tst = np.load('tests/test_datasets/p_predsol.npy').T
pred_sol_gpr = np.load('tests/test_datasets/p_predsol_gpr.npy').T
param = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


class TestReducedOrderModel(TestCase):
    def test_constructor(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf)

    def test_save(self):
        fname = 'ezyrb.tmp'
        pod = POD()
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
        np.save('tests/test_datasets/p_predsol.npy', pred_sol.T)
        np.testing.assert_allclose(pred_sol, pred_sol_tst, rtol=1e-4, atol=1e-5)

    def test_predict_02(self):
        np.random.seed(117)
        pod = POD(method='svd', rank=4)
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict([-.45, -.45])
        np.testing.assert_allclose(pred_sol, pred_sol_gpr, rtol=1e-4, atol=1e-5)

    def test_predict_03(self):
        pod = POD(method='svd', rank=3)
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict(db.parameters[2])
        assert pred_sol.shape == db.snapshots[0].shape

    def test_predict_04(self):
        pod = POD(method='svd', rank=3)
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict(db.parameters)
        assert pred_sol.shape == db.snapshots.shape

    def test_predict_scaler_01(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T, scaler_snapshots=scaler)
        rom = ROM(db, pod, rbf).fit()
        pred_sol = rom.predict(db.parameters[0])
        np.testing.assert_allclose(pred_sol, db._snapshots[0], rtol=1e-4, atol=1e-5)
        pred_sol = rom.predict(db.parameters[0:2])
        np.testing.assert_allclose(pred_sol, db._snapshots[0:2], rtol=1e-4, atol=1e-5)

    def test_predict_scaler_02(self):
        from sklearn.preprocessing import StandardScaler
        scaler_p = StandardScaler()
        scaler_s = StandardScaler()
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T, scaler_parameters=scaler_p, scaler_snapshots=scaler_s)
        rom = ROM(db, pod, rbf).fit()
        pred_sol = rom.predict(db._parameters[0])
        np.testing.assert_allclose(pred_sol, db._snapshots[0], rtol=1e-4, atol=1e-5)
        pred_sol = rom.predict(db._parameters[0:2])
        np.testing.assert_allclose(pred_sol, db._snapshots[0:2], rtol=1e-4, atol=1e-5)

    def test_predict_scaling_coeffs(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf, scaler).fit()
        pred_sol = rom.predict(db._parameters[0])
        np.testing.assert_allclose(pred_sol, db._snapshots[0], rtol=1e-4, atol=1e-5)
        pred_sol = rom.predict(db._parameters[0:2])
        np.testing.assert_allclose(pred_sol, db._snapshots[0:2], rtol=1e-4, atol=1e-5)

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
        lin = Linear()
        db = Database(param, snapshots.T)
        exact_len = len(db)
        approximations = [rbf, gpr, knr, rnr, lin]
        roms = [ROM(db, pod, app) for app in approximations]
        len_errors = [len(rom.loo_error()) for rom in roms]
        np.testing.assert_allclose(len_errors, exact_len)

    def test_loo_error_02(self):
        pod = POD()
        gpr = GPR()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, gpr)
        err = rom.loo_error(normalizer=False)
        np.testing.assert_allclose(
            err[0],
            np.array(0.639247),
            rtol=1e-3)

    def test_loo_error_singular_values(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()
        valid_svalues = rom.reduction.singular_values
        rom.loo_error()
        np.testing.assert_allclose(valid_svalues, rom.reduction.singular_values)

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


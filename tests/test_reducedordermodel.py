import numpy as np

from unittest import TestCase
from ezyrb import POD, GPR, RBF, Database
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

    def test_predict_01(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()
        pred_sol = rom.predict([-0.293344, -0.23120537])
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
        #rom = ROM(db, pod, RBF()).fit()
        #pred_sol = rom.predict([-.45, -.45])
        #print(pred_sol)
        rom = ROM(db, pod, gpr).fit()
        pred_sol = rom.predict(db.parameters[2])
        assert pred_sol.shape == db.snapshots[0].shape

    def test_loo_error(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf)
        err = rom.loo_error()
        np.testing.assert_allclose(
            err,
            np.array([421.299091, 344.571787,  48.711501, 300.490491]),
            rtol=1e-4)

    def test_optimal_mu(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T)
        rom = ROM(db, pod, rbf).fit()
        opt_mu = rom.optimal_mu()
        np.testing.assert_allclose(opt_mu, [[-0.17687147, -0.21820951]],
            rtol=1e-4)

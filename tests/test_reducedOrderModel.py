import numpy as np

from unittest import TestCase
from ezyrb import POD, RBF, Database, Scale
from ezyrb import ReducedOrderModel as ROM


snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
pred_sol_tst = np.load('tests/test_datasets/p_predsol.npy').T
param = np.array([
    [-.5, -.5],
    [ .5, -.5],
    [ .5,  .5],
    [-.5,  .5]])

class TestReducedOrderModel(TestCase):
    def test_constructor(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T, scaler_snapshots=Scale('minmax'))
        rom = ROM(db, pod, rbf)

    def test_predict(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T, scaler_snapshots=Scale('minmax'))
        rom = ROM(db, pod, rbf).fit()
        pred_sol = rom.predict([-0.293344, -0.23120537])
        
        np.testing.assert_allclose(pred_sol, pred_sol_tst, rtol=1e-5, atol=1e-8)
    
    def test_loo_error(self):
        pod = POD()
        rbf = RBF()
        db = Database(param, snapshots.T, scaler_snapshots=Scale('minmax'))
        rom = ROM(db, pod, rbf)
        err = rom.loo_error()
        np.testing.assert_allclose( err,
                np.array([2.41777673, 2.0500897 , 0.27498979, 1.80252838]),
                rtol=1e-5, atol=1e-8)
    
    def test_add_snapshot(self):
        pod = POD()
        rbf = RBF()
        db = Database(param[0:3], snapshots.T[0:3], scaler_snapshots=Scale('minmax'))
        rom = ROM(db, pod, rbf).fit()
        rom.add_snapshot(param[3:4], snapshots.T[3:4])
        pred_sol = rom.predict([-0.293344, -0.23120537])
        
        np.testing.assert_allclose(pred_sol, pred_sol_tst, rtol=1e-5, atol=1e-8)
    
        
    def test_add_snapshots(self):
        pod = POD()
        rbf = RBF()
        db = Database(param[0:2], snapshots.T[0:2], scaler_snapshots=Scale('minmax'))
        rom = ROM(db, pod, rbf).fit()
        rom.add_snapshot(param[2:4], snapshots.T[2:4])
        pred_sol = rom.predict([-0.293344, -0.23120537])
        
        np.testing.assert_allclose(pred_sol, pred_sol_tst, rtol=1e-5, atol=1e-8)


import numpy as np
import pytest

from ezyrb import POD, GPR, RBF, Database, ANN
from ezyrb import KNeighborsRegressor, RadiusNeighborsRegressor, Linear
from ezyrb import ReducedOrderModel as ROM
from ezyrb.plugin.scaler import DatabaseScaler

from sklearn.preprocessing import StandardScaler, MinMaxScaler

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
pred_sol_tst = np.load('tests/test_datasets/p_predsol.npy').T
pred_sol_gpr = np.load('tests/test_datasets/p_predsol_gpr.npy').T
param = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def test_constructor():
    pod = POD()
    import torch
    rbf = RBF()
    #rbf = ANN([10, 10], function=torch.nn.Softplus(), stop_training=[1000])
    db = Database(param, snapshots.T)
    # rom = ROM(db, pod, rbf, plugins=[DatabaseScaler(StandardScaler(), 'full', 'snapshots')])
    rom = ROM(db, pod, rbf, plugins=[
        DatabaseScaler(StandardScaler(), 'reduced', 'parameters'),
        DatabaseScaler(StandardScaler(), 'reduced', 'snapshots')
    ])
    rom.fit()
   
    


def test_values():
    pod = POD()
    rbf = RBF()
    db = Database(param, snapshots.T)
    rom = ROM(db, pod, rbf, plugins=[
        DatabaseScaler(StandardScaler(), 'reduced', 'snapshots'),
        DatabaseScaler(StandardScaler(), 'full', 'parameters')
    ])
    rom.fit()
    test_param = param[2]
    truth_sol = db.snapshots_matrix[2]
    predicted_sol = rom.predict(test_param)[0]
    np.testing.assert_allclose(predicted_sol, truth_sol, 
            rtol=1e-5, atol=1e-5)


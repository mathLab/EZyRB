import numpy as np
import pytest

from ezyrb import POD, GPR, RBF, Database, ANN
from ezyrb import KNeighborsRegressor, RadiusNeighborsRegressor, Linear
from ezyrb import ReducedOrderModel as ROM
from ezyrb.plugin.scaler import DatabaseScaler

from sklearn.preprocessing import StandardScaler

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T
pred_sol_tst = np.load('tests/test_datasets/p_predsol.npy').T
pred_sol_gpr = np.load('tests/test_datasets/p_predsol_gpr.npy').T
param = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def test_constructor():
    pod = POD()
    import torch
    rbf = RBF()
    rbf = ANN([10, 10], function=torch.nn.Softplus(), stop_training=[1000])
    db = Database(param, snapshots.T)
    # rom = ROM(db, pod, rbf, plugins=[DatabaseScaler(StandardScaler(), 'full', 'snapshots')])
    rom = ROM(db, pod, rbf, plugins=[
        DatabaseScaler(StandardScaler(), 'full', 'parameters'),
        DatabaseScaler(StandardScaler(), 'reduced', 'snapshots')
    ])
    rom.fit()
    print(rom.predict(rom.database.parameters_matrix))
    print(rom.database.snapshots_matrix)


# def test_values():
#     snap = Snapshot(test_value)
#     snap.values = test_value
#     snap = Snapshot(test_value, space=test_space)
#     with pytest.raises(ValueError):
#         snap.values = test_value[:-2]


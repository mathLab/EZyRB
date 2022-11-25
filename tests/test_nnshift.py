import numpy as np
import torch
from scipy import spatial
# import pytest

from ezyrb import POD, RBF, Database, Snapshot, Parameter, Linear, ANN
from ezyrb import ReducedOrderModel as ROM
from ezyrb.plugin import AutomaticShiftSnapshots

n_params = 15
params = np.linspace(0.5, 4.5, n_params).reshape(-1, 1)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def wave(t, res=256):
    x = np.linspace(0, 5, res)
    return x, gaussian(x, t, 0.1)


def test_constructor():
    interp = ANN([10, 10], torch.nn.Softplus, 1e-4)
    shift = ANN([10, 10], torch.nn.Softplus, 1e-4)
    AutomaticShiftSnapshots(shift, interp, RBF())


def test_fit_train():
    interp = ANN([10, 10], torch.nn.Softplus(), 1000, frequency_print=50, lr=0.03)
    shift = ANN([], torch.nn.LeakyReLU(), [2000, 1e-3], frequency_print=50, l2_regularization=0,  lr=0.002)
    nnspod = AutomaticShiftSnapshots(shift, interp, Linear(fill_value=0.0), barycenter_loss=10.)
    pod = POD(rank=1)
    rbf = RBF()
    db = Database()
    for param in params:
        space, values = wave(param)
        snap = Snapshot(values=values, space=space)
        db.add(Parameter(param), snap)

    for _ in range(20):
        rom = ROM(db, pod, rbf, plugins=[nnspod])
        rom.fit()

        pred = rom.predict(db.parameters_matrix)

        error = 0.0
        for (_, snap), (_, truth_snap) in zip(pred._pairs, db._pairs):
            tree = spatial.KDTree(truth_snap.space.reshape(-1, 1))
            for coord, value in zip(snap.space, snap.values):
                a = tree.query(coord)
                error += np.abs(value - truth_snap.values[a[1]])

        if error < 80.:
            break

    assert error < 80.

def test_fit_test():
    interp = ANN([10, 10], torch.nn.Softplus(), 1000, frequency_print=200, lr=0.03)
    shift = ANN([], torch.nn.LeakyReLU(), [1e-3, 4000], optimizer=torch.optim.Adam, frequency_print=200, l2_regularization=0,  lr=0.0023)
    nnspod = AutomaticShiftSnapshots(shift, interp, Linear(fill_value=0.0), barycenter_loss=20.)
    pod = POD(rank=1)
    rbf = RBF()
    db = Database()
    for param in params:
        space, values = wave(param)
        snap = Snapshot(values=values, space=space)
        db.add(Parameter(param), snap)
    db_train, db_test = db.split([len(db)-3, 3])
    rom = ROM(db_train, pod, rbf, plugins=[nnspod])
    for _ in range(10):
        rom.fit()
        if rom.plugins[0].shift_network.loss_trend[-1] < 1e-3:
            break
    pred = rom.predict(db_test.parameters_matrix)
    
    error = 0.0
    for (_, snap), (_, truth_snap) in zip(pred._pairs, db_test._pairs):
        tree = spatial.KDTree(truth_snap.space.reshape(-1, 1))
        for coord, value in zip(snap.space, snap.values):
            a = tree.query(coord)
            error += np.abs(value - truth_snap.values[a[1]])

    assert error < 25.

# def test_predict_ref():
#     pod = POD()
#     rbf = RBF()
#     db = Database()
#     for param in params:
#         space, values = wave(param)
#         snap = Snapshot(values=values, space=space)
#         db.add(Parameter(param), snap)
#     rom = ROM(db, pod, rbf, plugins=[
#         ShiftSnapshots(shift, Linear(fill_value=0.0))
#     ])
#     rom.fit()
#     pred = rom.predict(db._pairs[0][0].values)
#     np.testing.assert_array_almost_equal(
#         pred._pairs[0][1].values, db._pairs[0][1].values, decimal=1)


# def test_predict():
#     pod = POD()
#     rbf = Linear()
#     db = Database()
#     for param in params:
#         space, values = wave(param)
#         snap = Snapshot(values=values, space=space)
#         db.add(Parameter(param), snap)
#     rom = ROM(db, pod, rbf, plugins=[
#         ShiftSnapshots(shift, Linear(fill_value=0.0))
#     ])
#     rom.fit()
#     pred = rom.predict(db._pairs[10][0].values)

#     from scipy import spatial
#     tree = spatial.KDTree(db._pairs[10][1].space.reshape(-1, 1))
#     error = 0.0
#     for coord, value in zip(pred._pairs[0][1].space, pred._pairs[0][1].values):
#         a = tree.query(coord)
#         error += value - db._pairs[10][1].values[a[1]]

#     assert error < 1e-5

# def test_values():
#     snap = Snapshot(test_value)
#     snap.values = test_value
#     snap = Snapshot(test_value, space=test_space)
#     with pytest.raises(ValueError):
#         snap.values = test_value[:-2]

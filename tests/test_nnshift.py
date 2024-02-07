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
    seed = 147
    torch.manual_seed(seed)
    np.random.seed(seed)
    interp = ANN([10, 10], torch.nn.Softplus(), 1000, frequency_print=200, lr=0.03)
    shift = ANN([], torch.nn.LeakyReLU(), [2500, 1e-3], frequency_print=200, l2_regularization=0,  lr=0.0005)
    nnspod = AutomaticShiftSnapshots(shift, interp, Linear(fill_value=0.0), barycenter_loss=10.)
    pod = POD(rank=1)
    rbf = RBF()
    db = Database()
    for param in params:
        space, values = wave(param)
        snap = Snapshot(values=values, space=space)
        db.add(Parameter(param), snap)

    rom = ROM(db, pod, rbf, plugins=[nnspod])
    rom.fit()

    pred = rom.predict(db.parameters_matrix)

    error = 0.0
    for (_, snap), (_, truth_snap) in zip(pred._pairs, db._pairs):
        tree = spatial.KDTree(truth_snap.space.reshape(-1, 1))
        for coord, value in zip(snap.space, snap.values):
            a = tree.query(coord)
            error += np.abs(value - truth_snap.values[a[1]])

    assert error < 100.

###################### TODO: extremely long test, need to rethink it
# def test_fit_test():
#     interp = ANN([10, 10], torch.nn.Softplus(), 1000, frequency_print=200, lr=0.03)
#     shift = ANN([], torch.nn.LeakyReLU(), [1e-3, 4000], optimizer=torch.optim.Adam, frequency_print=200, l2_regularization=0,  lr=0.0023)
#     nnspod = AutomaticShiftSnapshots(shift, interp, Linear(fill_value=0.0), barycenter_loss=20.)
#     pod = POD(rank=1)
#     rbf = RBF()
#     db = Database()
#     for param in params:
#         space, values = wave(param)
#         snap = Snapshot(values=values, space=space)
#         db.add(Parameter(param), snap)
#     db_train, db_test = db.split([len(db)-3, 3])
#     rom = ROM(db_train, pod, rbf, plugins=[nnspod])
#     for _ in range(10):
#         rom.fit()
#         if rom.plugins[0].shift_network.loss_trend[-1] < 1e-3:
#             break
#     pred = rom.predict(db_test.parameters_matrix)
    
#     error = 0.0
#     for (_, snap), (_, truth_snap) in zip(pred._pairs, db_test._pairs):
#         tree = spatial.KDTree(truth_snap.space.reshape(-1, 1))
#         for coord, value in zip(snap.space, snap.values):
#             a = tree.query(coord)
#             error += np.abs(value - truth_snap.values[a[1]])

#     assert error < 25.

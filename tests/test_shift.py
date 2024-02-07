import numpy as np
# import pytest

from ezyrb import POD, RBF, Database, Snapshot, Parameter, Linear
from ezyrb import ReducedOrderModel as ROM
from ezyrb.plugin.shift import ShiftSnapshots

n_params = 15
params = np.linspace(0.5, 4.5, n_params).reshape(-1, 1)


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def wave(t, res=256):
    x = np.linspace(0, 5, res)
    return x, gaussian(x, t, 0.1)


def shift(time):
    return time-0.5


def test_constructor():
    ShiftSnapshots(shift, RBF())


def test_fit():
    pod = POD()
    rbf = RBF()
    db = Database()
    for param in params:
        space, values = wave(param)
        snap = Snapshot(values=values, space=space)
        db.add(Parameter(param), snap)
    rom = ROM(db, pod, rbf, plugins=[
        ShiftSnapshots(shift, RBF())
    ])
    rom.fit()


def test_predict_ref():
    pod = POD()
    rbf = RBF()
    db = Database()
    for param in params:
        space, values = wave(param)
        snap = Snapshot(values=values, space=space)
        db.add(Parameter(param), snap)
    rom = ROM(db, pod, rbf, plugins=[
        ShiftSnapshots(shift, Linear(fill_value=0.0))
    ])
    rom.fit()
    pred = rom.predict(db._pairs[0][0].values)
    np.testing.assert_array_almost_equal(
        pred._pairs[0][1].values, db._pairs[0][1].values, decimal=1)


def test_predict():
    pod = POD()
    rbf = Linear()
    db = Database()
    for param in params:
        space, values = wave(param)
        snap = Snapshot(values=values, space=space)
        db.add(Parameter(param), snap)
    rom = ROM(db, pod, rbf, plugins=[
        ShiftSnapshots(shift, Linear(fill_value=0.0))
    ])
    rom.fit()
    pred = rom.predict(db._pairs[10][0].values)

    from scipy import spatial
    tree = spatial.KDTree(db._pairs[10][1].space.reshape(-1, 1))
    error = 0.0
    for coord, value in zip(pred._pairs[0][1].space, pred._pairs[0][1].values):
        a = tree.query(coord)
        error += np.abs(value - db._pairs[10][1].values[a[1]])
        

    assert error < 1.

# def test_values():
#     snap = Snapshot(test_value)
#     snap.values = test_value
#     snap = Snapshot(test_value, space=test_space)
#     with pytest.raises(ValueError):
#         snap.values = test_value[:-2]

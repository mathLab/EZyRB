from ezyrb import Snapshot
import numpy as np

import pytest

test_space = np.linspace(0, 3, 24)
test_value = test_space**2

def test_costructor():
    snap = Snapshot(test_value)
    snap = Snapshot(test_value, space=test_space)
    np.testing.assert_array_equal(snap.space, test_space)
    np.testing.assert_array_equal(snap.values, test_value)

def test_values():
    snap = Snapshot(test_value)
    snap.values = test_value
    snap = Snapshot(test_value, space=test_space)
    with pytest.raises(ValueError):
        snap.values = test_value[:-2]

def test_snapshot_space():
    snap = Snapshot(test_value)

def test_flattened():
    snap = Snapshot(test_value)
    assert snap.flattened.ndim == 1
    snap_3d = Snapshot(np.random.uniform(0, 1, size=(30, 3)))
    assert snap_3d.flattened.ndim == 1



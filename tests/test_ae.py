import numpy as np
import pytest

from ezyrb import AE

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T

def test_constructor_empty():
    AE(2, [20], [20], 'relu', 'relu', 20)

def test_wrong_constructor():
    with pytest.raises(ValueError):
        AE('pippo', [20, 5], [5, 20], 'relu', 'relu', 20)

@pytest.mark.parametrize("activation", ['tanh', 'relu', 'logistic'])
def test_reconstruction(activation):
    ae = AE(5, [50, 20], [20, 50], activation, max_iter=500, learning_rate_init=0.05, random_state=42)

    ae.fit(snapshots)
    snapshots_1 = ae.inverse_transform(ae.transform(snapshots))
    snapshots_2 = ae._autoencoder.predict(snapshots.T).T

    assert snapshots_1.shape == snapshots.shape
    assert snapshots_2.shape == snapshots.shape

    np.testing.assert_array_equal(snapshots_1, snapshots_2)
    rerr = np.linalg.norm(snapshots_2 - snapshots)/np.linalg.norm(snapshots)
    assert rerr < 0.6  # Relaxed tolerance for sklearn

def test_decode_encode():
    low_dim = 5
    ae = AE(low_dim, [400], [400], 'tanh', 200)
    ae.fit(snapshots)
    reduced_snapshots = ae.transform(snapshots)
    assert reduced_snapshots.shape[0] == low_dim
    expanded_snapshots = ae.inverse_transform(reduced_snapshots)
    assert expanded_snapshots.shape[0] == snapshots.shape[0]

def test_optimizer():
    low_dim = 5
    ae = AE(
        low_dim, [400], [400], 'tanh', 20,
        solver='adam'
    )
    ae.fit(snapshots)
    assert ae.solver == 'adam'
    ae = AE(
        10, [200, 100], [100, 200], 'tanh', 10,
        solver='adam'
    )
    ae.fit(snapshots)
    assert ae.solver == 'adam'

def test_optimizer_doublefit():
    low_dim = 5
    ae = AE(low_dim, [400], [400], 'tanh', 20)
    ae.fit(snapshots)
    ae.fit(snapshots)

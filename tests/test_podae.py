import numpy as np

from ezyrb import AE, POD, PODAE

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T


def test_constructor_empty():
    ae = AE(2, [20], [20], 'relu', 20)
    pod = POD()
    PODAE(pod, ae)

def test_reconstruction():
    ae = AE(2, [10], [10], 'relu', 500, learning_rate_init=0.01, random_state=42, n_iter_no_change=50)
    pod = POD(rank=4)
    podae = PODAE(pod, ae)
    podae.fit(snapshots)
    snapshots_ = podae.inverse_transform(podae.transform(snapshots))
    rerr = np.linalg.norm(snapshots_ - snapshots)/np.linalg.norm(snapshots)
    assert rerr < 0.65

def test_decode_encode():
    low_dim = 2
    ae = AE(low_dim, [3 ], [3], 'relu', 1000, learning_rate_init=0.01, random_state=42, n_iter_no_change=50)
    pod = POD(rank=4)
    podae = PODAE(pod, ae)
    podae.fit(snapshots)
    reduced_snapshots = podae.reduce(snapshots)
    assert reduced_snapshots.shape[0] == low_dim
    expanded_snapshots = podae.expand(reduced_snapshots)
    assert expanded_snapshots.shape[0] == snapshots.shape[0]

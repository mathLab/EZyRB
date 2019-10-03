import numpy as np

from unittest import TestCase
from ezyrb import POD

snapshots = np.load('tests/test_datasets/p_snapshots.npy').T

class TestPOD(TestCase):

    def test_constructor_empty(self):
        a = POD()

    def test_1(self):
        POD('svd').reduce(snapshots)
        assert False #TODO check

    def test_2(self):
        POD('correlation_matrix').reduce(snapshots)
        assert False #TODO check

    def test_3(self):
        POD('randomized_svd').reduce(snapshots)
        assert False #TODO check

import numpy as np

from unittest import TestCase
from ezyrb import Database


class TestDatabase(TestCase):
    def test_constructor_empty(self):
        a = Database()

    def test_constructor_arg(self):
        Database(np.random.uniform(size=(10, 3)),
                 np.random.uniform(size=(10, 8)))

    def test_constructor_arg_wrong(self):
        with self.assertRaises(RuntimeError):
            Database(np.random.uniform(size=(9, 3)),
                     np.random.uniform(size=(10, 8)))

    def test_constructor_error(self):
        with self.assertRaises(RuntimeError):
            Database(np.eye(5))

    def test_getitem(self):
        org = Database(np.random.uniform(size=(10, 3)),
                       np.random.uniform(size=(10, 8)))
        new = org[2::2]
        assert new.parameters.shape[0] == new.snapshots.shape[0] == 4

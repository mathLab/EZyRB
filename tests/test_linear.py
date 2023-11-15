import numpy as np
import warnings

from unittest import TestCase
from ezyrb import Linear, Database, POD, ReducedOrderModel

class TestLinear(TestCase):
    def test_params(self):
        reg = Linear(fill_value=0)
        assert reg.fill_value == 0

    def test_default_params(self):
        reg = Linear()
        assert np.isnan(reg.fill_value)

    #def test_predict(self):
    #    reg = Linear()
    #    reg.fit([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0], [20, 5], [8, 6]])
    #    result = reg.predict([[1,2], [8,9], [6,7]])
    #    assert (result[0] == [1,0]).all()
    #    assert (result[1] == [8,6]).all()
    #    assert (result[2] == [20, 5]).all()

    def test_predict1d(self):
        reg = Linear()
        reg.fit([[1], [6], [8]], [[1, 0], [20, 5], [8, 6]])
        result = reg.predict([[1], [8], [6]])
        assert (result[0] == [1,0]).all()
        assert (result[1] == [8,6]).all()
        assert (result[2] == [20, 5]).all()

    def test_predict1d_2(self):
        reg = Linear()
        reg.fit([[1], [2]], [[1,1], [2,2]])
        result = reg.predict([[1.5]])
        assert (result[0] == [[1.5, 1.5]]).all()

    def test_predict1d_3(self):
        reg = Linear()
        reg.fit([1,2], [[1,1], [2,2]])
        result = reg.predict([[1.5]])
        assert (result[0] == [[1.5,1.5]]).all()

    def test_with_db_predict(self):
        reg = Linear()
        pod = POD()
        db = Database(np.array([1, 2, 3])[:,None], np.array([1, 5, 3])[:,None])
        rom = ReducedOrderModel(db, pod, reg)

        rom.fit()
        assert rom.predict([1]) == 1
        assert rom.predict([2]) == 5
        assert rom.predict([3]) == 3

        Y = np.random.uniform(size=(3, 3))
        db = Database(np.array([1, 2, 3]), Y)
        rom = ReducedOrderModel(db, POD(), Linear())
        rom.fit()
        assert rom.predict([1.]).shape == (3,)


    def test_wrong1(self):
        # wrong number of params
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            with self.assertRaises(Exception):
                reg = Linear()
                reg.fit([[1, 2], [6,], [8, 9]], [[1, 0], [20, 5], [8, 6]])

    def test_wrong2(self):
        # wrong number of values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            with self.assertRaises(Exception):
                reg = Linear()
                reg.fit([[1, 2], [6,], [8, 9]], [[20, 5], [8, 6]])

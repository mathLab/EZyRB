import numpy as np
import warnings

from unittest import TestCase
from ezyrb import Linear, Database, POD, ReducedOrderModel, RegularGrid


class TestRegularGrid(TestCase):
    def test_params(self):
        reg = RegularGrid(fill_value=0)
        assert reg.fill_value == 0

    def test_default_params(self):
        reg = RegularGrid()
        assert np.isnan(reg.fill_value)

    # def test_predict(self):
    #    reg = Linear()
    #    reg.fit([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0], [20, 5], [8, 6]])
    #    result = reg.predict([[1,2], [8,9], [6,7]])
    #    assert (result[0] == [1,0]).all()
    #    assert (result[1] == [8,6]).all()
    #    assert (result[2] == [20, 5]).all()

    def test_predict1d(self):
        reg = RegularGrid()
        x1 = np.array([1, 6, 8])
        V = np.array([[1, 0], [20, 5], [8, 6]])  # n, r = 3, 2
        grid = [x1, ]
        reg.fit(grid, V)
        result = reg.predict([[1], [8], [6]])
        assert (result[0] == [1, 0]).all()
        assert (result[1] == [8, 6]).all()
        assert (result[2] == [20, 5]).all()

    def test_predict1d_2(self):
        reg = RegularGrid()
        x1 = [1, 2]
        reg.fit([x1, ], [[1, 1], [2, 2]])
        result = reg.predict([[1.5]])
        assert (result[0] == [[1.5, 1.5]]).all()

    def test_predict2d(self):
        reg = RegularGrid()
        x1 = [1, 2, 3]
        x2 = [4, 5, 6, 7]
        V = [[1, 21], [2, 22], [3, 23], [4, 24], [5, 25], [6, 26],
             [7, 27], [8, 28], [9, 29], [10, 30], [11, 31], [12, 32]]
        reg.fit([x1, x2], V, method="linear")
        result = reg.predict([[1, 4], [1, 5]])
        assert (result[0] == [1, 21]).all()
        assert (result[1] == [2, 22]).all()

    # TODO: test kvargs? depend on scipy version....
    # TODO: rom.fit() does not work, use reduction.fit() and approximation.fit() instead.

    # def test_with_db_predict(self):
    #     reg = RegularGrid()
    #     pod = POD()
    #     x1 = np.array([1, 2, 3])
    #     xx, _ = np.meshgrid(x1, 1, indexing="ij")
    #     db = Database(xx,
    #                   np.array([1, 5, 3])[:, None])
    #     rom = ReducedOrderModel(db, pod, reg)

    #     rom.fit()
    #     assert rom.predict([1]) == 1
    #     assert rom.predict([2]) == 5
    #     assert rom.predict([3]) == 3

    # def test_wrong1(self):
    #     # wrong number of params
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             "ignore", category=np.VisibleDeprecationWarning)
    #         with self.assertRaises(Exception):
    #             reg = Linear()
    #             reg.fit([[1, 2], [6, ], [8, 9]], [[1, 0], [20, 5], [8, 6]])

    # def test_wrong2(self):
    #     # wrong number of values
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings(
    #             "ignore", category=np.VisibleDeprecationWarning)
    #         with self.assertRaises(Exception):
    #             reg = Linear()
    #             reg.fit([[1, 2], [6, ], [8, 9]], [[20, 5], [8, 6]])

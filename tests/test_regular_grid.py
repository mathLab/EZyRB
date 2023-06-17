import numpy as np
from unittest import TestCase
from ezyrb import RegularGrid  # Database, POD, ReducedOrderModel


class TestRegularGrid(TestCase):
    def test_params(self):
        reg = RegularGrid(fill_value=0)
        assert reg.fill_value == 0

    def test_default_params(self):
        reg = RegularGrid()
        assert np.isnan(reg.fill_value)

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

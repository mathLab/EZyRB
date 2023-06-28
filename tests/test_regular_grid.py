import numpy as np
from unittest import TestCase, main
from ezyrb import RegularGrid, Database, POD, ReducedOrderModel


class TestRegularGrid(TestCase):

    def test_1D_1mode(self):
        reg = RegularGrid()
        reg.fit(np.array([1, 2, 3])[:, None],
                np.array([1, 5, 3])[:, None])
        assert reg.predict([1]) == 1
        assert reg.predict([2]) == 5
        assert reg.predict([3]) == 3

    def test_predict1d(self):
        reg = RegularGrid()
        x1 = np.array([[1], [6], [8]])
        V = np.array([[1, 0], [20, 5], [8, 6]])  # n, r = 3, 2
        reg.fit(x1, V)
        result = reg.predict([[1], [8], [6]])
        assert (result[0] == [1, 0]).all()
        assert (result[1] == [8, 6]).all()
        assert (result[2] == [20, 5]).all()

    def test_predict1d_2(self):
        reg = RegularGrid()
        x1 = [1, 2]
        reg.fit(x1, [[1, 1], [2, 2]])
        result = reg.predict([[1.5]])
        assert (result[0] == [[1.5, 1.5]]).all()

    def test_predict2d(self):
        reg = RegularGrid()
        x1 = [1, 2, 3]
        x2 = [4, 5, 6, 7]
        xx1, xx2 = np.meshgrid(x1, x2, indexing="ij")
        points = np.c_[xx1.ravel(), xx2.ravel()]
        V = [[1, 21], [2, 22], [3, 23], [4, 24], [5, 25], [6, 26],
             [7, 27], [8, 28], [9, 29], [10, 30], [11, 31], [12, 32]]
        reg.fit(points, V, method="linear")
        result = reg.predict([[1, 4], [1, 5]])
        assert (result[0] == [1, 21]).all()
        assert (result[1] == [2, 22]).all()

    def test_get_grid_axes_2D(self):
        x1 = [.1, .2, .3]
        x2 = [4, 5, 6, 7]
        xx1, xx2 = np.meshgrid(x1, x2, indexing="ij")
        V = np.arange(len(x1)*len(x2)*2).reshape(-1, 2)
        pts = np.c_[xx1.ravel(), xx2.ravel()]

        random_order = np.arange(len(pts))
        np.random.shuffle(random_order)
        pts_scrmbld = pts[random_order, :]
        V_scrmbld = V[random_order, :]

        reg = RegularGrid()
        grid_axes, V_unscrambled = reg.get_grid_axes(pts_scrmbld, V_scrmbld)

        assert np.allclose(V_unscrambled, V)
        assert np.allclose(grid_axes[0], x1)
        assert np.allclose(grid_axes[1], x2)

    def test_get_grid_axes_3D(self):
        x1 = [.1, .2, .3]
        x2 = [4, 5, 6, 7]
        x3 = [.12, .34, .56, .78, .90]
        xx1, xx2, xx3 = np.meshgrid(x1, x2, x3, indexing="ij")
        V = np.arange(len(x1)*len(x2)*len(x3)*2).reshape(-1, 2)
        pts = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]

        random_order = np.arange(len(pts))
        np.random.shuffle(random_order)
        pts_scrmbld = pts[random_order, :]
        V_scrmbld = V[random_order, :]

        reg = RegularGrid()
        grid_axes, V_unscrambled = reg.get_grid_axes(pts_scrmbld, V_scrmbld)

        assert np.allclose(V_unscrambled, V)
        assert np.allclose(grid_axes[0], x1)
        assert np.allclose(grid_axes[1], x2)
        assert np.allclose(grid_axes[2], x3)

    def test_with_db_predict(self):
        reg = RegularGrid()
        pod = POD()
        db = Database(np.array([1, 2, 3])[:, None],
                      np.array([1, 5, 3])[:, None])
        rom = ReducedOrderModel(db, pod, reg)
        rom.fit()
        assert rom.predict([1]) == 1
        assert rom.predict([2]) == 5
        assert rom.predict([3]) == 3

    def test_fails(self):
        reg = RegularGrid()
        reg = RegularGrid()
        p = [[1, 2]]
        V = [[1, 1], [2, 2]]
        with self.assertRaises(ValueError):
            reg.fit(p, V)


if __name__ == "__main__":
    main()

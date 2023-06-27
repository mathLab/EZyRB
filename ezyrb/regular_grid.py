"""
Module for higher order interpolation on regular grids
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from .approximation import Approximation


class RegularGrid(Approximation):
    """
    Multidimensional interpolator on regular grids.
    See also scipy's RegularGridInterpolator for information on kwargs.

    :param array_like points: the coordinates of the points on a regular grid.
    :param array_like values: The (vector-) data on the regular grid in
        n dimensions.

    :Example:

        >>> import ezyrb
        >>> import numpy as np
        >>> def f(x, y, z):
        ...     return 2 * x**3 + 3 * y**2 - z
        >>> x = np.linspace(1, 4, 11)
        >>> y = np.linspace(4, 7, 22)
        >>> z = np.linspace(7, 9, 33)
        >>> xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
        >>> points = np.c_[xg.ravel(), yg.ravel(), zg.ravel()]
        >>> data_mode_x = f(xg, yg, zg).reshape(-1, 1)
        # lets assume we have 2 modes, i.e. a rank 2 model
        >>> data = np.concatenate((data_mode_x, data_mode_x/10), axis=1)
        >>> rgi = ezyrb.RegularGrid()
        >>> rgi.fit(points, data, method="linear")
        >>> pts = np.array([[2.1, 6.2, 8.3],
        ...                 [3.3, 5.2, 7.1],
        ...                 [1., 4., 7.],
        ...                 [4., 7., 9.]])
        >>> rgi.predict(pts)
        array([[125.80469388,  12.58046939],
               [146.30069388,  14.63006939],
               [ 43.        ,   4.3       ],
               [266.        ,  26.6       ]])
        >>> f(pts[:, 0], pts[:, 1], pts[:, 2])
        array([125.542, 145.894,  43.   , 266.   ])
        >>> f(pts[:, 0], pts[:, 1], pts[:, 2])/10
        array([12.5542, 14.5894,  4.3   , 26.6   ])
    """

    def __init__(self):
        self.interpolator = None

    def get_grid_axes(self, pts_scrmbld, vals_scrmbld):
        """
        Calculates the grid axes from a meshed grid and re-orders the values so
        they fit on a mesh generated with
        numpy.meshgrid(ax1, ax2, ..., axn, indexing="ij")

        :param array_like pts_scrmbld: the coordinates of the points.
        :param array_like vals_scrmbld: the (vector-)values in the points.
        :return: The grid axes given as a tuple of ndarray, with shapes
            (m1, ), …, (mn, ) and values mapped on the ordered grid.
        :rtype: (list, numpy.ndarray)

        """
        grid_axes = []
        iN = []  # index in dimension N
        nN = []  # size of dimension N
        dim = pts_scrmbld.shape[1]
        for i in range(dim):
            xn, unique_inverse_n = np.unique(pts_scrmbld[:, i],
                                             return_inverse=True)
            grid_axes.append(xn)
            nN.append(len(xn))
            iN.append(unique_inverse_n)
        if np.prod(nN) != len(vals_scrmbld):
            msg = "Values don't match grid. Make sure to pass a list of "\
                  "points that are on a regular grid! Be aware of floating "\
                  "point precision."
            raise ValueError(msg)
        new_row_index = np.ravel_multi_index(iN, nN)
        reverse_scrambling = np.argsort(new_row_index)
        vals_on_regular_grid = vals_scrmbld[reverse_scrambling]
        return grid_axes, vals_on_regular_grid

    def fit(self, points, values, **kwargs):
        """
        Construct the interpolator given `points` and `values`.
        Assumes that the points are on a regular grid, fails when not.
        see scipy.interpolate.RegularGridInterpolator

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        points = np.asarray(points)
        if not np.issubdtype(points.dtype, np.number):
            raise ValueError('Invalid format or dimension for the argument'
                             '`points`.')
        if points.ndim == 1:
            points = points[:, None]
        vals = np.asarray(values)
        grid_axes, values_grd = self.get_grid_axes(points, vals)
        shape = [len(ax) for ax in grid_axes]
        shape.append(-1)
        self.interpolator = RGI(grid_axes, values_grd.reshape(shape), **kwargs)

    def predict(self, new_point):
        """
        Evaluate interpolator at given `new_point`, can be multiple points.

        :param array_like new_point: the coordinates of the given point(s).
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        new_point = np.asarray(new_point)
        if new_point.ndim == 1:
            new_point = new_point[:, None]
        return self.interpolator(new_point)

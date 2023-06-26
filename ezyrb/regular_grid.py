"""
Module for higher order interpolation on regular grids
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .approximation import Approximation
from datetime import datetime as dt


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
        self.dim = None
        self.n_modes = 0
        self.mode_nr = None

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
        # be aware of floating point precision in points!
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
            raise ValueError("points and values are not on a regular grid")
        new_row_index = calculate_flat_index(iN, nN)
        reverse_scrambling = np.argsort(new_row_index)
        vals_on_regular_grid = vals_scrmbld[reverse_scrambling, :]
        return grid_axes, vals_on_regular_grid

    def fit(self, points, values, **kwargs):
        """
        Construct the interpolator given `points` and `values`.
        Assumes that the points are on a regular grid, fails when not.
        see scipy.interpolate.RegularGridInterpolator

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        # we have two options
        # 1.: we could make an interpolator for every mode and its coefficients
        # or 2.: we "interpolate" the mode number
        # option 1 is cleaner, but option 2 performs better
        # X = U S VT, X being shaped (m, n)
        points = np.array(points)
        if not np.issubdtype(points.dtype, np.number):
            raise ValueError('Invalid format or dimension for the argument'
                             '`points`.')

        if len(points.shape) == 1:
            points.shape = (-1, 1)

        self.dim = len(points[0])
        vals = np.asarray(values)
        grid_axes, values_grd = self.get_grid_axes(points, vals)
        self.n_modes = vals.T.shape[0]  # vals = (S@VT).T = S@V
        if self.n_modes > 1:
            self.mode_nr = np.arange(self.n_modes)
            extended_grid = [self.mode_nr, *grid_axes]
            shape = [self.n_modes, ]
        else:
            extended_grid = grid_axes
            shape = []
        for i in range(self.dim):
            shape.append(len(grid_axes[i]))
        assert np.prod(shape) == values_grd.size, "Values don't match grid. "\
            "Make sure to pass a grid, not a list of points!\n"\
            "HINT: did you use rom.fit()? This method does not work with a "\
            "grid. Use reduction.fit(...) and approximation.fit(...) instead."
        self.interpolator = RegularGridInterpolator(extended_grid,
                                                    values_grd.T.reshape(
                                                        shape),
                                                    **kwargs)

    def predict(self, new_point):
        new_point = np.array(new_point)
        if len(new_point.shape) == 1:
            new_point.shape = (-1, 1)

        dim = self.dim
        if self.n_modes > 1:
            xi_extended = np.zeros((len(self.mode_nr), len(new_point), dim+1))
            xi_extended[:, :, 0] = self.mode_nr[:, None]
            for i in range(dim):
                xi_extended[:, :, i+1] = np.array(new_point)[:, i]
        else:
            xi_extended = new_point
        return self.interpolator(xi_extended).T


def calculate_flat_index(iN, nN):
    """
    Calculates the flat index for a multidimensional array given the indices
    and dimensions.

    :param list iN: indices representing the position of the element(s)
                    in each dimension.
    :param list nN: size of the array in each dimension.

    :rtype: numpy.ndarray
    """
    # index = i1 + n1 * (i2 + n2 * (... (iN-1 + nN-1 * iN) ...))
    if len(iN) != len(nN):
        raise ValueError("The lengths of iN and nN should be the same.")

    if any((i < 0).any() or (i >= n).any() for i, n in zip(iN, nN)):
        raise ValueError("The indices are out of bounds.")

    index = 0
    for i, n in zip(iN, nN):
        index = i + n * index

    return index

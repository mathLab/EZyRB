"""
Module for higher order interpolation on regular grids
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .approximation import Approximation


class RegularGrid(Approximation):
    """
    Multidimensional interpolator on regular grids.
    See scipy's RegularGridInterpolator

    :param float fill_value: value used to fill in for requested points outside
        of the convex hull of the input points. If not provided, then the
        default is numpy.nan.
    """

    def __init__(self, fill_value=np.nan):
        self.fill_value = fill_value
        self.interpolator = None
        self.dim = None
        self.mode_nr = None

    def fit(self, points, values, **kvargs):
        """
        Construct the interpolator given `points` and `values`.
        see scipy.interpolate.RegularGridInterpolator

        :param array_like points: with shapes (m1, ), ..., (mn, )
            The points defining the regular grid in n dimensions. The points in
            each dimension (i.e. every elements of the points tuple) must be
            strictly ascending or descending.

        :param array_like values: shape (m1, ..., mn, ...)
            The data on the regular grid in n dimensions.
        """
        # we have two options
        # 1.: we could make an interpolator for every mode and its coefficients
        # or 2.: we "interpolate" the mode number
        # option 1 is cleaner, but option 2 performs better
        # X = U S VT, X being shaped (m, n)

        self.dim = len(points)
        vals = np.asarray(values)
        r = vals.T.shape[0]  # vals = (S*VT).T
        self.mode_nr = np.arange(r)
        extended_grid = [self.mode_nr, *points]
        shape = [r, ]
        for i in range(self.dim):
            shape.append(len(points[i]))
        assert np.prod(shape) == vals.size, "Values don't match grid. "\
            "Make sure to pass a grid, not a list of points!\n"\
            "HINT: did you use rom.fit()? This method does not work with a "\
            "grid. Use reduction.fit(...) and approximation.fit(...) instead."
        self.interpolator = RegularGridInterpolator(extended_grid,
                                                    vals.T.reshape(shape),
                                                    fill_value=self.fill_value,
                                                    **kvargs)

    def predict(self, new_point):
        dim = self.dim
        xi_extended = np.zeros((len(self.mode_nr), len(new_point), dim+1))
        xi_extended[:, :, 0] = self.mode_nr[:, None]
        for i in range(dim):
            xi_extended[:, :, i+1] = np.array(new_point)[:, i]
        return self.interpolator(xi_extended).T

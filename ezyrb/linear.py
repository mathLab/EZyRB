"""
Module for Linear N-Dimensional Interpolation
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator as LinearNDInterp
from scipy.interpolate import interp1d

from .approximation import Approximation


class Linear(Approximation):
    """
    Multidimensional linear interpolator.

    :param float fill_value: value used to fill in for requested points outside
        of the convex hull of the input points. If not provided, then the
        default is numpy.nan.
    """
    def __init__(self, fill_value=np.nan):
        self.fill_value = fill_value
        self.interpolator = None

    def fit(self, points, values):
        """
        Construct the interpolator given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        # the first dimension is the list of parameters, the second one is
        # the dimensionality of each tuple of parameters (we look for
        # parameters of dimensionality one)
        as_np_array = np.array(points)
        if not np.issubdtype(as_np_array.dtype, np.number):
            raise ValueError('Invalid format or dimension for the argument'
                             '`points`.')

        if as_np_array.shape[-1] == 1:
            as_np_array = np.squeeze(as_np_array, axis=-1)

        if as_np_array.ndim == 1 or (as_np_array.ndim == 2
                                     and as_np_array.shape[1] == 1):
            self.interpolator = interp1d(as_np_array, values, axis=0)
        else:
            self.interpolator = LinearNDInterp(points,
                                               values,
                                               fill_value=self.fill_value)

    def predict(self, new_point):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        return self.interpolator(new_point)

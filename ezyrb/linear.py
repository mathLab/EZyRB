"""
Module for Linear N-Dimensional Interpolation
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator

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

    def fit(self, points, values):
        """
        Construct the interpolator given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        self.intepolator = LinearNDInterpolator(points, values,
            fill_value=self.fill_value)

    def predict(self, new_point):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        return self.interpolator(new_point)

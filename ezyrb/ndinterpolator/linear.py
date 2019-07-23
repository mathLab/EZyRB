"""
Module for Linear Interpolation
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d


class LinearInterpolator(object):
    """
    Multidimensional linear interpolator.

    :param array_like points: the coordinates of the points.
    :param array_like values: the values in the points.
    """

    def __init__(self, points, values):
    
        if points.ndim is 1 or points.shape[1] is 1:
            self._interpolator = interp1d(points.flatten(), values.T)
        else:
            self._interpolator = LinearNDInterpolator(points, values)


    def __call__(self, new_points):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        
        value = self._interpolator(new_points)
        if isinstance(self._interpolator, interp1d):
            value = value.T
        return value

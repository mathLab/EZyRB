"""
Module for Radial Basis Function Interpolation
"""

import numpy as np
from scipy.interpolate import Rbf

from .approximation import Approximation


class RBF(Approximation):
    """
    Multidimensional interpolator using Radial Basis Function.

    :param array_like points: the coordinates of the points.
    :param array_like values: the values in the points.
    :param float radius: the radius used in the basis functions. Default is 1.0.
    :param norm: The function that returns the distance between two points. It
        has to be a function that take as input a vector and return a float
        number.  Default is 'euclidean'.
    :type norm: str or callable
    :param callable basis: the basis function.

    :cvar array_like points: the coordinates of the points.
    :cvar float radius: the radius used in the basis functions. Default is 1.0.
    :cvar callable basis: the basis function.
    :cvar numpy.ndarray weights: the weights matrix.
    """

    def __init__(self, kernel='multiquadric', smooth=0):
        self.kernel = kernel
        self.smooth = smooth

    def fit(self, points, values):
        self.interpolators = []
        for value in values:
            argument = np.hstack([points, value.reshape(-1, 1)]).T
            self.interpolators.append(
                Rbf(*argument, smooth=self.smooth, function=self.kernel))

    def predict(self, new_point):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        return np.array([interp(*new_point) for interp in self.interpolators])

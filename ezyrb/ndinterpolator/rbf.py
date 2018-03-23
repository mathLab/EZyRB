"""
Module for Radial Basis Function Interpolation
"""

import numpy as np
from scipy.spatial.distance import cdist as distance_matrix


class RBFInterpolator(object):
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

    @staticmethod
    def multi_quadratic(X, r):
        """
        It implements the following formula:

        .. math::

            \\varphi( \\boldsymbol{x} ) = \\sqrt{\\boldsymbol{x}^2 +
            r^2}

        :param X: the vector x in the formula above.
        :type X: array_like or float
        :param float r: the parameter r in the formula above.
        :return: the result of the formula above.
        :rtype: array_like or float
        """
        return np.sqrt(r**2 + X**2)

    def __init__(self, points, values, radius=1.0, norm='euclidean',
                 basis=None):

        self.basis = basis
        self.points = points
        self.radius = radius

        if self.basis is None:
            self.basis = self.multi_quadratic

        self.weights = np.linalg.solve(
            self.basis(
                distance_matrix(points, points, metric=norm), self.radius),
            values)

    def __call__(self, new_points):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        return self.basis(
            distance_matrix(new_points, self.points), self.radius).dot(
                self.weights)

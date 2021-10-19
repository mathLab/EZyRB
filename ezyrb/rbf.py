"""Module for Radial Basis Function Interpolation."""

import numpy as np
from scipy.interpolate import Rbf

from .approximation import Approximation


class RBF(Approximation):
    """
    Multidimensional interpolator using Radial Basis Function.

    :param kernel: The radial basis function; the default is ‘multiquadric’.
    :type kernel: str or callable
    :param float smooth: values greater than zero increase the smoothness of
        the approximation. 0 is for interpolation (default), the function will
        always go through the nodal points in this case.

    :cvar kernel: The radial basis function; the default is ‘multiquadric’.
    :cvar list interpolators: the RBF interpolators (the number of
        interpolators depenend by the dimensionality of the output)

    Example:
    >>> import ezyrb
    >>> import numpy as np
    >>>
    >>> x = np.random.uniform(-1, 1, size=(4, 2))
    >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
    >>> rbf = ezyrb.RBF()
    >>> rbf.fit(x, y)
    >>> y_pred = rbf.predict(x)
    >>> print(np.allclose(y, y_pred))

    """

    def __init__(self, kernel='multiquadric', smooth=0):
        self.kernel = kernel
        self.smooth = smooth
        self.interpolators = None

    def fit(self, points, values):
        """
        Construct the interpolator given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        self.interpolators = []
        for value in values.T:
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
        new_point = np.array(new_point)
        return np.array([interp(*new_point.T) for interp in
                         self.interpolators]).T

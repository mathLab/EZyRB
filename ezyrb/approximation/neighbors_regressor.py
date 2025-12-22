"""Module for generic NeighborsRegressor."""

import numpy as np
from .approximation import Approximation


class NeighborsRegressor(Approximation):
    """
    A generic superclass for wrappers of *NeighborsRegressor from sklearn.
    
    This class provides a common interface for neighbor-based regression methods.

    :param kwargs: Arguments passed to the internal instance of
        *NeighborsRegressor.
    
    :Example:
    
        >>> import numpy as np
        >>> from ezyrb import KNeighborsRegressor
        >>> x = np.random.uniform(-1, 1, size=(20, 2))
        >>> y = np.sin(x[:, 0]) + np.cos(x[:, 1])
        >>> knn = KNeighborsRegressor(n_neighbors=5)
        >>> knn.fit(x, y)
        >>> y_pred = knn.predict(x[:5])
    """
    def fit(self, points, values):
        """
        Construct the interpolator given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        points = np.array(points).reshape(len(points), -1)
        values = np.array(values)

        self.regressor.fit(points, values)

    def predict(self, new_point):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        if isinstance(new_point, (list, np.ndarray)):
            new_point = np.array(new_point).reshape(len(new_point), -1)
        else:
            new_point = np.array([new_point])

        return self.regressor.predict(new_point)

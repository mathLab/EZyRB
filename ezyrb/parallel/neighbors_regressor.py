"""Module for generic NeighborsRegressor."""

import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN
from .approximation import Approximation

class NeighborsRegressor(Approximation):
    """
    A generic superclass for wrappers of *NeighborsRegressor from sklearn.

    :param kwargs: arguments passed to the internal instance of
        *NeighborsRegressor.
    """
    @task(target_direction=INOUT)
    def fit(self, points, values):
        """
        Construct the interpolator given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        points = np.array(points).reshape(len(points), -1)
        values = np.array(values)

        self.regressor.fit(points, values)

    @task(returns=np.ndarray, target_direction=IN)
    def predict(self, new_point, scaler_red):
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

        predicted_red_sol = np.atleast_2d(self.regressor.predict(new_point))
        if scaler_red:  # rescale modal coefficients
            predicted_red_sol = scaler_red.inverse_transform(
                predicted_red_sol)
        predicted_red_sol = predicted_red_sol.T
        return predicted_red_sol

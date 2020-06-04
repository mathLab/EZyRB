"""
Module wrapper exploiting `GPy` for Gaussian Process Regression
"""
import GPy
import numpy as np
from scipy.optimize import minimize

from .approximation import Approximation

class GPR(Approximation):
    """
    Multidimensional regression using Gaussian process.

    :cvar numpy.ndarray X_sample: the array containing the input points,
        arranged by row.
    :cvar numpy.ndarray Y_sample: the array containing the output values,
        arranged by row.
    :cvar GPy.models.GPRegression model: the regression model.
    """

    def __init__(self):
        self.X_sample = None
        self.Y_sample = None
        self.model = None

    def fit(self, points, values, kern=None, optimization_restart=20):
        """
        Construct the interpolator given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        """
        self.X_sample = np.asarray(points)
        self.Y_sample = np.asarray(values)

        if kern is None:
            kern = GPy.kern.RBF(
                input_dim=points.shape[1],
                ARD=False)

        self.model = GPy.models.GPRegression(
            points,
            values,
            kern,
            normalizer=True)

        self.model.optimize_restarts(optimization_restart, verbose=False)

    def predict(self, new_points):
        """
        Query the regression model at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        return self.model.predict(new_points)

    def optimal_mu(self, bounds, optimization_restart=10):
        """
        Proposes the next sampling point by looking at the point where the
        Gaussian covariance is maximized.  A gradient method (with multi
        starting points) is adopted for the optimization.

        :param numpy.ndarray bounds: the boundaries in the gradient
            optimization. The shape must be (*input_dim*, 2), where *input_dim*
            is the dimension of the input points.
        :param int optimization_restart: the number of restart in the gradient
            optimization. Default is 10.
        """
        dim = self.X_sample.shape[1]
        min_val = 1
        min_x = None

        def min_obj(X, f=np.linalg.norm):
            return -f(self.predict(X.reshape(1, -1))[1])

        initial_starts = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(optimization_restart, dim))

        # Find the best optimum by starting from n_restart different random
        # points.
        for x0 in initial_starts:
            res = minimize(
                min_obj,
                x0,
                self.model,
                bounds=bounds,
                method='L-BFGS-B')

            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(1, -1)

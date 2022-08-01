"""
Module wrapper exploiting `GPy` for Gaussian Process Regression
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor

from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN
from .approximation import Approximation

class GPR(Approximation):
    """
    Multidimensional regression using Gaussian process.

    :cvar numpy.ndarray X_sample: the array containing the input points,
        arranged by row.
    :cvar numpy.ndarray Y_sample: the array containing the output values,
        arranged by row.
    :cvar GPy.models.GPRegression model: the regression model.

    :Example:

        >>> import ezyrb
        >>> import numpy as np
        >>> x = np.random.uniform(-1, 1, size=(4, 2))
        >>> y = (np.sin(x[:, 0]) + np.cos(x[:, 1]**3)).reshape(-1, 1)
        >>> gpr = ezyrb.GPR()
        >>> gpr.fit(x, y)
        >>> y_pred = gpr.predict(x)
        >>> print(np.allclose(y, y_pred))

    """
    def __init__(self):
        self.X_sample = None
        self.Y_sample = None
        self.model = None

    @task(target_direction=INOUT)
    def fit(self,
            points,
            values,
            kern=None,
            normalizer=True,
            optimization_restart=20):
        """
        Construct the regression given `points` and `values`.

        :param array_like points: the coordinates of the points.
        :param array_like values: the values in the points.
        :param sklearn.gaussian_process.kernels.Kernel kern: kernel object from
            sklearn.
        :param bool normalizer: whether to normilize `values` or not.
            Defaults to True.
        :param int optimization_restart: number of restarts for the
            optimization. Defaults to 20.
        """
        self.X_sample = np.array(points)
        self.Y_sample = np.array(values)
        if self.X_sample.ndim == 1:
            self.X_sample = self.X_sample.reshape(-1, 1)
        if self.Y_sample.ndim == 1:
            self.Y_sample = self.Y_sample.reshape(-1, 1)

        self.model = GaussianProcessRegressor(
            kernel=kern, n_restarts_optimizer=optimization_restart,
            normalize_y=normalizer)
        self.model.fit(self.X_sample, self.Y_sample)

    @task(returns=np.ndarray, target_direction=IN)
    def predict(self, new_points, scaler_red, return_variance=False):
        """
        Predict the mean and the variance of Gaussian distribution at given
        `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :param bool return_variance: flag to return also the variance.
            Default is False.
        :return: the mean and the variance
        :rtype: (numpy.ndarray, numpy.ndarray)
        """
        new_points = np.atleast_2d(new_points)
        predicted_red_sol = np.atleast_2d(self.model.predict(new_points,
            return_std=return_variance))
        if scaler_red:  # rescale modal coefficients
            predicted_red_sol = scaler_red.inverse_transform(
                predicted_red_sol)
        predicted_red_sol = predicted_red_sol.T
        return predicted_red_sol

    def optimal_mu(self, bounds, optimization_restart=10):
        """
        Proposes the next sampling point by looking at the point where the
        Gaussian covariance is maximized. A gradient method (with multi
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

        def min_obj(X):
            return -1 * np.linalg.norm(self.predict(X.reshape(1, -1), True)[1])

        initial_starts = np.random.uniform(bounds[:, 0],
                                           bounds[:, 1],
                                           size=(optimization_restart, dim))

        # Find the best optimum by starting from n_restart different random
        # points.
        for x0 in initial_starts:
            res = minimize(min_obj, x0, bounds=bounds, method='L-BFGS-B')

            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x

        return min_x.reshape(1, -1)

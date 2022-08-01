"""
Module for Linear N-Dimensional Interpolation
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator as LinearNDInterp
from scipy.interpolate import interp1d

from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN
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

    @task(target_direction=INOUT)
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

    @task(returns=np.ndarray, target_direction=IN)
    def predict(self, new_point, scaler_red):
        """
        Evaluate interpolator at given `new_points`.

        :param array_like new_points: the coordinates of the given points.
        :return: the interpolated values.
        :rtype: numpy.ndarray
        """
        new_red_snap = self.interpolator(new_point)

        # Drop the fist 1 from resulted from interp1d
        # (1, 1, latent) --> (1, latent)
        if (new_red_snap.shape[0] == 1) and (new_red_snap.shape[1] == 1):
            new_red_snap = np.squeeze(new_red_snap, axis=0)

        predicted_red_sol = np.atleast_2d(new_red_snap)

        if scaler_red:  # rescale modal coefficients
            predicted_red_sol = scaler_red.inverse_transform(
                predicted_red_sol)
        predicted_red_sol = predicted_red_sol.T

        return predicted_red_sol

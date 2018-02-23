"""
Class for the response surface methodology.
"""

from ezyrb.parametricspace import ParametricSpace
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import os


class Interpolation(ParametricSpace):
    """
    Documentation

    :cvar scipy.interpolate.LinearNDInterpolator interpolator: interpolating
        object for the basis interpolation
    """

    def __init__(self):

        self._interpolator = None

    def generate(self, points, snapshots, interpolator=LinearNDInterpolator):
        """
        Generate the reduced space using the proper orthogonal decomposition:
        the matrix that contains the `snapshots` computed for the `points` is
        decomposed using SVD tecnique to obtain the POD basis. The interpolator
        is built combining this basis.

        :param Snapshots snapshots: the snapshots.
        :param Points points: the parametric points where snapshots were
            computed.
        """
        self.interpolator = interpolator(points.values.T, snapshots.values.T)

    def __call__(self, value):
        """
        Project a new parametric point onto the space and a new
        approximated solution is provided.

        :param numpy.ndarray value: the new parametric point
        """
        return self.interpolator(value)

    @staticmethod
    def loo_error(points, snapshots, func=np.linalg.norm):
        """
        Compute the error for each parametric point with a leave-one-out (loo)
        strategy.

        :param Points points: the points where snapshots were computed.
        :param Snapshots snapshots: the snapshots.
        :param function func: the function to estimate error; default is the
            numpy.linalg.norm
        """
        loo_error = np.zeros(points.size)

        for j in np.arange(points.size):

            remaining_index = list(range(j)) + list(range(j + 1, points.size))
            remaining_snaps = snapshots[remaining_index]
            remaining_pts = points[remaining_index]

            subspace = Interpolation()
            subspace.generate(remaining_pts, remaining_snaps)
            projection = subspace(points[j].values)
            if projection is not float:
                projection = np.sum(remaining_snaps.values) / (points.size - 1)

            loo_error[j] = func(snapshots[j].values - projection)
            loo_error[j] /= func(snapshots[0].values)

        return loo_error

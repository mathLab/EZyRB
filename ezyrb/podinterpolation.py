"""
Module for the generation of the reduced space by using the Proper Orthogonal
Decomposition
"""

from ezyrb.parametricspace import ParametricSpace
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import os


class PODInterpolation(ParametricSpace):
    """
    Documentation

    :cvar numpy.ndarray pod_basis: basis extracted from the proper orthogonal
        decomposition.
    :cvar scipy.interpolate.LinearNDInterpolator interpolator: interpolating
        object for the pod basis interpolation
    """

    def __init__(self):

        self._interpolator = None
        self._pod_basis = None

    @property
    def pod_basis(self):
        return self._pod_basis

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
        eig_vec, eig_val = np.linalg.svd(
            snapshots.weighted, full_matrices=False)[0:2]

        self._pod_basis = np.sqrt(snapshots.weights) * eig_vec
        coefs = self._pod_basis.T.dot(snapshots.weighted)
        self._interpolator = interpolator(points.values.T, coefs)

    def __call__(self, value):
        """
        Project a new parametric point onto the reduced space and a new
        approximated solution is provided.

        :param numpy.ndarray value: the new parametric point
        """
        return self._pod_basis.dot(self._interpolator(value).T)

    @staticmethod
    def loo_error(points, snapshots, func=np.linalg.norm):
        """
        Compute the error for each parametric point as projection of the
        snapshot onto the POD basis with a leave-one-out (loo) strategy.

        :param Points points: the points where snapshots were computed.
        :param Snapshots snapshots: the snapshots.
        :param function func: the function to estimate error; default is the
            numpy.linalg.norm
        """
        loo_error = np.zeros(points.size)

        for j in np.arange(points.size):

            remaining_index = list(range(j)) + list(range(j + 1, points.size))
            remaining_snaps = snapshots[remaining_index]

            eigvec = np.linalg.svd(
                remaining_snaps.weighted, full_matrices=False)[0]

            loo_basis = np.sqrt(remaining_snaps.weights) * eigvec

            projection = np.sum(
                np.array([
                    np.dot(snapshots[j].weighted, basis) * basis
                    for basis in loo_basis.T
                ]),
                axis=0)

            error = (snapshots[j].values - projection) * snapshots.weights
            loo_error[j] = func(error) / func(snapshots[0].values)

        return loo_error

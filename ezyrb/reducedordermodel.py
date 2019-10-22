"""
Module for the Reduced Order Modeling
"""
import numpy as np
import math
from ezyrb import Database, Scale
from scipy.spatial import Delaunay


class ReducedOrderModel(object):
    def __init__(self, database, reduction, approximation):
        self.database = database
        self.reduction = reduction
        self.approximation = approximation

    def fit(self):
        """
        Calculate reduced space
        """
        self.approximation.fit(self.database.parameters,
                               self.reduction.reduce(self.database.snapshots.T))

        return self

    def predict(self, mu):
        """
        Calculate predicted solution for given mu
        """
        return self.database.scaler_snapshots.inverse(
            self.reduction.expand(self.approximation.predict(mu)))

    def loo_error(self, norm=np.linalg.norm):
        """
        Estimate the approximation error using *leave-one-out* strategy. The
        main idea is to create several reduced spaces by combining all the
        snapshots except one. The error vector is computed as the difference
        between the removed snapshot and the projection onto the properly
        reduced space. The procedure repeats for each snapshot in the database.
        The `func` is applied on each vector of error to obtained a float
        number.

        :param function func: the function used to assign at each vector of
            error a float number. It has to take as input a 'numpy.ndarray` and
            returns a float. Default value is the L2 norm.
        :return: the vector that contains the errors estimated for all
            parametric points.
        :rtype: numpy.ndarray
        """

        points = self.database.parameters.shape[0]

        error = np.zeros(points)

        for j in np.arange(points):
            remaining_index = list(range(j)) + list(range(j + 1, points))
            remaining_snaps = self.database.snapshots[remaining_index]
            remaining_param = self.database.parameters[remaining_index]

            db = Database(remaining_param,
                          remaining_snaps,
                          scaler_snapshots=Scale('minmax'))
            rom = ReducedOrderModel(db, self.reduction,
                                    self.approximation).fit()

            error[j] = norm(self.database.snapshots[j] -
                            rom.predict(self.database.parameters[j]))

        return error

    def add_snapshot(self, new_parameters, new_snapshots):
        """
        This method adds the new solution to the database and the new parameter
        values to the parameter points.

        :param numpy.ndarray new_parameters: the parameters value to add to database.
        :param numpy.ndarray new_snapshots: the snapshots to add to database.
        """
        self.database.add(new_parameters, new_snapshots)
        self.fit()

    def optimal_mu(self, error=None, k=1):
        """
        Return the parametric points where new high-fidelity solutions have to
        be computed in ordere to globaly reduce the estimated error. These
        points are the barycentric center of the region (simplex) with higher
        error.

        :param numpy.ndarray error: the estimated error evaluated for each
            snapshot; if error array is not passed, it is computed using
            :func:`loo_error` with the default function. Default value is None.
        :param int k: the number of optimal points to return. Default value is
            1.
        :return: the optimal points
        :rtype: list(numpy.ndarray)
        """
        if error is None:
            error = self.loo_error()

        mu = self.database.parameters
        tria = Delaunay(mu)

        error_on_simplex = np.array([
            np.sum(error[smpx]) * self._simplex_volume(mu[smpx])
            for smpx in tria.simplices
        ])

        barycentric_point = []
        for index in np.argpartition(error_on_simplex, -k)[-k:]:
            worst_tria_pts = mu[tria.simplices[index]]
            worst_tria_err = error[tria.simplices[index]]

            barycentric_point.append(
                np.average(worst_tria_pts, axis=0, weights=worst_tria_err))

        return barycentric_point

    def _simplex_volume(self, vertices):
        """
         Method implementing the computation of the volume of a N dimensional
         simplex.
         Source from: `wikipedia.org/wiki/Simplex
         <https://en.wikipedia.org/wiki/Simplex>`_.
         :param numpy.ndarray simplex_vertices: Nx3 array containing the
             parameter values representing the vertices of a simplex. N is the
             dimensionality of the parameters.
         :return: N dimensional volume of the simplex.
         :rtype: float
         """
        distance = np.transpose([vertices[0] - vi for vi in vertices[1:]])
        return np.abs(
            np.linalg.det(distance) / math.factorial(vertices.shape[1]))

import numpy as np
from ezyrb import POD, RBF, Database, Scale
from ezyrb.utilities import simplex_volume
from scipy.spatial import Delaunay


class ReducedOrderModel(object):
    def __init__(self, database, reduction, approximation):
        self.database = database
        self.reduction = reduction
        self.approximation = approximation

    def fit(self):
        self.approximation.fit(self.database.parameters,
                               self.reduction.reduce(self.database.snapshots.T))

        return self

    def predict(self, mu):
        print(self.approximation.predict(mu))
        return self.database.scaler_snapshots.inverse(
            self.reduction.expand(self.approximation.predict(mu)))

    def loo_error(self, norm=np.linalg.norm):

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
        This methos adds the new solution to the database and the new parameter
        values to the parameter points

        :param numpy array new_parameters: the parameters value to add to database.
        :param numpy array new_snapshots: the snapshots to add to database.
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
            np.sum(error[smpx]) * simplex_volume(mu[smpx])
            for smpx in tria.simplices
        ])

        barycentric_point = []
        for index in np.argpartition(error_on_simplex, -k)[-k:]:
            worst_tria_pts = mu[tria.simplices[index]]
            worst_tria_err = error[tria.simplices[index]]

            barycentric_point.append(
                np.average(worst_tria_pts, axis=0, weights=worst_tria_err))

        return barycentric_point

"""
Reduced Order Model class
"""
import numpy as np
from ezyrb import POD, Database, Scale

class ReducedOrderModel(object):
    def __init__(self, database, reduction, approximation):
        self.database = database
        self.reduction = reduction
        self.approximation = approximation

    def fit(self):
        """
        Calculate reduced space
        """
        self.approximation.fit(
            self.database.parameters,
            self.reduction.reduce(self.database.snapshots.T))

        return self

    def predict(self, mu):
        """
        Calculate predicted solution for given mu
        """
        print(self.approximation.predict(mu))
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

            db = Database(remaining_param, remaining_snaps,
                          scaler_snapshots=Scale('minmax'))
            rom = ReducedOrderModel(db, self.reduction,
                                    self.approximation).fit()

            error[j] = norm(self.database.snapshots[j] -
                                rom.predict(self.database.parameters[j]))

        return error

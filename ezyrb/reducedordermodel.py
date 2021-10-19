"""Module for the Reduced Order Modeling."""

import math
import copy
import numpy as np
from scipy.spatial.qhull import Delaunay
from sklearn.model_selection import KFold


class ReducedOrderModel():
    """
    Reduced Order Model class.

    This class performs the actual reduced order model using the selected
    methods for approximation and reduction.
    """
    def __init__(self, database, reduction, approximation):
        self.database = database
        self.reduction = reduction
        self.approximation = approximation

    def fit(self, *args, **kwargs):
        r"""
        Calculate reduced space

        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        """
        self.reduction.fit(self.database.snapshots.T)
        self.approximation.fit(
            self.database.parameters,
            self.reduction.reduce(self.database.snapshots.T).T, *args,
            **kwargs)

        return self

    def predict(self, mu):
        """
        Calculate predicted solution for given mu
        """
        predicted_sol = self.reduction.expand(
            np.atleast_2d(self.approximation.predict(mu)).T)
        if 1 in predicted_sol.shape:
            predicted_sol = predicted_sol.ravel()
        return predicted_sol

    def test_error(self, test, norm=np.linalg.norm):
        """
        Compute the mean norm of the relative error vectors of predicted
        test snapshots.

        :param database.Database test: the input test database.
        :param function norm: the function used to assign at the vector of
            errors a float number. It has to take as input a 'numpy.ndarray'
            and returns a float. Default value is the L2 norm.
        :return: the mean L2 norm of the relative errors of the estimated
            test snapshots.
        :rtype: numpy.ndarray
        """
        predicted_test = self.predict(test.parameters)
        return np.mean(
            norm(predicted_test - test.snapshots, axis=1) /
            norm(test.snapshots, axis=1))

    def kfold_cv_error(self, n_splits, *args, norm=np.linalg.norm, **kwargs):
        r"""
        Split the database into k consecutive folds (no shuffling by default).
        Each fold is used once as a validation while the k - 1 remaining folds
        form the training set. If `n_splits` is equal to the number of
        snapshots this function is the same as `loo_error` but the error here
        is relative and not absolute.

        :param int n_splits: number of folds. Must be at least 2.
        :param function norm: function to apply to compute the relative error
            between the true snapshot and the predicted one.
            Default value is the L2 norm.
        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        :return: the vector containing the errors corresponding to each fold.
        :rtype: numpy.ndarray
        """
        error = []
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(self.database):
            new_db = self.database[train_index]
            rom = type(self)(new_db, copy.deepcopy(self.reduction),
                             copy.deepcopy(self.approximation)).fit(
                                 *args, **kwargs)

            error.append(rom.test_error(self.database[test_index], norm))

        return np.array(error)

    def loo_error(self, *args, norm=np.linalg.norm, **kwargs):
        r"""
        Estimate the approximation error using *leave-one-out* strategy. The
        main idea is to create several reduced spaces by combining all the
        snapshots except one. The error vector is computed as the difference
        between the removed snapshot and the projection onto the properly
        reduced space. The procedure repeats for each snapshot in the database.
        The `norm` is applied on each vector of error to obtained a float
        number.

        :param function norm: the function used to assign at each vector of
            error a float number. It has to take as input a 'numpy.ndarray` and
            returns a float. Default value is the L2 norm.
        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        :return: the vector that contains the errors estimated for all
            parametric points.
        :rtype: numpy.ndarray
        """
        error = np.zeros(len(self.database))
        db_range = list(range(len(self.database)))

        for j in db_range:
            remaining_index = db_range[:]
            remaining_index.remove(j)
            new_db = self.database[remaining_index]
            rom = type(self)(new_db, copy.deepcopy(self.reduction),
                             copy.deepcopy(self.approximation)).fit(
                                 *args, **kwargs)

            error[j] = norm(self.database.snapshots[j] -
                            rom.predict(self.database.parameters[j]))

        return error

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

"""Module for the Reduced Order Modeling."""

import math
import copy
import pickle
import numpy as np
from scipy.spatial.qhull import Delaunay
from sklearn.model_selection import KFold

class ReducedOrderModel():
    """
    Reduced Order Model class.

    This class performs the actual reduced order model using the selected
    methods for approximation and reduction.

    :param ezyrb.Database database: the database to use for training the reduced
        order model.
    :param ezyrb.Reduction reduction: the reduction method to use in reduced order
        model.
    :param ezyrb.Approximation approximation: the approximation method to use in
        reduced order model.
    :param object scaler_red: the scaler for the reduced variables (eg. modal
        coefficients). Default is None.

    :cvar ezyrb.Database database: the database used for training the reduced
        order model.
    :cvar ezyrb.Reduction reduction: the reduction method used in reduced order
        model.
    :cvar ezyrb.Approximation approximation: the approximation method used in
        reduced order model.
    :cvar object scaler_red: the scaler for the reduced variables (eg. modal
        coefficients).

    :Example:

         >>> from ezyrb import ReducedOrderModel as ROM
         >>> from ezyrb import POD, RBF, Database
         >>> pod = POD()
         >>> rbf = RBF()
         >>> # param, snapshots and new_param are assumed to be declared
         >>> db = Database(param, snapshots)
         >>> rom = ROM(db, pod, rbf).fit()
         >>> rom.predict(new_param)

    """
    def __init__(self, database, reduction, approximation, scaler_red=None):
        self.database = database
        self.reduction = reduction
        self.approximation = approximation
        self.scaler_red = scaler_red

    def fit(self, *args, **kwargs):
        r"""
        Calculate reduced space

        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        """
        self.reduction.fit(self.database.snapshots.T)
        reduced_output = self.reduction.transform(self.database.snapshots.T).T

        if self.scaler_red:
            reduced_output = self.scaler_red.fit_transform(reduced_output)

        self.approximation.fit(
            self.database.parameters,
            reduced_output,
            *args,
            **kwargs)

        return self

    def predict(self, mu):
        """
        Calculate predicted solution for given mu
        """
        mu = np.atleast_2d(mu)
        if hasattr(self, 'database') and self.database.scaler_parameters:
            mu = self.database.scaler_parameters.transform(mu)

        predicted_red_sol = np.atleast_2d(self.approximation.predict(mu))

        if self.scaler_red:  # rescale modal coefficients
            predicted_red_sol = self.scaler_red.inverse_transform(
                predicted_red_sol)

        predicted_sol = self.reduction.inverse_transform(predicted_red_sol.T)

        if hasattr(self, 'database') and self.database.scaler_snapshots:
            predicted_sol = self.database.scaler_snapshots.inverse_transform(
                    predicted_sol.T).T

        if 1 in predicted_sol.shape:
            predicted_sol = predicted_sol.ravel()
        else:
            predicted_sol = predicted_sol.T
        return predicted_sol

    def save(self, fname, save_db=True, save_reduction=True, save_approx=True):
        """
        Save the object to `fname` using the pickle module.

        :param str fname: the name of file where the reduced order model will
            be saved.
        :param bool save_db: Flag to select if the `Database` will be saved.
        :param bool save_reduction: Flag to select if the `Reduction` will be
            saved.
        :param bool save_approx: Flag to select if the `Approximation` will be
            saved.

        Example:

        >>> from ezyrb import ReducedOrderModel as ROM
        >>> rom = ROM(...) #  Construct here the rom
        >>> rom.fit()
        >>> rom.save('ezyrb.rom')
        """
        rom_to_store = copy.copy(self)

        if not save_db:
            del rom_to_store.database
        if not save_reduction:
            del rom_to_store.reduction
        if not save_approx:
            del rom_to_store.approximation

        with open(fname, 'wb') as output:
            pickle.dump(rom_to_store, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        """
        Load the object from `fname` using the pickle module.

        :return: The `ReducedOrderModel` loaded

        Example:

        >>> from ezyrb import ReducedOrderModel as ROM
        >>> rom = ROM.load('ezyrb.rom')
        >>> rom.predict(new_param)
        """
        with open(fname, 'rb') as output:
            rom = pickle.load(output)

        return rom

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
            indeces = np.array([True] * len(self.database))
            indeces[j] = False

            new_db = self.database[indeces]
            test_db = self.database[~indeces]
            rom = type(self)(new_db, copy.deepcopy(self.reduction),
                             copy.deepcopy(self.approximation)).fit(
                                 *args, **kwargs)

            error[j] = rom.test_error(test_db)

        return error

    def optimal_mu(self, error=None, k=1):
        """
        Return the parametric points where new high-fidelity solutions have to
        be computed in order to globally reduce the estimated error. These
        points are the barycentric center of the region (simplex) with higher
        error.

        :param numpy.ndarray error: the estimated error evaluated for each
            snapshot; if error array is not passed, it is computed using
            :func:`loo_error` with the default function. Default value is None.
        :param int k: the number of optimal points to return. Default value is
            1.
        :return: the optimal points
        :rtype: numpy.ndarray
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

        return np.asarray(barycentric_point)

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

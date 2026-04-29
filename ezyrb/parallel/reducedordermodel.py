"""Module for the Reduced Order Modeling."""

import math
import copy
import pickle
import numpy as np
from scipy.spatial import Delaunay
from sklearn.model_selection import KFold
from pycompss.api.api import compss_wait_on

from ..database import Database
from ..reducedordermodel import ReducedOrderModelInterface


class ReducedOrderModel(ReducedOrderModelInterface):
    """
    Reduced Order Model class.

    This class performs the actual reduced order model using the selected
    methods for approximation and reduction.

    :param ezyrb.Database database: the database to use for training the reduced
        order model.
    :param ezyrb.Reduction reduction: the reduction method to use in reduced
        order model.
    :param ezyrb.Approximation approximation: the approximation method to use in
        reduced order model.
    :param list plugins: list of plugins to use in the reduced order model.
    """

    def __init__(self, database, reduction, approximation, plugins=None):
        self.database = database
        self.reduction = reduction
        self.approximation = approximation
        self.plugins = plugins if plugins is not None else []

    def fit(self, *args, **kwargs):
        r"""
        Calculate reduced space

        :param \*args: additional parameters to pass to the `fit` method.
        :param \**kwargs: additional parameters to pass to the `fit` method.
        """
        # Assign the initial training database
        self.train_full_database = self.database

        self._execute_plugins("fit_preprocessing")
        self._execute_plugins("fit_before_reduction")

        # Fit reduction and transform
        self.reduction.fit(self.train_full_database.snapshots_matrix.T)
        reduced_output = self.reduction.transform(
            self.train_full_database.snapshots_matrix.T
        ).T

        # Store the reduced database for plugins
        self.train_reduced_database = Database(
            self.train_full_database.parameters_matrix, reduced_output
        )

        self._execute_plugins("fit_after_reduction")
        self._execute_plugins("fit_before_approximation")

        # Fit approximation on the reduced database
        self.approximation.fit(
            self.train_reduced_database.parameters_matrix,
            self.train_reduced_database.snapshots_matrix,
            *args,
            **kwargs,
        )

        self._execute_plugins("fit_after_approximation")
        self._execute_plugins("fit_postprocessing")

        return self

    def predict(self, parameters):
        r"""
        Predict the solution for given parameters mu.

        This method distributes the evaluation tasks across the
        available computational nodes using the PyCOMPSs framework.
        """
        is_db = hasattr(parameters, "parameters_matrix")
        mu = (
            parameters.parameters_matrix if is_db else np.atleast_2d(parameters)
        )

        # Setup dummy test_full_database required by some preprocessing plugins
        dummy_snaps = np.zeros(
            (len(mu), self.train_full_database.snapshots_matrix.shape[1])
        )
        self.test_full_database = Database(mu, dummy_snaps)

        # The scaler plugin modifies parameters here BEFORE approximation,
        # so we must initialize this object early with dummy snapshots.
        dummy_red_snaps = np.zeros(
            (len(mu), self.train_reduced_database.snapshots_matrix.shape[1])
        )
        self.predict_reduced_database = Database(mu, dummy_red_snaps)

        self._execute_plugins("predict_preprocessing")
        self._execute_plugins("predict_before_approximation")

        # Predict the reduced solution (using potentially scaled parameters)
        predicted_red_sol = self.approximation.predict(
            self.predict_reduced_database.parameters_matrix
        )

        # Update by creating a NEW database, as snapshots_matrix is read-only
        self.predict_reduced_database = Database(
            self.predict_reduced_database.parameters_matrix, predicted_red_sol
        )

        self._execute_plugins("predict_after_approximation")
        self._execute_plugins("predict_before_expansion")

        # Expand back to full space
        predicted_sol = self.reduction.inverse_transform(
            self.predict_reduced_database.snapshots_matrix.T
        ).T

        # Store the final result for plugins
        self.predicted_full_database = Database(
            self.predict_reduced_database.parameters_matrix, predicted_sol
        )

        self._execute_plugins("predict_after_expansion")
        self._execute_plugins("predict_postprocessing")

        if is_db:
            return self.predicted_full_database
        return self.predicted_full_database.snapshots_matrix

    def test_error(self, test, norm=np.linalg.norm, relative=True):
        """
        Compute the mean norm of the relative error vectors of predicted
        test snapshots.
        """
        predicted_test = self.predict(test.parameters_matrix)

        if hasattr(predicted_test, "snapshots_matrix"):
            pred_snaps = predicted_test.snapshots_matrix
        else:
            pred_snaps = predicted_test

        if relative:
            return np.mean(
                norm(pred_snaps - test.snapshots_matrix, axis=1)
                / norm(test.snapshots_matrix, axis=1)
            )
        else:
            return np.mean(norm(pred_snaps - test.snapshots_matrix, axis=1))

    def save(self, fname, save_db=True, save_reduction=True, save_approx=True):
        """Save the object to `fname` using the pickle module."""
        rom_to_store = copy.copy(self)

        if not save_db:
            del rom_to_store.database
        if not save_reduction:
            del rom_to_store.reduction
        if not save_approx:
            del rom_to_store.approximation

        with open(fname, "wb") as output:
            pickle.dump(rom_to_store, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fname):
        """Load the object from `fname` using the pickle module."""
        with open(fname, "rb") as output:
            rom = pickle.load(output)
        return rom

    def kfold_cv_error(self, n_splits, *args, norm=np.linalg.norm, **kwargs):
        """Split the database into k consecutive folds."""
        error = []
        predicted_test = []  # to save my future objects
        original_test = []
        kf = KFold(n_splits=n_splits)
        for train_index, test_index in kf.split(self.database):
            new_db = self.database[train_index]
            rom = type(self)(
                new_db,
                copy.deepcopy(self.reduction),
                copy.deepcopy(self.approximation),
                plugins=[copy.deepcopy(p) for p in self.plugins],
            ).fit(*args, **kwargs)

            test = self.database[test_index]
            pred = rom.predict(test.parameters_matrix)
            if hasattr(pred, "snapshots_matrix"):
                pred = pred.snapshots_matrix
            predicted_test.append(pred)
            original_test.append(test.snapshots_matrix)

        predicted_test = compss_wait_on(predicted_test)
        for j in range(len(predicted_test)):
            error.append(
                np.mean(
                    norm(predicted_test[j] - original_test[j], axis=1)
                    / norm(original_test[j], axis=1)
                )
            )

        return np.array(error)

    def loo_error(self, *args, norm=np.linalg.norm, **kwargs):
        """Estimate the approximation error using *leave-one-out* strategy."""
        error = np.zeros(len(self.database))
        db_range = list(range(len(self.database)))
        predicted_test = []  # to save my future objects
        original_test = []

        for j in db_range:
            indeces = np.array([True] * len(self.database))
            indeces[j] = False

            new_db = self.database[indeces]
            test_db = self.database[~indeces]
            rom = type(self)(
                new_db,
                copy.deepcopy(self.reduction),
                copy.deepcopy(self.approximation),
                plugins=[copy.deepcopy(p) for p in self.plugins],
            ).fit(*args, **kwargs)

            pred = rom.predict(test_db.parameters_matrix)
            if hasattr(pred, "snapshots_matrix"):
                pred = pred.snapshots_matrix
            predicted_test.append(pred)
            original_test.append(test_db.snapshots_matrix)

        predicted_test = compss_wait_on(predicted_test)
        for j in range(len(predicted_test)):
            error[j] = np.mean(
                norm(predicted_test[j] - original_test[j], axis=1)
                / norm(original_test[j], axis=1)
            )

        return error

    def optimal_mu(self, error=None, k=1):
        """Return the parametric points where new high-fidelity solutions have to be computed."""
        if error is None:
            error = self.loo_error()

        mu = self.database.parameters_matrix
        tria = Delaunay(mu)

        error_on_simplex = np.array(
            [
                np.sum(error[smpx]) * self._simplex_volume(mu[smpx])
                for smpx in tria.simplices
            ]
        )

        barycentric_point = []
        for index in np.argpartition(error_on_simplex, -k)[-k:]:
            worst_tria_pts = mu[tria.simplices[index]]
            worst_tria_err = error[tria.simplices[index]]

            barycentric_point.append(
                np.average(worst_tria_pts, axis=0, weights=worst_tria_err)
            )

        return np.asarray(barycentric_point)

    def _simplex_volume(self, vertices):
        """Method implementing the computation of the volume of a N dimensional simplex."""
        distance = np.transpose([vertices[0] - vi for vi in vertices[1:]])
        return np.abs(
            np.linalg.det(distance) / math.factorial(vertices.shape[1])
        )

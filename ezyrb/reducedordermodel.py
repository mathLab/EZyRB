"""Module for the Reduced Order Modeling."""

import math
import copy
import pickle
import numpy as np
from scipy.spatial.qhull import Delaunay
from sklearn.model_selection import KFold
from .database import Database
from .reduction import Reduction
from .approximation import Approximation

from abc import ABC, abstractmethod

class ReducedOrderModelInterface(ABC):

    def _execute_plugins(self, when):
        """
        Execute the plugins at the specified time, if the plugin has the
        specified method.

        :param str when: the time when the plugins have to be executed.
            The available times are:
                - 'fit_preprocessing'
                - 'fit_before_reduction'
                - 'fit_after_reduction'
                - 'fit_before_approximation'
                - 'fit_after_approximation'
                - 'fit_postprocessing'
                - 'predict_preprocessing'
                - 'predict_postprocessing'
        """
        for plugin in self.plugins:
            if hasattr(plugin, when):
                getattr(plugin, when)(self) 


class ReducedOrderModel(ReducedOrderModelInterface):
    """
    Reduced Order Model class.

    This class performs the actual reduced order model using the selected
    methods for approximation and reduction.

    :param ezyrb.Database database: the database to use for training the
        reduced order model.
    :param ezyrb.Reduction reduction: the reduction method to use in reduced
        order model.
    :param ezyrb.Approximation approximation: the approximation method to use
        in reduced order model.
    :param list plugins: list of plugins to use in the reduced order model.

    :cvar ezyrb.Database database: the database used for training the reduced
        order model.
    :cvar ezyrb.Reduction reduction: the reduction method used in reduced order
        model.
    :cvar ezyrb.Approximation approximation: the approximation method used in
        reduced order model.
    :cvar list plugins: list of plugins used in the reduced order model.

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
    def __init__(self, database, reduction, approximation,
                 plugins=None):

        self.database = database
        self.reduction = reduction
        self.approximation = approximation

        if plugins is None:
            plugins = []

        self.plugins = plugins

        self.clean()        

    def clean(self):
        self.train_full_database = None
        self.train_reduced_database = None
        self.predict_full_database = None
        self.predict_reduced_database = None
        self.test_full_database = None
        self.test_reduced_database = None
        self.validation_full_database = None
        self.validation_reduced_database = None
    
    @property
    def database(self):
        return self._database
    
    @database.setter
    def database(self, value):

        if not isinstance(value, Database):
            raise TypeError(
                "The database has to be an instance of the Database class, or a dictionary of Database.")

        self._database = value

    @database.deleter
    def database(self):
        del self._database

    @property
    def reduction(self):
        return self._reduction
    
    @reduction.setter
    def reduction(self, value):
        if not isinstance(value, Reduction):
            raise TypeError(
                "The reduction has to be an instance of the Reduction class, or a dict of Reduction.")

        self._reduction = value

    @reduction.deleter
    def reduction(self):
        del self._reduction

    @property
    def approximation(self):
        return self._approximation
    
    @approximation.setter
    def approximation(self, value):
        if not isinstance(value, Approximation):
            raise TypeError(
                "The approximation has to be an instance of the Approximation class, or a dict of Approximation.")
        
        self._approximation = value

    @approximation.deleter
    def approximation(self):
        del self._approximation

    @property
    def n_database(self):
        value_, class_ = self.database, Database
        return len(value_) if not isinstance(value_, class_) else 1

    @property
    def n_reduction(self):
        value_, class_ = self.reduction, Reduction
        return len(value_) if not isinstance(value_, class_) else 1

    @property
    def n_approximation(self):
        value_, class_ = self.approximation, Approximation
        return len(value_) if isinstance(value_, class_) else 1

    def fit_reduction(self):

        # for k, rom_ in self.roms.items():
        #     rom_['reduction'].fit(rom_['database'].snapshots_matrix.T)
        if not hasattr(self, 'train_full_database'):
            raise RuntimeError

        self.reduction.fit(self.train_full_database.snapshots_matrix.T)

    def _reduce_database(self, db):
        return Database(
            db.parameters_matrix,
            self.reduction.transform(db.snapshots_matrix.T).T
        )

    def fit_approximation(self):

        if not hasattr(self, 'train_reduced_database'):
            raise RuntimeError

        self.approximation.fit(self.train_reduced_database.parameters_matrix,
                           self.train_reduced_database.snapshots_matrix)

        # ddd


        # if self.n_database == 1 and self.n_reduction == 1:
        #     self.train_full_database = self.database
        #     self.reduction.fit(self.database.snapshots_matrix.T)

        # elif self.n_database == 1 and self.n_reduction > 1:
        #     self.train_full_database = self.database
        #     for reduction in self.reduction:
        #         reduction.fit(self.database.snapshots_matrix.T)

        # elif self.n_database > 1 and self.n_reduction == 1:
        #     self.train_full_database = self.database
        #     self.reduction = [
        #         (k, copy.deepcopy(self.reduction))
        #         for k in self.train_full_database
        #     ]
        #     print(self.reduction)
        #     for reduction, database in zip(self.reduction, self.train_full_database):
        #         self.reduction[reduction].fit(self.train_full_database[database].snapshots_matrix.T)
                 
        # elif self.n_database > 1 and self.n_reduction > 1:
        #     raise NotImplementedError
        # else:
        #     raise RuntimeError

    def fit(self):
        r"""
        Calculate reduced space

        """
        self._execute_plugins('fit_preprocessing')

        if self.train_full_database is None:
           self.train_full_database = copy.deepcopy(self.database)

        self._execute_plugins('fit_before_reduction')

        self.fit_reduction()
        self.train_reduced_database = self._reduce_database(
            self.train_full_database)

        self._execute_plugins('fit_after_reduction')

        self._execute_plugins('fit_before_approximation')

        self.fit_approximation()
                
        self._execute_plugins('fit_after_approximation')

        return self

    def predict(self, parameters):
        """
        Calculate predicted solution for given parameters. If `parameters` is
        a 2D array, the function returns a 2D array of predicted solutions.
        If `parameters` is a Database, the function returns the database of
        predicted solutions.

        :return: the database containing all the predicted solution (with
            corresponding parameters).
        :rtype: Database
        """
        self._execute_plugins('predict_preprocessing')

        if isinstance(parameters, Database):
            self.predict_reduced_database = parameters

        elif isinstance(parameters, (list, np.ndarray, tuple)):

            parameters = np.atleast_2d(parameters)
            self.predict_reduced_database = Database(
                parameters=parameters,
                snapshots=[None] * len(parameters)
            )
        else:
            raise TypeError

        self._execute_plugins('predict_before_approximation')
        # print(self.predict_reduced_database)
        # print(self.predict_reduced_database._pairs)
        # print(self.predict_reduced_database._pairs[0])
        # print(self.predict_reduced_database._pairs[0][1].values)

        print(self.predict_reduced_database.parameters_matrix)
        print(self.approximation.predict(
                self.predict_reduced_database.parameters_matrix))
        self.predict_reduced_database = Database(
            self.predict_reduced_database.parameters_matrix,
            self.approximation.predict(
                self.predict_reduced_database.parameters_matrix).reshape(
                    self.predict_reduced_database.parameters_matrix.shape[0], -1
                )
        )
        # print(self.predict_reduced_database)
        # print(self.predict_reduced_database._pairs)
        # print(self.predict_reduced_database._pairs[0])
        # print(self.predict_reduced_database._pairs[0][1].values)

        self._execute_plugins('predict_after_approximation')

        self._execute_plugins('predict_before_expansion')


        # print(self.predict_reduced_database.snapshots_matrix)
        # print(self.reduction.inverse_transform(
        #             self.predict_reduced_database.snapshots_matrix.T).T)

        self.predict_full_database = Database(
            self.predict_reduced_database.parameters_matrix,
            self.reduction.inverse_transform(
                    self.predict_reduced_database.snapshots_matrix.T).T
        )
        self._execute_plugins('predict_after_expansion')

        self._execute_plugins('predict_postprocessing')

        if isinstance(parameters, Database):
            return self.predict_full_database
        else:
            return self.predict_full_database.snapshots_matrix


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
        predicted_test = self.predict(test.parameters_matrix)
        return np.mean(
            norm(predicted_test - test.snapshots_matrix,
            axis=1) / norm(test.snapshots_matrix, axis=1))

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
                             copy.deepcopy(self.approximation)).fit()

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

        mu = self.database.parameters_matrix
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

class MultiReducedOrderModel(ReducedOrderModelInterface):
    """
    Multiple Reduced Order Model class.

    This class performs the actual reduced order model using the selected
    methods for approximation and reduction.

    :param ezyrb.Database database: the database to use for training the
        reduced order model.
    :param ezyrb.Reduction reduction: the reduction method to use in reduced
        order model.
    :param ezyrb.Approximation approximation: the approximation method to use
        in reduced order model.
    :param list plugins: list of plugins to use in the reduced order model.

    :cvar ezyrb.Database database: the database used for training the reduced
        order model.
    :cvar ezyrb.Reduction reduction: the reduction method used in reduced order
        model.
    :cvar ezyrb.Approximation approximation: the approximation method used in
        reduced order model.
    :cvar list plugins: list of plugins used in the reduced order model.

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
    def __init__(self, *args, plugins=None, rom_plugin=None):

        if len(args) == 3:
            self.database = args[0]
            self.reduction = args[1]
            self.approximation = args[2]

            from itertools import product

            element_keys = product(
                self.database.keys(),
                self.reduction.keys(),
                self.approximation.keys()
            )
            self.roms = {
                
                tuple(key): ReducedOrderModel(
                    copy.deepcopy(self.database[key[0]]),
                    copy.deepcopy(self.reduction[key[1]]),
                    copy.deepcopy(self.approximation[key[2]])
                )
                for key in element_keys
            }

        elif len(args) == 1 and isinstance(args[0], dict):
            self.roms = args[0]

        if plugins is None:
            plugins = []
        self.plugins = plugins

        if rom_plugin is not None:
            for rom_ in self.roms.values():
                rom_.plugins.append(rom_plugin)

    @property
    def database(self):
        return self._database
    
    @database.setter
    def database(self, value):

        if not isinstance(value, (Database, dict)):
            raise TypeError(
                "The database has to be an instance of the Database class, or a dictionary of Database.")

        if isinstance(value, Database):
            self._database = {0: value}
        else: 
            self._database = value

    @database.deleter
    def database(self):
        del self._database

    @property
    def reduction(self):
        return self._reduction
    
    @reduction.setter
    def reduction(self, value):
        if not isinstance(value, (Reduction, dict)):
            raise TypeError(
                "The reduction has to be an instance of the Reduction class, or a dict of Reduction.")

        if isinstance(value, Reduction):
            self._reduction = {0: value}
        else: 
            self._reduction = value

    @reduction.deleter
    def reduction(self):
        del self._reduction

    @property
    def approximation(self):
        return self._approximation
    
    @approximation.setter
    def approximation(self, value):
        if not isinstance(value, (Approximation, dict)):
            raise TypeError(
                "The approximation has to be an instance of the Approximation class, or a dict of Approximation.")
        
        if isinstance(value, Approximation):
            self._approximation = {0: value}
        else:
            self._approximation = value

    @property
    def n_database(self):
        value_, class_ = self.database, Database
        return len(value_) if not isinstance(value_, class_) else 1

    @property
    def n_reduction(self):
        value_, class_ = self.reduction, Reduction
        return len(value_) if not isinstance(value_, class_) else 1

    @property
    def n_approximation(self):
        value_, class_ = self.approximation, Approximation
        return len(value_) if isinstance(value_, class_) else 1

        # ddd


        # if self.n_database == 1 and self.n_reduction == 1:
        #     self.train_full_database = self.database
        #     self.reduction.fit(self.database.snapshots_matrix.T)

        # elif self.n_database == 1 and self.n_reduction > 1:
        #     self.train_full_database = self.database
        #     for reduction in self.reduction:
        #         reduction.fit(self.database.snapshots_matrix.T)

        # elif self.n_database > 1 and self.n_reduction == 1:
        #     self.train_full_database = self.database
        #     self.reduction = [
        #         (k, copy.deepcopy(self.reduction))
        #         for k in self.train_full_database
        #     ]
        #     print(self.reduction)
        #     for reduction, database in zip(self.reduction, self.train_full_database):
        #         self.reduction[reduction].fit(self.train_full_database[database].snapshots_matrix.T)
                 
        # elif self.n_database > 1 and self.n_reduction > 1:
        #     raise NotImplementedError
        # else:
        #     raise RuntimeError

    def fit(self):
        r"""
        Calculate reduced space

        """
        self._execute_plugins('fit_preprocessing')

        for rom_ in self.roms.values():
            rom_.fit()

        self._execute_plugins('fit_postprocessing')
        # print(self.database)
        # print(self.reduction)
        # print(self.approximation)

        # from itertools import product
        # element_keys = product(
        #     self.database.keys(),
        #     self.reduction.keys(),
        #     self.approximation.keys()
        # )
        # self.roms = {
            
        #     tuple(key): {
        #         'database': copy.deepcopy(self.database[key[0]]),
        #         'reduction': copy.deepcopy(self.reduction[key[1]]),
        #         'approximation': copy.deepcopy(self.approximation[key[2]])
        #     }
        #     for key in element_keys
        # }
        # print(self.roms)
        # self._full_database = copy.deepcopy(self.database)

        # # FULL-ORDER PREPROCESSING here
        # for plugin in self.plugins:
        #     plugin.fom_preprocessing(self)

        # self.fit_reduction()
        # # self.reduction.fit(self._full_database.snapshots_matrix.T)
        # # print(self.reduction.singular_values)
        # # print(self._full_database.snapshots_matrix)
        # reduced_snapshots = self.reduction.transform(
        #     self._full_database.snapshots_matrix.T).T

        # self._reduced_database = Database(
        #     self._full_database.parameters_matrix, reduced_snapshots)

        # # REDUCED-ORDER PREPROCESSING here
        # for plugin in self.plugins:
        #     plugin.rom_preprocessing(self)

        # self.approximation.fit(
        #     self._reduced_database.parameters_matrix,
        #     self._reduced_database.snapshots_matrix)

        return self

    def predict(self, parameters):
        """
        Calculate predicted solution for given mu

        :return: the database containing all the predicted solution (with
            corresponding parameters).
        :rtype: Database
        """

        self._execute_plugins('predict_preprocessing')

        self.multi_predict_database = {}
        for k, rom_ in self.roms.items():
            print(rom_.predict_full_database)
            print(rom_.predict_full_database.snapshots_matrix)
            print(rom_.predict_full_database.parameters_matrix)
            self.multi_predict_database[k] = rom_.predict(rom_.predict_full_database)
            print(self.multi_predict_database)
            

        self._execute_plugins('predict_postprocessing')
            
        return self.multi_predict_database

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
        predicted_test = self.predict(test.parameters_matrix)
        return np.mean(
            norm(predicted_test.snapshots_matrix - test.snapshots_matrix,
            axis=1) / norm(test.snapshots_matrix, axis=1))

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
                             copy.deepcopy(self.approximation)).fit()

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

        mu = self.database.parameters_matrix
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

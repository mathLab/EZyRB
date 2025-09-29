from .plugin import Plugin
import numpy as np
from ..approximation.rbf import RBF
from ..approximation.linear import Linear
from ..database import Database
import copy
from scipy.optimize import minimize


class Aggregation(Plugin):
    """
    The `Aggregation` plugin class.
    It implements a general framework for space-dependent aggregation of
    multiple reduced order models (ROMs). The aggregation weights can be
    computed either through standard space-dependent methods (e.g., Gaussian
    functions) or by fitting a regression/interpolation model (e.g., RBF,
    linear, ANN) on the validation set.

    Reference: Ivagnes A., Tonicello N., Rozza G. (2024). `Enhancing non-intrusive
    reduced-order models with space-dependent aggregation methods`, Acta Mechanica.

    :param fit_function: a regression/interpolation model to fit the weights
        in the validation set. If None, standard space-dependent aggregation
        methods are used (default is None).
    :param predict_function: a regression/interpolation model to predict the
        weights in the test set. Default is a linear interpolation.

    Example:
    >>> from ezyrb import POD, RBF, GPR, Database, Snapshot, Parameter, Linear, ANN
    >>> from ezyrb import ReducedOrderModel as ROM
    >>> from ezyrb.plugin import Aggregation
    >>> pod = POD(rank=10)
    >>> rbf = RBF()
    >>> gpr = GPR()
    >>> db = Database(param, snapshots.T)
    >>> rom1 = ROM(db, pod, rbf)
    >>> rom2 = ROM(db, pod, gpr)
    >>> mrom = MROM({'rbf': rom1, 'gpr': rom2}, plugins=[Aggregation(
    ...     fit_function=None, predict_function=Linear())])
    >>> mrom.fit()
    >>> pred = mrom.predict(new_param)
    """

    def __init__(self, fit_function=None, predict_function=Linear()):
        super().__init__()
        self.fit_function = fit_function
        self.predict_function = predict_function

    def _check_sum_gaussians(self, mrom, sum_gaussians, gaussians):
        """
        Method to check if the sum of the Gaussian functions is
        on-zero in all the points of the validation set.
        If there are zero values, the same weight/gaussian is taken for all the ROMs
        at those points.

        :param mrom: the multi reduced order model.
        :param sum_gaussians: the sum of the Gaussian functions.
        :param gaussians: the Gaussian functions.

        :return: the corrected Gaussian functions.
        """
        zero_values = sum_gaussians < 1e-5 # tolerance to avoid numerical issues
       
        if zero_values.any():
            gaussians[:, zero_values] = 1/len(mrom.roms) # equal weights
        return gaussians

    def _optimize_sigma(self, mrom, sigma_range=[1e-5, 1e-2]):
        """
        Method to optimize the sigma parameter in the Gaussian functions
        through a minimization procedure on the validation set.

        :param mrom: the multi reduced order model.
        :param sigma_range: the range of values to search for the optimal sigma.

        :return: the optimal sigma value.
        """
        def obj_func(sigma):
            # compute test error of multiROM from current sigma
            g_sigma = self._compute_validation_weights(mrom, sigma, normalized=True)
            mrom_prediction = np.zeros_like(mrom.validation_full_database.snapshots_matrix)
            for i, rom in enumerate(mrom.roms):
                mrom_prediction += g_sigma[i, ...] * mrom.roms[rom].predict(mrom.validation_full_database).snapshots_matrix
            test_error = np.mean(
                np.linalg.norm(mrom_prediction - mrom.validation_full_database.snapshots_matrix,
            axis=1) /  np.linalg.norm(mrom.validation_full_database.snapshots_matrix, axis=1))
            return test_error
        # minimization procedure
        res = minimize(obj_func, x0=sigma_range[0],
               method="L-BFGS-B", bounds=[sigma_range])
        print('Optimal sigma value in weights: ', res.x)
        return res.x

    def _compute_validation_weights(self, mrom, sigma, normalized=False):
        """
        Method to compute the weights in the validation set
        through standard space-dependent aggregation methods.

        :param mrom: the multi reduced order model.
        :param sigma: the sigma parameter in the Gaussian functions.
        :param normalized: if True, the weights are normalized to sum to 1
            at each spatial point (default is False).

        :return: the weights in the validation set.
        """
        validation_predicted = dict()
        for name, rom in mrom.roms.items():
            validation_predicted[name] = rom.predict(rom.validation_full_database).snapshots_matrix

        g = {}
        for k, v in validation_predicted.items():
            snap = rom.validation_full_database.snapshots_matrix
            g[k] = np.exp(- ((v - snap)**2)/(2 * (sigma**2)))

        g_tensor = np.array([g[k] for k in g.keys()])
        g_tensor = self._check_sum_gaussians(mrom, g_tensor.sum(axis=0), g_tensor)

        if normalized:
            g_tensor /= np.sum(g_tensor, axis=0)

        return g_tensor

    def fit_postprocessing(self, mrom):
        """
        Compute/fit the weights in the validation set.
        If fit_function is None, the weights are computed through standard
        space-dependent aggregation methods (e.g., Gaussian functions).
        If fit_function is not None, the weights are fitted through the
        provided regression/interpolation model (e.g., RBF, linear, ANN).

        :param mrom: the multi reduced order model.

        :return: None
        """
        rom = list(mrom.roms.values())[0]
        
        # concatenate params and space
        params = rom.validation_full_database.parameters_matrix
        input_list = []
        # Loop over the number of snapshots in the validation database
        for i in range(rom.validation_full_database.snapshots_matrix.shape[0]):
            # Get the space coordinates for the i-th snapshot
            space = rom.validation_full_database.get_snapshot_space(i)
            # Get the parameters for the i-th snapshot
            param = rom.validation_full_database.parameters_matrix[i, :]
            # Create the input array for the i-th snapshot
            snapshot_input = np.hstack([
                space,
                np.tile(param, (space.shape[0], 1))
            ])
            # Append the input array to the list
            input_list.append(snapshot_input)
        # Concatenate the input arrays for all snapshots
        input_ = np.vstack(input_list)

        # Fit the regression/interpolation that will be used to predict the
        # weights in the test database
        if self.fit_function is None:
            
            optimal_sigma = self._optimize_sigma(mrom)
            g_tensor = self._compute_validation_weights(mrom, optimal_sigma, normalized=False)
 
            self.predict_functions = []
            for i, rom in enumerate(mrom.roms.values()):
                g_ = g_tensor[i, ...].reshape(space.shape[0] * params.shape[0], -1)
                # do a copy of the function used to predict the weights,
                # otherwise we use the same for all the ROMs
                rom_func = copy.deepcopy(self.predict_function)
                # replace NaN with 0
                # TODO: this is a temporary fix, we should handle NaNs in a better way
                g_[np.isnan(g_)] = 1/len(mrom.roms)
                rom_func.fit(input_, g_)
                self.predict_functions.append(rom_func)
        
 
         # directly fit a regression to minimize the discrepancy between the aggregation and the snapshots
        elif self.fit_function is not None:
            snaps = rom.validation_full_database.snapshots_matrix
            self.fit_function.fit(input_, snaps.reshape(space.shape[0] * params.shape[0], 1))


    def predict_postprocessing(self, mrom):
        """
        Compute the weights in the test set and the predicted solution
        as a convex combination of the predicted solutions of the
        individual ROMs.

        :param mrom: the multi reduced order model.

        :return: None
        """

        # prepare the input for the prediction
        predict_weights = {}
        
        db = list(mrom.multi_predict_database.values())[0]
        
        input_list = []
        
        # Loop over the number of snapshots in the prediction database
        for i in range(mrom.predict_full_database.parameters_matrix.shape[0]):  
            # Get the space coordinates for the i-th snapshot
            space = mrom.train_full_database.get_snapshot_space(i)
            # Get the parameters for the i-th snapshot
            param = mrom.predict_full_database.parameters_matrix[i, :]
            # Create the input array for the i-th snapshot
            snapshot_input = np.hstack([
                space,
                np.tile(param, (space.shape[0], 1))
            ])
            # Append the input array to the list
            input_list.append(snapshot_input)
        # Concatenate the input arrays for all snapshots
        input_ = np.vstack(input_list)
        
        # initialize weights
        mrom.weights_predict = {}
        # predict weights with regression and normalize them
        # case without fit_function (where we use the standard space-dependent aggregation methods)
        if self.fit_function is None and self.predict_function is not None:
            
            gaussians_test = []
            for i, rom in enumerate(mrom.roms.values()):
                g_ = self.predict_functions[i].predict(input_)
                gaussians_test.append(g_)
            gaussians_test = np.array(gaussians_test)
            gaussians_test = gaussians_test.reshape(len(mrom.roms),
                    db.parameters_matrix.shape[0], space.shape[0])
            # normalize weights
            predict_weights = gaussians_test/np.sum(gaussians_test, axis=0)
            # compute predicted solution as convex combination of the reduced solutions
            for i, rom in enumerate(mrom.roms):
                mrom.weights_predict[rom] = predict_weights[i, ...]
                db = mrom.multi_predict_database[rom]
 
        # case with fit_function (ANN)
        elif self.fit_function is not None:
            weights = self.fit_function.predict(input_)
            # compute predicted solution as convex combination of the reduced solutions
            for i, rom in enumerate(mrom.roms):
                mrom.weights_predict[rom] = weights[..., i].reshape(
                    db.parameters_matrix.shape[0], -1)
        
        # compute the prediction
        prediction = np.sum([mrom.weights_predict[k] *
                mrom.multi_predict_database[k].snapshots_matrix for k in
                list(mrom.roms.keys())], axis=0)

        mrom.predict_full_database = Database(
            mrom.predict_full_database.parameters_matrix,
            prediction
            )
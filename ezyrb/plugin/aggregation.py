from .plugin import Plugin
import numpy as np
from ..approximation.rbf import RBF
from ..approximation.linear import Linear
from ..database import Database
import copy


class Aggregation(Plugin):

    def __init__(self, fit_function=None, predict_function=Linear()):
        super().__init__()
        self.fit_function = fit_function
        self.predict_function = predict_function

    def validation_weights(self, mrom, normalized=True):
        if self.fit_function is None:
            # proceed with standard space-dependent aggregation methods
            validation_predicted = dict()
            for name, rom in mrom.roms.items():
                validation_predicted[name] = rom.predict(rom.validation_full_database).snapshots_matrix

            g = {}
            sigma = 0.5 # for now, we use a fixed sigma, we can optimize it
            for k, v in validation_predicted.items():
                snap = rom.validation_full_database.snapshots_matrix
                g[k] = np.exp(- ((v - snap)**2)/(2 * (sigma**2)))

            g_tensor = np.array([g[k] for k in g.keys()])

        else:
            raise NotImplementedError

        if normalized:
            g_tensor /= np.sum(g_tensor, axis=0)
        return g_tensor


    def fit_postprocessing(self, mrom):
        rom = list(mrom.roms.values())[0]

        # concatenate params and space
        # TODO: space should be handled in a different way (snapshots can have
        # different spaces)
        space = rom.validation_full_database._pairs[0][1].space
        params = rom.validation_full_database.parameters_matrix

        input_ = np.hstack([
            np.tile(space, (params.shape[0], 1)),
            np.repeat(params, space.shape[0], axis=0)
        ])

        # Fit the regression/interpolation that will be used to predict the
        # weights in the test database
        if self.fit_function is None:
            g_tensor = self.validation_weights(mrom, normalized=False)

            self.predict_functions = []
            for i, rom in enumerate(mrom.roms.values()):
                g_ = g_tensor[i, ...].reshape(space.shape[0] * params.shape[0], -1)
                # do a copy of the function used to predict the weights,
                # otherwise we use the same for all the ROMs
                rom_func = copy.deepcopy(self.predict_function)
                # replace NaN with 0
                # TODO: this is a temporary fix, we should handle NaNs in a better way
                g_[np.isnan(g_)] = 0
                try:
                    rom_func.fit(input_, g_)
                except MemoryError:
                    rom_func.fit(input_[::20, :], g_[::20, :])
                self.predict_functions.append(rom_func)
            mrom.weights_validation = {}
            for i, rom in enumerate(mrom.roms):
                mrom.weights_validation[rom] = g_tensor[i, ...]/np.sum(
                        g_tensor, axis=0)


        # directly fit a neural network to minimize the discrepancy between the aggregation and the snapshots
        elif self.fit_function is not None:
            snaps = rom.validation_full_database.snapshots_matrix
            self.fit_function.fit(input_, snaps.reshape(space.shape[0] * params.shape[0], 1))


    def predict_postprocessing(self, mrom):
        # prepare the input for the prediction
        space = list(mrom.roms.values())[0].validation_full_database._pairs[0][1].space
        predict_weights = {}

        db = list(mrom.multi_predict_database.values())[0]
        input_ = np.hstack([
            np.tile(space, (db.parameters_matrix.shape[0], 1)),
            np.repeat(db.parameters_matrix, space.shape[0], axis=0)
        ])
        # initialize the predicted solution
        predicted_solution = np.zeros((db.parameters_matrix.shape[0],
                db.snapshots_matrix.shape[1]))

        # initialize weights
        mrom.weights_predict = {}

        # predict weights with regression and normalize them
        # case with fit_function (ANN)
        if self.fit_function is None:
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
                predicted_solution += db.snapshots_matrix * mrom.weights_predict[rom]

        # case without fit_function (where we use the standard space-dependent aggregation methods)
        elif self.fit_function is not None:
            weights = self.fit_function.predict(input_)
            weights /= np.sum(weights, axis=-1)[:, np.newaxis]
            # compute predicted solution as convex combination of the reduced solutions
            for i, rom in enumerate(mrom.roms):
                rom_pred = mrom.roms[rom].predict(db.parameters_matrix)
                mrom.weights_predict[rom] = weights[..., i].reshape(
                    db.parameters_matrix.shape[0], space.shape[0])
                predicted_solution += mrom.weights_predict[rom] * rom_pred

        # store the predicted solution in the reduced database
        mrom.predict_reduced_database = Database(
                mrom.predict_full_database.parameters_matrix,
                predicted_solution)


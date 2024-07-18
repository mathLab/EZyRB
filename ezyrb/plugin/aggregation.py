from .plugin import Plugin
import numpy as np

class Aggregation(Plugin):

    def __init__(self):
        super().__init__()

    def fit_postprocessing(self, mrom):

        validation_predicted = dict()
        for name, rom in mrom.roms.items():
            validation_predicted[name] = rom.predict(rom.validation_full_database.parameters_matrix)

        g = {}
        sigma = 0.1
        for k, v in validation_predicted.items():
            g[k] = np.exp(- (v - rom.validation_full_database.snapshots_matrix)**2/(2 * (sigma**2)))

        g_tensor = np.array([g[k] for k in g.keys()])
        g_tensor /= np.sum(g_tensor, axis=0)

        # concatenate params and space
        space = rom.validation_full_database._pairs[0][1].space
        params = rom.validation_full_database.parameters_matrix
        # compute the aggregated solution
        print(g_tensor.shape)
        weights = []
        for i in range(params.shape[0]):
            a = g_tensor[:, i, :].T
            b = rom.validation_full_database.snapshots_matrix[i]
            param = rom.validation_full_database.parameters_matrix[i]

            w_param = []

            for a_, b_ in zip(a, b):
                A = np.ones((a.shape[1], 2))
                A[0] = a_

                B = np.ones(2)
                B[0] = b_
                # print(A)
                # print(B)

                try:
                    w = np.linalg.solve(A, B).reshape(1, -1)
                except np.linalg.LinAlgError:
                    w = np.zeros(shape=(1, 2)) + 0.5

                w_param.append(w)

            w_param = np.concatenate(w_param)
            weights.append(
                np.hstack(
                    (
                        space, 
                        param.repeat(space.shape[0])[:, None], 
                        w_param
                    )
                )
            )

        weights = np.vstack(weights)

        from ..approximation.rbf import RBF
        from ..approximation.linear import Linear

        self.rbf = Linear()
        self.rbf.fit(weights[::10, :-2], weights[::10, -2:])


    def predict_postprocessing(self, mrom):

        space = list(mrom.roms.values())[0].validation_full_database._pairs[0][1].space
        predict_weights = {}
        db = list(mrom.multi_predict_database.values())[0]
        input_ = np.hstack([
            np.tile(space, (db.parameters_matrix.shape[0], 1)),
            np.repeat(db.parameters_matrix, space.shape[0], axis=0)
        ])
        predict_weights = self.rbf.predict(input_)
        predicted_solution = np.zeros((db.parameters_matrix.shape[0], db.snapshots_matrix.shape[1]))
        print(predicted_solution.shape)
        for w, db in zip(predict_weights.T, mrom.multi_predict_database.values()):
            predicted_solution += db.snapshots_matrix * w.reshape(db.snapshots_matrix.shape[0], -1)
            
        #     input_ = np.hstack([
        #         np.tile(space, (db.parameters_matrix.shape[0], 1)),
        #         np.repeat(db.parameters_matrix, space.shape[0], axis=0)
        #     ])
        #     predict_weights[k] = self.rbf.predict(input_)
        #     print(predict_weights[k])


        return predicted_solution



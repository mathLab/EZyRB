import numpy as np
import torch
from ezyrb import Approximation
from ezyrb.approximation import ANN

class ANN_Demo(ANN):
    def __init__(self, mrom, layers, function, stop_training, loss=None,
                 optimizer=torch.optim.Adam, lr=0.001, l2_regularization=0,
                 frequency_print=10, last_identity=True):
        super().__init__(layers, function, stop_training, loss=None,
                 optimizer=torch.optim.Adam, lr=0.001, l2_regularization=0,
                 frequency_print=10, last_identity=True)

        # import useful data from multirom and roms predictions
        self.mrom = mrom
        self.params = list(self.mrom.roms.values())[0].validation_full_database.parameters_matrix

        # import ROMs and validation predictions of all ROMs
        self.rom_validation_predictions = {}
        for rom in self.mrom.roms:
            rom_pred = self.mrom.roms[rom]
            rom_pred = rom_pred.predict(self.params)
            rom_pred = rom_pred.reshape(rom_pred.shape[0]*rom_pred.shape[1], 1)
            self.rom_validation_predictions[rom] = self._convert_numpy_to_torch(rom_pred)


    def _build_model_(self, points):
        layers = self.layers.copy()
        layers.insert(0, points.shape[1])
        layers.append(len(self.mrom.roms))
        self.model = self._list_to_sequential(layers, self.function)


    def fit(self, points, values): # points=(x, mu) and values=(snapshots)
        self._build_model_(points)
        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.lr, weight_decay=self.l2_regularization)

        points = self._convert_numpy_to_torch(points)
        values = self._convert_numpy_to_torch(values)

        # train the neural network
        n_epoch = 1
        flag = True
        while flag:
            # compute output of ANN (numerators of weights, that have to be
            # normalized)
            y_pred = self.model(points)
            # normalize outputs
            y_pred /= torch.sum(y_pred, dim=-1, keepdim=True)

            # compute aggregated solution from output weights of ANN
            aggr_pred = torch.zeros(values.shape)
            for i, rom in enumerate(self.mrom.roms):
                weight = y_pred.clone()[..., i].unsqueeze(-1)
                aggr_pred += weight*self.rom_validation_predictions[rom]

            # difference between aggregated solution and exact solution
            loss = self.loss(aggr_pred, values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scalar_loss = loss.item()
            self.loss_trend.append(scalar_loss)

            for criteria in self.stop_training:
                if isinstance(criteria, int):  # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):  # stop criteria is float
                    if scalar_loss < criteria:
                        flag = False

            if (flag is False or
                    n_epoch == 1 or n_epoch % self.frequency_print == 0):
                print(f'[epoch {n_epoch:6d}]\t{scalar_loss:e}')

            n_epoch += 1

        return optimizer

    def predict(self, x):
        return super().predict(x)


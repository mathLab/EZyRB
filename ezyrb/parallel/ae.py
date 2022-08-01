"""
Module for FNN-Autoencoders.
"""

import torch
from torch import nn
from .ann import ANN
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN
from .reduction import Reduction

class AE(Reduction, ANN):
    """
    Feed-Forward AutoEncoder class (AE)

    :param list layers_encoder: ordered list with the number of neurons of
        each hidden layer for the encoder
    :param list layers_decoder: ordered list with the number of neurons of
        each hidden layer for the decoder
    :param torch.nn.modules.activation function_encoder: activation function
        at each layer for the encoder, except for the output layer at with
        Identity is considered by default.  A single activation function can
        be passed or a list of them of length equal to the number of hidden
        layers.
    :param torch.nn.modules.activation function_decoder: activation function
        at each layer for the decoder, except for the output layer at with
        Identity is considered by default.  A single activation function can
        be passed or a list of them of length equal to the number of hidden
        layers.
    :param list stop_training: list with the maximum number of training
        iterations (int) and/or the desired tolerance on the training loss
        (float).
    :param torch.nn.Module loss: loss definition (Mean Squared if not
        given).
    :param torch.optim optimizer: the torch class implementing optimizer.
        Default value is `Adam` optimizer.
    :param float lr: the learning rate. Default is 0.001.

    :Example:
        >>> from ezyrb import AE
        >>> import torch
        >>> f = torch.nn.Softplus
        >>> low_dim = 5
        >>> optim = torch.optim.Adam
        >>> ae = AE([400, low_dim], [low_dim, 400], f(), f(), 2000)
        >>> # or ...
        >>> ae = AE([400, 10, 10, low_dim], [low_dim, 400], f(), f(), 1e-5,
        >>>          optimizer=optim)
        >>> ae.fit(snapshots)
        >>> reduced_snapshots = ae.reduce(snapshots)
        >>> expanded_snapshots = ae.expand(reduced_snapshots)
    """
    def __init__(self,
                 layers_encoder,
                 layers_decoder,
                 function_encoder,
                 function_decoder,
                 stop_training,
                 loss=None,
                 optimizer=torch.optim.Adam,
                 lr=0.001):

        if loss is None:
            loss = torch.nn.MSELoss()

        if not isinstance(function_encoder, list):
            # Single activation function passed
            function_encoder = [function_encoder] * (len(layers_encoder))
        if not isinstance(function_decoder, list):
            # Single activation function passed
            function_decoder = [function_decoder] * (len(layers_decoder))

        if not isinstance(stop_training, list):
            stop_training = [stop_training]

        self.layers_encoder = layers_encoder
        self.layers_decoder = layers_decoder
        self.function_encoder = function_encoder
        self.function_decoder = function_decoder
        self.loss = loss

        self.stop_training = stop_training
        self.loss_trend = []
        self.encoder = None
        self.decoder = None
        self.optimizer = optimizer
        self.model = None
        self.lr = lr

    class InnerAE(torch.nn.Module):
        """
        Autoencoder as a pytorch module
        """
        def __init__(self, outer_instance):
            super().__init__()
            self.encoder = outer_instance.encoder
            self.decoder = outer_instance.decoder

        def forward(self, x):
            """
            Compute the forward of the autoencoder
            """
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def _build_model(self, values):
        """
        Build the torch model.

        Considering the number of neurons per layer (self.layers), a
        feed-forward NN is defined:
            - activation function from layer i>=0 to layer i+1:
              self.function[i]; activation function at the output layer:
              Identity (by default).

        :param numpy.ndarray values: the set values one wants to reduce.
        """
        layers_encoder = self.layers_encoder.copy()
        layers_encoder.insert(0, values.shape[1])

        layers_decoder = self.layers_decoder.copy()
        layers_decoder.append(values.shape[1])

        layers_encoder_torch = []
        for i in range(len(layers_encoder) - 2):
            layers_encoder_torch.append(
                nn.Linear(layers_encoder[i], layers_encoder[i + 1]))
            layers_encoder_torch.append(self.function_encoder[i])
        layers_encoder_torch.append(
            nn.Linear(layers_encoder[-2], layers_encoder[-1]))

        layers_decoder_torch = []
        for i in range(len(layers_decoder) - 2):
            layers_decoder_torch.append(
                nn.Linear(layers_decoder[i], layers_decoder[i + 1]))
            layers_decoder_torch.append(self.function_decoder[i])
        layers_decoder_torch.append(
            nn.Linear(layers_decoder[-2], layers_decoder[-1]))
        self.encoder = nn.Sequential(*layers_encoder_torch)
        self.decoder = nn.Sequential(*layers_decoder_torch)
        # Creating the model adding the encoder and the decoder
        self.model = self.InnerAE(self)

    @task(target_direction=INOUT)
    def fit(self, values):
        """
        Build the AE given 'values' and perform training.

        Training procedure information:
            -  optimizer: Adam's method with default parameters (see, e.g.,
               https://pytorch.org/docs/stable/optim.html);
            -  loss: self.loss (if none, the Mean Squared Loss is set by
               default).
            -  stopping criterion: the fulfillment of the requested tolerance
               on the training loss compatibly with the prescribed budget of
               training iterations (if type(self.stop_training) is list); if
               type(self.stop_training) is int or type(self.stop_training) is
               float, only the number of maximum iterations or the accuracy
               level on the training loss is considered as the stopping rule,
               respectively.

        :param numpy.ndarray values: the (training) values in the points.
        """
        values = values.T
        self._build_model(values)
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        values = self._convert_numpy_to_torch(values)
        n_epoch = 1
        flag = True
        while flag:
            y_pred = self.model(values)
            loss = self.loss(y_pred, values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_trend.append(loss.item())
            for criteria in self.stop_training:
                if isinstance(criteria, int):
                    # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):
                    # stop criteria is float
                    if loss.item() < criteria:
                        flag = False

            n_epoch += 1

    @task(returns=np.ndarray, target_direction=IN)
    def transform(self, X, scaler_red):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        X = self._convert_numpy_to_torch(X).T
        g = self.encoder(X)
        reduced_output = (g.cpu().detach().numpy().T).T
        if scaler_red:
            reduced_output = scaler_red.fit_transform(reduced_output)
        return reduced_output

    @task(returns=np.ndarray, target_direction=IN)
    def inverse_transform(self, g, database):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.
        """
        g = self._convert_numpy_to_torch(g).T
        u = self.decoder(g)
        predicted_sol = u.cpu().detach().numpy().T

        if database and database.scaler_snapshots:
            predicted_sol = database.scaler_snapshots.inverse_transform(
                    predicted_sol.T).T

        if 1 in predicted_sol.shape:
            predicted_sol = predicted_sol.ravel()
        else:
            predicted_sol = predicted_sol.T

        return predicted_sol

    def reduce(self, X, scaler_red):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).

        .. note::

            Same as `transform`. Kept for backward compatibility.
        """
        return self.transform(X, scaler_red)

    def expand(self, g, database):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.

        .. note::

            Same as `inverse_transform`. Kept for backward compatibility.
        """
        return self.inverse_transform(g, database)

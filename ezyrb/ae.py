"""
Module for FNN-Autoencoders.
"""

import torch
from .reduction import Reduction
from .ann import ANN


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
    :param float l2_regularization: the L2 regularization coefficient, it
        corresponds to the "weight_decay". Default is 0 (no regularization).
    :param int frequency_print: the frequency in terms of epochs of the print
        during the training of the network.
    :param boolean last_identity: Flag to specify if the last activation
        function is the identity function. In the case the user provides the
        entire list of activation functions, this attribute is ignored. Default
        value is True.

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
                 lr=0.001,
                 l2_regularization=0,
                 frequency_print=10,
                 last_identity=True):

        if layers_encoder[-1] != layers_decoder[0]:
            raise ValueError('Wrong dimension in encoder and decoder layers')

        if loss is None:
            loss = torch.nn.MSELoss()

        if not isinstance(function_encoder, list):
            # Single activation function passed
            layers = layers_encoder
            nl = len(layers)-1 if last_identity else len(layers)
            function_encoder = [function_encoder] * nl

        if not isinstance(function_decoder, list):
            # Single activation function passed
            layers = layers_decoder
            nl = len(layers)-1 if last_identity else len(layers)
            function_decoder = [function_decoder] * nl

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
        self.lr = lr
        self.frequency_print = frequency_print
        self.l2_regularization = l2_regularization

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
        self.encoder = self._list_to_sequential(layers_encoder,
                                                self.function_encoder)

        layers_decoder = self.layers_decoder.copy()
        layers_decoder.append(values.shape[1])
        self.decoder = self._list_to_sequential(layers_decoder,
                                                self.function_decoder)

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

        optimizer = self.optimizer(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr, weight_decay=self.l2_regularization)

        values = self._convert_numpy_to_torch(values)

        n_epoch = 1
        flag = True
        while flag:
            y_pred = self.decoder(self.encoder(values))

            loss = self.loss(y_pred, values)

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

    def transform(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        X = self._convert_numpy_to_torch(X).T
        g = self.encoder(X)
        return g.cpu().detach().numpy().T

    def inverse_transform(self, g):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.
        """
        g = self._convert_numpy_to_torch(g).T
        u = self.decoder(g)
        return u.cpu().detach().numpy().T

    def reduce(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).

        .. note::

            Same as `transform`. Kept for backward compatibility.
        """
        return self.transform(X)

    def expand(self, g):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.

        .. note::

            Same as `inverse_transform`. Kept for backward compatibility.
        """
        return self.inverse_transform(g)

"""
Module for FNN-Autoencoders.
"""

import logging
import numpy as np
from .reduction import Reduction
from ..approximation import ANN

logger = logging.getLogger(__name__)


class AE(Reduction):
    """
    Feed-Forward AutoEncoder class (AE)

    :param int latent_dim: the dimension of the latent space
    :param list layers_encoder: ordered list with the number of neurons of
        each hidden layer for the encoder
    :param list layers_decoder: ordered list with the number of neurons of
        each hidden layer for the decoder
    :param str activation: activation function for the encoder
        ('relu', 'tanh', 'logistic', 'identity'). Default is 'tanh'.
    :param int max_iter: maximum number of training
        iterations (int) or desired tolerance on training loss (float).
    :param str solver: the solver for weight optimization ('adam', 'sgd',
        'lbfgs'). Default is 'adam'.
    :param float lr: the learning rate. Default is 0.001.
    :param float alpha: L2 regularization coefficient. Default is 0.
    :param int frequency_print: the frequency of printing during training.
        Default is 10.

    :Example:
        >>> from ezyrb import AE
        >>> low_dim = 5
        >>> ae = AE([400, low_dim], [low_dim, 400], 'tanh', 'tanh', 2000)
        >>> ae.fit(snapshots)
        >>> reduced_snapshots = ae.reduce(snapshots)
        >>> expanded_snapshots = ae.expand(reduced_snapshots)
    """

    def __init__(
        self,
        latent_dim,
        layers_encoder,
        layers_decoder,
        activation="tanh",
        max_iter=200,
        solver="adam",
        learning_rate_init=0.001,
        alpha=0,
        frequency_print=10,
        **kwargs,
    ):

        logger.debug(
            "Initializing AE with encoder layers=%s, decoder layers=%s",
            layers_encoder,
            layers_decoder,
        )

        if layers_encoder[-1] != layers_decoder[0]:
            logger.error(
                "Dimension mismatch: encoder output=%d, decoder input=%d",
                layers_encoder[-1],
                layers_decoder[0],
            )
            raise ValueError("Wrong dimension in encoder and decoder layers")

        if not isinstance(latent_dim, int):
            logger.error(
                "latent_dim should be an integer, got %s", type(latent_dim)
            )
            raise ValueError("latent_dim should be an integer")

        self.latent_dim = latent_dim
        self.layers_encoder = layers_encoder
        self.layers_decoder = layers_decoder
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.frequency_print = frequency_print
        self.loss_trend = []
        self.extra_kwargs = kwargs

        # Models
        self._autoencoder = None  # Full trained model
        self.encoder = None  # For encoding
        self.decoder = None  # For decoding

    def fit(self, values):
        """
        Build and train the autoencoder.

        :param numpy.ndarray values: the (training) values in the points.
        """
        logger.info("Starting AE training")
        values = values.T

        # Combine encoder and decoder layers
        combined_layers = (
            self.layers_encoder + [self.latent_dim] + self.layers_decoder
        )

        logger.debug(
            "Training full autoencoder with layers: %s", combined_layers
        )

        # Train full autoencoder: input -> latent -> reconstruction
        self._autoencoder = ANN(
            combined_layers,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            **self.extra_kwargs,
        )

        # Train to reconstruct input
        self._autoencoder.fit(values, values)
        self.loss_trend = self._autoencoder.loss_trend

        # Now create encoder and decoder and copy weights
        logger.debug("Creating encoder and decoder from trained autoencoder")

        # Create encoder
        self.encoder = ANN(
            self.layers_encoder,
            activation=self.activation,
            solver="adam",
            max_iter=1,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
        )
        # Create decoder
        self.decoder = ANN(
            self.layers_decoder,
            activation=self.activation,
            solver="adam",
            max_iter=1,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
        )
        # Dummy fit to initialize structure
        dummy_latent = np.zeros((values.shape[0], self.latent_dim))
        # Dummy fit to initialize structure
        self.decoder.fit(dummy_latent, values)
        self.encoder.fit(values, dummy_latent)

        # Copy encoder weights from autoencoder
        n_encoder_layers = len(self.encoder.model.coefs_)
        for i in range(n_encoder_layers):
            self.encoder.model.coefs_[i] = self._autoencoder.model.coefs_[
                i
            ].copy()
            self.encoder.model.intercepts_[i] = (
                self._autoencoder.model.intercepts_[i].copy()
            )

        # Copy decoder weights from autoencoder
        n_decoder_layers = len(self.decoder.model.coefs_)
        for i in range(n_decoder_layers):
            src_idx = len(self.encoder.model.coefs_) + i
            self.decoder.model.coefs_[i] = self._autoencoder.model.coefs_[
                src_idx
            ].copy()
            self.decoder.model.intercepts_[i] = (
                self._autoencoder.model.intercepts_[src_idx].copy()
            )

        print("dentro fit")
        print(values.shape)
        print(self._autoencoder.predict(values))
        red = self.encoder.predict(values)
        print(red.shape)
        full = self.decoder.predict(red)
        print(full.shape)
        print(self.decoder.predict(self.encoder.predict(values)))
        logger.info("AE training completed")

    def transform(self, X):
        """
        Reduces the given snapshots (encode).

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        logger.debug("Encoding %d snapshots", X.shape[0])
        if self.encoder is None:
            raise RuntimeError("Autoencoder not fitted yet")
        return self.encoder.predict(X.T).T

    def inverse_transform(self, g):
        """
        Projects a reduced to full order solution (decode).

        :param numpy.ndarray g: the latent variables.
        """
        logger.debug("Decoding %d latent vectors", g.shape[0])
        if self.decoder is None:
            raise RuntimeError("Autoencoder not fitted yet")

        if self.activation == "tanh":
            g = np.tanh(g)
        elif self.activation == "logistic":
            g = 1 / (1 + np.exp(-g))
        elif self.activation == "relu":
            g = np.maximum(0, g)
        elif self.activation == "identity":
            pass

        return self.decoder.predict(g.T).T

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

        :param numpy.ndarray g: the latent variables.

        .. note::

            Same as `inverse_transform`. Kept for backward compatibility.
        """
        return self.inverse_transform(g)

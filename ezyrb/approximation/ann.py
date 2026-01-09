"""
Module for Artificial Neural Network (ANN) Prediction.
"""

import logging
import numpy as np
from sklearn.neural_network import MLPRegressor
from .approximation import Approximation

logger = logging.getLogger(__name__)


class ANN(Approximation):
    """
    Feed-Forward Artifical Neural Network (ANN) using sklearn's MLPRegressor.

    :param list layers: ordered list with the number of neurons of each hidden
        layer.
    :param str activation: activation function for the hidden layers.
        Options: 'identity', 'logistic', 'tanh', 'relu' (default).
    :param str solver: the solver for weight optimization. Options: 'lbfgs',
        'sgd', 'adam' (default).
    :param int max_iter: maximum number of iterations. Default is 200.
    :param float tol: tolerance for the optimization. Default is 1e-4.
    :param float learning_rate_init: initial learning rate (only for 'sgd'
        or 'adam'). Default is 0.001.
    :param float alpha: L2 penalty (regularization term) parameter.
        Default is 0.0001.
    :param int frequency_print: the frequency in terms of epochs to print
        training progress. Default is 10.
    :param int random_state: random state for reproducibility. Default is None.
    :param bool early_stopping: whether to use early stopping to terminate
        training when validation score is not improving. Default is False.
    :param float validation_fraction: proportion of training data to set aside
        as validation set for early stopping. Default is 0.1.

    :Example:
        >>> import ezyrb
        >>> import numpy as np
        >>> x = np.random.uniform(-1, 1, size=(4, 2))
        >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
        >>> ann = ezyrb.ANN([10, 5], activation='tanh', max_iter=20000)
        >>> ann.fit(x, y)
        >>> y_pred = ann.predict(x)
        >>> print(y)
        >>> print(y_pred)
        >>> print(len(ann.loss_trend))
        >>> print(ann.loss_trend[-1])

    .. note::
        This module provides a wrapper around sklearn's MLPRegressor for
        multidimensional function approximation using feed-forward neural
        networks. It is not intended for deep learning tasks, but rather for
        approximating functions based on given data points. For more advanced
        deep learning applications, consider using dedicated libraries such as
        :ref:`PINA`.
    """

    def __init__(
        self,
        layers,
        activation="tanh",
        max_iter=200,
        solver="adam",
        learning_rate_init=0.001,
        alpha=0.0001,
        frequency_print=10,
        **kwargs,
    ):
        logger.debug(
            "Initializing ANN with layers=%s, activation=%s, "
            "solver=%s, max_iter=%d, lr=%f, alpha=%f",
            layers,
            activation,
            solver,
            max_iter,
            learning_rate_init,
            alpha,
        )

        self.layers = layers
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.frequency_print = frequency_print
        self.extra_kwargs = kwargs

        self.model = None
        self.loss_trend = []

        logger.info("ANN initialized with sklearn MLPRegressor")

    def fit(self, points, values):
        """
        Build the ANN given 'points' and 'values' and perform training.

        :param numpy.ndarray points: the coordinates of the given (training)
            points.
        :param numpy.ndarray values: the (training) values in the points.
        """
        logger.debug(
            "Fitting ANN with points shape: %s, values shape: %s",
            points.shape,
            values.shape,
        )

        # Create the MLPRegressor model
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.layers),
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            verbose=False,
            **self.extra_kwargs,
        )

        # Custom training loop to track loss and print progress
        self.loss_trend = []

        # For sklearn, we need to do partial fitting to track loss
        # We'll use the standard fit but access loss_curve_ afterwards
        logger.info("Starting ANN training")

        if self.frequency_print > 0:
            # Monkey patch to capture loss during training
            original_fit = self.model.fit

            def fit_with_logging(X, y):
                result = original_fit(X, y)
                if hasattr(self.model, "loss_curve_"):
                    self.loss_trend = list(self.model.loss_curve_)
                    for i, loss in enumerate(self.loss_trend):
                        if (
                            i == 0
                            or i == len(self.loss_trend) - 1
                            or (i + 1) % self.frequency_print == 0
                        ):
                            print(f"[epoch {i+1:6d}]\t{loss:e}")
                return result

            fit_with_logging(points, values)
        else:
            self.model.fit(points, values)
            if hasattr(self.model, "loss_curve_"):
                self.loss_trend = list(self.model.loss_curve_)

        logger.info(
            "ANN training completed after %d iterations", self.model.n_iter_
        )
        if self.loss_trend:
            logger.debug("Final loss: %f", self.loss_trend[-1])

        return self

    def predict(self, new_point):
        """
        Evaluate the ANN at given 'new_points'.

        :param array_like new_points: the coordinates of the given points.
        :return: the predicted values via the ANN.
        :rtype: numpy.ndarray
        """
        logger.debug(
            "Predicting with ANN for %d points",
            np.atleast_2d(new_point).shape[0],
        )
        new_point = np.atleast_2d(new_point)
        y_new = self.model.predict(new_point)
        return y_new

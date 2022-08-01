"""
Module for Artificial Neural Network (ANN) Prediction.
"""

import torch
from torch import nn
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import INOUT, IN
from .approximation import Approximation

class ANN(Approximation):
    """
    Feed-Forward Artifical Neural Network class (ANN).

    :param list layers: ordered list with the number of neurons of each
        hidden layer.
    :param torch.nn.modules.activation function: activation function at each
        layer, except for the output layer at with Identity is considered by
        default. A single activation function can be passed or a list of
        them of length equal to the number of hidden layers.
    :param list stop_training: list with the maximum number of training
        iterations (int) and/or the desired tolerance on the training loss
        (float).
    :param torch.nn.Module loss: loss definition (Mean Squared if not
        given).

    :Example:
        >>> import ezyrb
        >>> import numpy as np
        >>> import torch.nn as nn
        >>> x = np.random.uniform(-1, 1, size =(4, 2))
        >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
        >>> ann = ezyrb.ANN([10, 5], nn.Tanh(), [20000,1e-5])
        >>> ann.fit(x, y)
        >>> y_pred = ann.predict(x)
        >>> print(y)
        >>> print(y_pred)
        >>> print(len(ann.loss_trend))
        >>> print(ann.loss_trend[-1])
    """
    def __init__(self, layers, function, stop_training, loss=None):

        if loss is None:
            loss = torch.nn.MSELoss()

        if not isinstance(function, list):  # Single activation function passed
            function = [function] * (len(layers))

        if not isinstance(stop_training, list):
            stop_training = [stop_training]

        self.layers = layers
        self.function = function
        self.loss = loss
        self.stop_training = stop_training

        self.loss_trend = []
        self.model = None
        self.optimizer = None

    def _convert_numpy_to_torch(self, array):
        """
        Converting data type.

        :param numpy.ndarray array: input array.
        :return: the tensorial counter-part of the input array.
        :rtype: torch.Tensor.
        """
        return torch.from_numpy(array).float()

    def _convert_torch_to_numpy(self, tensor):
        """
        Converting data type.

        :param torch.Tensor tensor: input tensor.
        :return: the vectorial counter-part of the input tensor.
        :rtype: numpy.ndarray.
        """
        return tensor.detach().numpy()

    def _build_model(self, points, values):
        """
        Build the torch model.
        Considering the number of neurons per layer (self.layers), a
        feed-forward NN is defined:
        -  activation function from layer i>=0 to layer i+1:
           self.function[i]; activation function at the output layer:
           Identity (by default).
        :param numpy.ndarray points: the coordinates of the given (training)
        points.
        :param numpy.ndarray values: the (training) values in the points.
        """
        layers = self.layers.copy()
        layers.insert(0, points.shape[1])
        layers.append(values.shape[1])
        layers_torch = []
        for i in range(len(layers) - 2):
            layers_torch.append(nn.Linear(layers[i], layers[i + 1]))
            layers_torch.append(self.function[i])
        layers_torch.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layers_torch)

    @task(target_direction=INOUT)
    def fit(self, points, values):
        """
        Build the ANN given 'points' and 'values' and perform training.

        Training procedure information:
            -  optimizer: Adam's method with default parameters (see, e.g.,
               https://pytorch.org/docs/stable/optim.html);
            -  loss: self.loss (if none, the Mean Squared Loss is set by
               default).
            -  stopping criterion: the fulfillment of the requested tolerance on
               the training loss compatibly with the prescribed budget of
               training iterations (if type(self.stop_training) is list); if
               type(self.stop_training) is int or type(self.stop_training) is
               float, only the number of maximum iterations or the accuracy
               level on the training loss is considered as the stopping rule,
               respectively.

        :param numpy.ndarray points: the coordinates of the given (training)
            points.
        :param numpy.ndarray values: the (training) values in the points.
        """

        self._build_model(points, values)
        self.optimizer = torch.optim.Adam(self.model.parameters())

        points = self._convert_numpy_to_torch(points)
        values = self._convert_numpy_to_torch(values)

        n_epoch = 1
        flag = True
        while flag:
            y_pred = self.model(points)
            loss = self.loss(y_pred, values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_trend.append(loss.item())
            for criteria in self.stop_training:
                if isinstance(criteria, int):  # stop criteria is an integer
                    if n_epoch == criteria:
                        flag = False
                elif isinstance(criteria, float):  # stop criteria is float
                    if loss.item() < criteria:
                        flag = False

            n_epoch += 1

    @task(returns=np.ndarray, target_direction=IN)
    def predict(self, new_point, scaler_red):
        """
        Evaluate the ANN at given 'new_points'.

        :param array_like new_points: the coordinates of the given points.
        :return: the predicted values via the ANN.
        :rtype: numpy.ndarray
        """
        new_point = self._convert_numpy_to_torch(np.array(new_point))
        y_new = self.model(new_point)
        predicted_red_sol = np.atleast_2d(self._convert_torch_to_numpy(y_new))
        if scaler_red:  # rescale modal coefficients
            predicted_red_sol = scaler_red.inverse_transform(
                predicted_red_sol)
        predicted_red_sol = predicted_red_sol.T
        return predicted_red_sol
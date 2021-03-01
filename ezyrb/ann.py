"""
Module for Artificial Neural Network (ANN) Prediction.
"""
import torch
import torch.nn as nn
import numpy as np
from .approximation import Approximation

class ANN(Approximation):
    """
    Feed-Forward Artifical Neural Network (ANN).
    :param int trained_epoch: number of already trained iterations.
    :param criterion: Loss definition (Mean Squared). 
    :type criterion: torch.nn.Module.
    
    Example:
    >>> import ezyrb
    >>> import numpy as np
    >>> x = np.random.uniform(-1, 1, size =(4, 2))
    >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
    >>> ann = ezyrb.ANN()
    >>> ann.fit(x, y)
    >>> y_pred = ann.predict(x)
    >>> print(y)
    >>> print(y_pred)
    """
        
    def __init__(self, layers, function, stop_training, loss=None):

        if loss is None: loss = torch.nn.MSELoss
        if optimizer is None: optimizer = torch.optim.Adam
        if not isinstance(function, list): # Single activation function passed
            function = [function] * (len(layers)-1)


        self.layers = layers
        self.function = function
        self.loss = loss
        self.stop_training = stop_training

        self.loss_trend = []   
        self.model = None

    def _convert_numpy_to_torch(self, array):
        """
        Converting data type.
        TODO: type should be not forced to `float`
        """
        return torch.from_numpy(array).float()

    def _convert_torch_to_numpy(self, tensor):
        """
        Converting data type.
        """
        return tensor.detach().numpy()

    def _build_model(self):
        """
        Build the torch model
        """
        self.layers.insert(0, points.shape[1])
        self.layers.append(values.shape[1]
        layers = []
        for i in range(len(layers)-1):
            layers.append(nn.Linear(self.layers[i], self.layers[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.layers[-2], self.layers[-1]))
        self.model = nn.Sequential(*layers)
        
    def fit(self, points, values):
        """
        Build the ANN given 'points' and 'values' and perform training.
        
        Given the number of neurons per layer, a feed-forward NN is defined. 
        By default: 
           - niter, number of training iterations: 20000;
           - activation function in each inner layer: Tanh; activation function
             at the output layer: Identity;
           - optimizer: Adam's method with default parameters 
             (see, e.g., https://pytorch.org/docs/stable/optim.html);
           - loss: Mean Squared Loss.
        
        :param numpy.ndarray points: the coordinates of the given (training) points.
        :param numpy.ndarray values: the (training) values in the points.
        :return the training loss value at termination (after niter iterations).
        :rtype: float.
        """

        if self.model is None:
           self._build_model() 

        optimizer = torch.optim.Adam(self.model.parameters())
 
        points = self._convert_numpy_to_torch(points)
        values = self._convert_numpy_to_torch(values)

        n_epoch = 0

        while True:
            y_pred = self.model(points)
            loss = self.criterion(y_pred, values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_trend.append(loss.item())

            if isinstance(self.stop_training, int): # stop criteria is an integer
                if n_epoch == self.stop_training: break
            elif isinstance(self.stop_training, float): # stop criteria is float
                if loss.item() < self.stop_training: break 

            n_epoch += 1
                
    
    def predict(self, new_point):
        """
        Evaluate the ANN at given 'new_points'.
        
        :param array_like new_points: the coordinates of the given points.
        :return: the predicted values via the ANN.
        :rtype: numpy.ndarray
        """
        new_point = self._convert_numpy_to_torch(new_point)
        y_new = self.model(new_point)
        return self._convert_torch_to_numpy(y_new)

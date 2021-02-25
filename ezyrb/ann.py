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
    : param int trained_epoch: number of already trained iterations.
    : param criterion: Loss definition (Mean Squared). 
    : type criterion: torch.nn.modules.loss.MSELoss.
    
    Example:
<<<<<<< HEAD
    >>> import ezyrb
    >>> import numpy as np
    >>> x = np.random.uniform(-1, 1, size =(4, 2))
    >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
    >>> ann = ezyrb.ANN()
=======
    >>> import numpy as np
    >>> from ann import ANN
    >>> x = np.random.uniform(-1, 1, size = (4, 2))
    >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
    >>> ann = ANN()
>>>>>>> 359dc0ccc1c6d01703d0a34655ec5cfccd1f725a
    >>> ann.fit(x, y)
    >>> y_pred = ann.predict(x)
    >>> print(y)
    >>> print(y_pred)
    """
        
    def __init__(self):
<<<<<<< HEAD
        self.trained_epoch = 0   
=======
        # number of already trained iterations
        self.trained_epoch = 0   
        # the Mean Squared Loss is considered
>>>>>>> 359dc0ccc1c6d01703d0a34655ec5cfccd1f725a
        self.criterion = torch.nn.MSELoss()
        
        
    def fit(self, points, values):
        """
        Build the ANN given 'points' and 'values' and perform training.
        
        Given the number of neurons per layer, a feed-forward NN is defined. 
        By default: 
<<<<<<< HEAD
           - niter, number of training iterations: 20000;
=======
           - niter, number of training iterations: 5000;
>>>>>>> 359dc0ccc1c6d01703d0a34655ec5cfccd1f725a
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
<<<<<<< HEAD
        layers = [points.shape[1], 10, 5, values.shape[1]]
        niter = 20000
=======
        # ordered list with the number of neurons per layer (to be modified by the user)
        # (i.e., layers[i] = number of neurons in the layer i)
        layers = [points.shape[1], 10, 5, values.shape[1]]
        niter = 5000
>>>>>>> 359dc0ccc1c6d01703d0a34655ec5cfccd1f725a
        arguments = []
        for i in range(len(layers)-2):
            arguments.append(nn.Linear(layers[i], layers[i+1]))
            arguments.append(nn.Tanh())
        arguments.append(nn.Linear(layers[len(layers)-2], layers[len(layers)-1]))
        arguments.append(nn.Identity())
<<<<<<< HEAD
        self.model = nn.Sequential(*arguments)
=======
        # ANN structural model definition
        self.model = nn.Sequential(*arguments)
        # setting of the optimization solver (Adam)
>>>>>>> 359dc0ccc1c6d01703d0a34655ec5cfccd1f725a
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        points = torch.from_numpy(points).float()
        values = torch.from_numpy(values).float()
        for epoch in range(niter):
<<<<<<< HEAD
            y_pred = self.model(points)
            loss = self.criterion(y_pred, values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.trained_epoch += niter
        return loss.item()
    
    def predict(self, new_point):
=======
            # forward propagation (net evaluation at 'points')
            y_pred = self.model(points)
            # compute the training loss
            loss = self.criterion(y_pred, values)
            # zero the gradients
            self.optimizer.zero_grad()
            # loss values extraction (type: float)
            old_loss = loss.item() 
            # perform a backward propagation
            loss.backward()
            # parameters update 
            self.optimizer.step()
        # update of the number of trained iterations
        self.trained_epoch += niter
        return loss.item()
    
    def predict(self,new_point):
>>>>>>> 359dc0ccc1c6d01703d0a34655ec5cfccd1f725a
        """
        Evaluate the ANN at given 'new_points'.
        
        :param array_like new_points: the coordinates of the given points.
        :return: the predicted values via the ANN.
        :rtype: numpy.ndarray
        """
        new_point = np.array(new_point)
        new_point = torch.from_numpy(new_point).float()
        y_new = self.model(new_point)
        return y_new.detach().numpy()
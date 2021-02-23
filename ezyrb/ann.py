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
    
    Example:
    >>> import ezyrb
    >>> import torch
    >>> import torch.nn as nn
    >>> import numpy as np
    >>> 
    >>> x = np.random.uniform(-1, 1, size=(4, 2))
    >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1]**3)]).T
    >>> ann = ezyrb.ANN()
    >>> ann.fit(x,y)
    >>> y_pred = ann.predict(x)
    >>> print(y)
    >>> print(y_pred)
    """
        
    def __init__(self):
        self.trained_epoch = 0   
        # number of already trained iterations
        self.criterion = torch.nn.MSELoss()
        # the Mean Squared Loss is considered
        
    def fit(self, points, values):
        """
        Build the ANN given 'points' and 'values' and perform training.
        
        Given the number of neurons per layer, a feed-forward NN is defined. 
        By default: 
           - niter, number of training iterations: 5000;
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
        layers = [points.shape[1], 10, 5, values.shape[1]]
        # ordered list with the number of neurons per layer (to be modified by the user)
        # (i.e., layers[i]=number of neurons in the layer i)
        niter=5000
        arguments = []
        for i in range(len(layers)-2):
            arguments.append(nn.Linear(layers[i], layers[i+1]))
            arguments.append(nn.Tanh())
        arguments.append(nn.Linear(layers[len(layers)-2], layers[len(layers)-1]))
        arguments.append(nn.Identity())
        self.model = nn.Sequential(*arguments)
        # ANN structural model definition
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # setting of the optimization solver (Adam)
        points = torch.from_numpy(points).float()
        values = torch.from_numpy(values).float()
        for epoch in range(niter):
            y_pred = self.model(points)
            # forward propagation (net evaluation at 'points')
            loss = self.criterion(y_pred, values)
            # compute the training loss
            self.optimizer.zero_grad()
            # zero the gradients
            old_loss = loss.item() 
            # loss values extraction (type: float)
            loss.backward()
            # perform a backward propagation
            self.optimizer.step()
            # parameters update           
        self.trained_epoch += niter
        # update of the number of trained iterations
        return loss.item()
    
    def predict(self,new_point):
        """
        Evaluate the ANN at given 'new_points'.
        
        :param (list or numpy.ndarray) new_points: the coordinates of the given points.
        :return: the predicted values via the ANN.
        :rtype: numpy.ndarray
        """
        new_point=np.array(new_point)
        new_point = torch.from_numpy(new_point).float()
        y_new=self.model(new_point)
        return y_new.detach().numpy()
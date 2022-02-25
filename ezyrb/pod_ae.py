"""
Module for FNN-Autoencoders.
"""

import torch
import torch.nn as nn
from .reduction import Reduction
from .ae import AE
from .pod import POD


class PODAE(POD, AE):
    """
    Feed-Forward AutoEncoder class with POD (AE)
    """
    def __init__(self, pod, ae):
        self.pod = pod
        self.ae = ae

    def fit(self, X):
        """
        """
        self.pod.fit(X)
        coefficients = self.pod.reduce(X)
        self.ae.fit(coefficients)

    def reduce(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        coeff = self.pod.reduce(X)
        coeff = self._convert_numpy_to_torch(coeff).T
        g = self.ae.encoder(coeff)
        return g.cpu().detach().numpy().T

    def expand(self, g):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.
        """
        g = self._convert_numpy_to_torch(g).T
        u = self.ae.decoder(g)
        u = u.cpu().detach().numpy().T
        return self.pod.expand(u)

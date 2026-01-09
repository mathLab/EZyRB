"""
Module for FNN-Autoencoders.
"""

from .ae import AE
from .pod import POD


class PODAE(POD, AE):
    """
    Combined POD and AutoEncoder reduction class.

    This class first applies POD to reduce the dimensionality, then uses
    an autoencoder for further reduction in the latent space.

    :param POD pod: The POD instance for initial reduction.
    :param AE ae: The AutoEncoder instance for latent space reduction.
    """

    def __init__(self, pod, ae):
        """
        Initialize the PODAE reducer.

        :param POD pod: The POD instance.
        :param AE ae: The AutoEncoder instance.
        """
        self.pod = pod
        self.ae = ae

    def fit(self, X):
        """
        Fit the PODAE on the snapshots.

        First applies POD, then trains the autoencoder on POD coefficients.

        :param numpy.ndarray X: The input snapshots matrix (stored by column).
        """
        self.pod.fit(X)
        coefficients = self.pod.transform(X)
        self.ae.fit(coefficients)

    def transform(self, X):
        """
        Reduces the given snapshots.

        :param numpy.ndarray X: the input snapshots matrix (stored by column).
        """
        coeff = self.pod.reduce(X)
        g = self.ae.encoder.predict(coeff)
        return g.T

    def inverse_transform(self, g):
        """
        Projects a reduced to full order solution.

        :param: numpy.ndarray g the latent variables.
        """
        u = self.ae.decoder.predict(g.T)
        u = u.T
        return self.pod.expand(u)

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

"""Module for the Approximation abstract class"""

from abc import ABC, abstractmethod


class Approximation(ABC):
    """
    The abstract `Approximation` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """
    @abstractmethod
    def fit(self, points, values):
        """Abstract `fit`"""

    @abstractmethod
    def predict(self, new_point):
        """Abstract `predict`"""

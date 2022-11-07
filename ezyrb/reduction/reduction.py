"""Module for the Reduction abstract class."""

from abc import ABC, abstractmethod


class Reduction(ABC):
    """
    The abstract `Approximation` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """
    @abstractmethod
    def fit(self):
        """Abstract `fit`"""

    @abstractmethod
    def transform(self):
        """Abstract `transform`"""

    @abstractmethod
    def inverse_transform(self):
        """Abstract `inverse_transform`"""

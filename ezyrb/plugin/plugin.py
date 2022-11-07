"""Module for the Plugin abstract class."""

from abc import ABC, abstractmethod


class Plugin(ABC):
    """
    The abstract `Approximation` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """
    @abstractmethod
    def perform(self):
        """Abstract `perform`"""

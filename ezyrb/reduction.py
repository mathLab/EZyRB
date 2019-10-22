"""
Module for the Reduction abstract class
"""

from abc import ABC, abstractmethod


class Reduction(ABC):
    @abstractmethod
    def reduce(self):
        pass

    @abstractmethod
    def expand(self):
        pass

"""
Module for the Approximation abstract class
"""

from abc import ABC, abstractmethod


class Approximation(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

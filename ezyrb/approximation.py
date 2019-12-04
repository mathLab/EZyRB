"""
Module for the Approximation abstract class
"""
from abc import ABC, abstractmethod

class Approximation(ABC):
    @abstractmethod
    def fit(self, points, values):
        pass

    @abstractmethod
    def predict(self, new_point):
        pass

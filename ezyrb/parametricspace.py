"""
Module with the base class for a generic space.
"""
import pickle


class ParametricSpace(object):
    """
    Abstract class.
    """

    def __init__(self):
        raise NotImplemented

    def __call__(self, value):
        """
        Abstract method to approximate the value of a generic point.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplemented

    def save(self, filename):
        """
        """
        with open(filename, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """
        """
        with open(filename, 'r') as f:
            return pickle.load(f)

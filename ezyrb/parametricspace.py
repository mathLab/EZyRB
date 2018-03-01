"""
Module with the base class for a generic space.
"""
import pickle


class ParametricSpace(object):
    """
    Abstract class.
    """

    def __init__(self):
        raise NotImplementedError

    def __call__(self, value):
        """
        Abstract method to approximate the value of a generic point.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError

    def save(self, filename):
        """
        Save the space on file.

        :param str filename: the filename where the space has to be saved.
        """
        with open(filename, 'w') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """
        Load the space from a file.

        :param str filename: the filename where the space has been saved.
        :return: the space instance.
        :rtype: ParametricSpace
        """
        with open(filename, 'r') as f:
            return pickle.load(f)

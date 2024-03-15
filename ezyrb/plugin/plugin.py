"""Module for the Plugin abstract class."""

from abc import ABC


class Plugin(ABC):
    """
    The abstract `Approximation` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """
    def fit_preprocessing(self, rom):
        """ Void """
        pass

    def before_fit_reduction(self, rom):
        """ Void """
        pass

    def after_fit_reduction(self, rom):
        """ Void """
        pass
    
    def before_fit_approximation(self, rom):
        """ Void """
        pass

    def after_fit_approximation(self, rom):
        """ Void """
        pass

    def fit_postprocessing(self, rom):
        """ Void """
        pass

    def predict_preprocessing(self, rom):
        """ Void """
        pass

    def predict_postprocessing(self, rom):
        """ Void """
        pass




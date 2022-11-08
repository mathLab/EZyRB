"""Module for the Plugin abstract class."""

from abc import ABC


class Plugin(ABC):
    """
    The abstract `Approximation` class.

    All the classes that implement the input-output mapping should be inherited
    from this class.
    """
    def fom_preprocessing(self, rom):
        """ Void """
        pass

    def rom_preprocessing(self, rom):
        """ Void """
        pass

    def rom_postprocessing(self, rom):
        """ Void """
        pass

    def fom_postprocessing(self, rom):
        """ Void """
        pass

""" Module for parameter object """
import numpy as np

class Parameter:

    def __init__(self, values):
        self.values = values

    @property
    def values(self):
        """ Get the snapshot values. """
        return self._values

    @values.setter
    def values(self, new_values):
        if np.asarray(new_values).ndim != 1:
            raise ValueError('only 1D array are usable as parameter.')
        self._values = new_values
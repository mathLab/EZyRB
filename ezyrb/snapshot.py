""" Module for discretized solution object """

import numpy as np
import matplotlib.pyplot as plt


class Snapshot:

    def __init__(self, values, space=None):
        self.values = values
        self.space = space

    @property
    def values(self):
        """ Get the snapshot values. """
        return self._values

    @values.setter
    def values(self, new_values):
        if hasattr(self, 'space') and self.space is not None:
            if len(self.space) != len(new_values):
                raise ValueError('invalid ndof for the current space.')

        self._values = new_values

    @property
    def space(self):
        """ Get the snapshot space. """
        return self._space

    @space.setter
    def space(self, new_space):
        if hasattr(self, 'values') and self.values is not None:
            if new_space is not None and len(self.values) != len(new_space):
                raise ValueError('invalid ndof for the current space.')

        self._space = new_space

    @property
    def flattened(self):
        """ return the values in 1D array """
        return self.values.flatten()

    def plot(self):
        """ Plot the snapshot, if possible. """

        if self.space is None:
            print('No space set, unable to plot.')
            return

        if np.asarray(self.space).ndim == 1:
            plt.plot(self.space, self.values)
        else:
            raise NotImplementedError

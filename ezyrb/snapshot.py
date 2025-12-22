""" Module for discretized solution object """

import numpy as np
import matplotlib.pyplot as plt


class Snapshot:
    """
    Class for representing a discretized solution snapshot.
    
    This class encapsulates solution values and their spatial coordinates,
    providing methods for manipulation and visualization.
    
    :param array_like values: The solution values.
    :param array_like space: The spatial coordinates corresponding to the values.
        Default is None.
    
    :Example:
    
        >>> import numpy as np
        >>> from ezyrb import Snapshot
        >>> space = np.linspace(0, 1, 50)
        >>> values = np.sin(2 * np.pi * space)
        >>> snap = Snapshot(values, space)
        >>> print(snap.values.shape)
        (50,)
        >>> print(snap.space.shape)
        (50,)
    """

    def __init__(self, values, space=None):
        """
        Initialize a Snapshot object.
        
        :param values: The solution values. Can be a Snapshot instance or
            an array-like object.
        :param space: The spatial coordinates. Default is None.
        """
        if isinstance(values, Snapshot):
            self.values = values.values
            self.space = values.space
        else:
            self.values = values
            self.space = space

    @property
    def values(self):
        """ 
        Get the snapshot values.
        """
        return self._values

    @values.setter
    def values(self, new_values):
        """
        Set the snapshot values with validation.
        
        :param array_like new_values: The new snapshot values.
        :raises ValueError: If the length of new values doesn't match the space.
        """
        if hasattr(self, 'space') and self.space is not None:
            if len(self.space) != len(new_values):
                raise ValueError('invalid ndof for the current space.')

        self._values = new_values

    @property
    def space(self):
        """ 
        Get the snapshot space.
        """
        return self._space

    @space.setter
    def space(self, new_space):
        """
        Set the snapshot space with validation.
        
        :param array_like new_space: The new spatial coordinates.
        :raises ValueError: If the length of new space doesn't match the values.
        """
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

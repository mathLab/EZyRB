"""
Module for the manipulation of the solution
"""
from ezyrb.filehandler import FileHandler
import numpy as np


class Snapshots(object):
    """
    :param str output_name: the name of the output to extract from the solution
        files; it has to be the same for all the solution files.
    :param str weight_name: the name of the weights.
    :param str dformat: it allows to choose if the solutions are stored by
        points or by cells; allowed strings are 'cell' or 'point'. Default
        value is 'cell'.

    :cvar str _o_name: the output name.
    :cvar str _w_name: the weight name.
    :cvar numpy.ndarray _values: the solutions extracted from files, stored by
        column.
    :cvar numpy.ndarray _weighted: the weighted solutions extracted from files,
        stored by column.
    :cvar numpy.ndarray or int _weights: the weights extracted from a file.
    :cvar str _dformat: the flag to select the data type.
    :cvar list(str) _files: the files from which to extract the solutions.

    """

    def __init__(self, output_name, weight_name=None, dformat='cell'):

        self._o_name = output_name
        self._w_name = weight_name

        self._values = np.ndarray(shape=(0, 0))
        self._weighted = np.ndarray(shape=(0, 0))
        self._weights = 1.0

        self._dformat = dformat

        self._files = []

    def __getitem__(self, val):
        ret = Snapshots(self._o_name, self._w_name, self._dformat)
        ret._values = self._values[:, val]
        ret._weighted = self._weighted[:, val]
        ret._weights = self._weights
        if isinstance(val, list):
            ret._files = [self._files[i] for i in val]
        else:
            ret._files = self._files[val]

        return ret

    @property
    def files(self):
        """
        The list of file from which to extract the snapshots.

        :type: list(str)
        """
        return self._files

    @property
    def values(self):
        """
        The snapshots; it is a *n* x *m* matrix that contains the snapshots
        stored by column; *n* is the output dimension, *m* is the number of
        snapshots.

        :type: numpy.ndarray
        """
        return self._values

    @property
    def weighted(self):
        """
        The weighted snapshots; it is a *n* x *m* matrix that contains the
        snapshots stored by column; *n* is the output dimension, *m* is the
        number of snapshots.

        :type: numpy.ndarray
        """
        return self._weighted

    @property
    def weights(self):
        """
        The weights extracted.

        :type: int or array_like
        """
        return self._weights

    @property
    def size(self):
        """
        The number of the snapshots.

        :type: int
        """
        return self.values.shape[1]

    @property
    def dimension(self):
        """
        The dimension of the snapshots.

        :type: int
        """
        return self.values.shape[0]

    def append(self, filename):
        """
        Append the `filename` to the list of the solution files; update the
        values and the weighted values with the extracted solution.

        :param str filename: the name of file to append to the solution files.
        """
        if not isinstance(filename, str):
            raise TypeError

        if not self.values.size and self._w_name:

            self._weights = FileHandler(filename).get_dataset(
                self._w_name, self._dformat)

        self._files.append(filename)

        v = FileHandler(filename).get_dataset(self._o_name, self._dformat)

        self._append_values(v)
        self._append_weighted(v)

    def _append_values(self, values):
        """
        Private method to append an output.

        :param array_like values: the output.
        """

        array = np.asarray(values).reshape(-1, 1)
        try:
            self._values = np.append(self._values, array, 1)
        except ValueError:
            self._values = array

    def _append_weighted(self, values):
        """
        Private method to append a weighted output.

        :param array_like values: the output.
        """
        print(values.shape)
        array = np.dot(np.sqrt(self._weights), values.T).reshape(-1, 1)
        try:
            self._weighted = np.append(self._weighted, array, 1)
        except ValueError:
            self._weighted = array

"""
Module to handle Matlab files.
"""
import os
import numpy as np
import scipy.io as sio


class MatlabHandler(object):
    """
    Matlab format file handler class.
    You are NOT supposed to call directly this class constructor (use
    :class:`.FileHandler` constructor instead).

    :cvar str _filename: name of file to handle.
    :cvar dict _cached_data: private attribute to save a copy of last data
        processed; it allows to reduce IO operation on the same file. Initial
        value is None.

    """

    def __init__(self, filename):

        self._filename = filename
        self._cached_data = None

    def get_dataset(self, output_name):
        """
        Method to parse the `filename`. It returns a matrix with all the values
        of the chosen output.

        :param str output_name: name of the output of interest inside the
            mat file.

        :return: a *n_points*-by-*n_components* matrix containing the values of
            the chosen output.
        :rtype: numpy.ndarray
        """
        if self._cached_data is None:
            if not os.path.isfile(self._filename):
                raise RuntimeError("{0!s} doesn't exist".format(
                    os.path.abspath(self._filename)))

            self._cached_data = sio.loadmat(self._filename)

        if not output_name in self._cached_data:
            raise RuntimeError("File has no " + output_name + " field.")

        return np.array(self._cached_data[output_name])

    def set_dataset(self, output_values, output_name):
        """
        Writes to filename the given output. `output_values` is a matrix that
        contains the new values of output to write, `output_name` is a string
        that indicates name of output to write.

        :param numpy.ndarray output_values: it is a
            *n_points*-by-*n_components* matrix containing the output values.
        :param str output_name: name of the output.
        """
        sio.savemat(self._filename, {output_name: output_values})
        self._cached_data = None

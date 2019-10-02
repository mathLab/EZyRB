"""
Module for the snapshots database collected during the Offline stage
"""
import numpy

class Database(object):

    def __init__(self, parameters=None, snapshots=None, scaler_parameters=None, scaler_snapshots=None):
        """
        Database class

        :param array_like parameters: the input parameters 
        :param array_like snapshots: the input snapshots
        :param Scale scaler_parameters: the scaler for the parameters. Default
            is None meaning no scaling.
        :param Scale scaler_snapshots: the scaler for the snapshots. Default is
            None meaning no scaling.
        """
        self._parameters = None
        self._snapshots = None
        self.scaler_parameters = scaler_parameters
        self.scaler_snapshots = scaler_snapshots

        # if only parameters or snapshots are provided
        if (parameters is None) ^ (snapshots is None):
            raise RuntimeError

        if not (parameters is None) and not(snapshots is None):
            self.add(parameters, snapshots)

    @property
    def parameters(self):
        """
        The matrix containing the input parameters (by row).

        :rtype: numpy.ndarray
        """
        if self.scaler_parameters:
            return self.scaler_parameters(self._parameters)
        else:
            return self._parameters

    @property
    def snapshots(self):
        """
        The matrix containing the snapshots (by row).

        :rtype: numpy.ndarray
        """
        if self.scaler_snapshots:
            return self.scaler_snapshots(self._snapshots)
        else:
            return self._snapshots


    def add(self, parameters, snapshots):
        """
        """
        
        if len(parameters) is not len(snapshots):
            raise RuntimeError('Different number of parameters and snapshots.')
        if self._parameters is None and self._snapshots is None:
            self._parameters = parameters
            self._snapshots = snapshots
        else:
            self._parameters = np.vstack([self._parameters, parameters])
            self._snapshots = np.vstack([self._snapshots, snapshots])

        return self

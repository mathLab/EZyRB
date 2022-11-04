"""Module for the snapshots database collected during the Offline stage."""

import numpy as np

from .parameter import Parameter
from .snapshot import Snapshot

class Database():
    """
    Database class

    :param array_like parameters: the input parameters
    :param array_like snapshots: the input snapshots
    :param Scale scaler_parameters: the scaler for the parameters. Default
        is None meaning no scaling.
    :param Scale scaler_snapshots: the scaler for the snapshots. Default is
        None meaning no scaling.
    :param array_like space: the input spatial data
    """
    def __init__(self, parameters=None, snapshots=None):
        self._pairs = []

        if parameters is None and snapshots is None:
            return

        if len(parameters) != len(snapshots):
            raise ValueError

        for param, snap in zip(parameters, snapshots):
            self.add(Parameter(param), Snapshot(snap))

    @property
    def parameters_matrix(self):
        """
        The matrix containing the input parameters (by row).

        :rtype: numpy.ndarray
        """
        return np.asarray([pair[0].values for pair in self._pairs])

    @property
    def snapshots_matrix(self):
        """
        The matrix containing the snapshots (by row).

        :rtype: numpy.ndarray
        """
        return np.asarray([pair[1].flattened for pair in self._pairs])

    def __getitem__(self, val):
        """
        This method returns a new Database with the selected parameters and
        snapshots.

        .. warning:: The new parameters and snapshots are a view of the
            original Database.
        """
        view = Database()
        view._pairs = self._pairs[val]
        return view

    def __len__(self):
        """
        This method returns the number of snapshots.

        :rtype: int
        """
        return len(self._pairs)

    def add(self, parameter, snapshot):
        """
        Add (by row) new sets of snapshots and parameters to the original
        database.

        :param Parameter parameter: the parameter to add.
        :param Snapshot snapshot: the snapshot to add.
        """
        if not isinstance(parameter, Parameter):
            raise ValueError

        if not isinstance(snapshot, Snapshot):
            raise ValueError

        self._pairs.append((parameter, snapshot))        

        return self


    # def split(self, chunks, seed=None):
    #     """

    #     >>> db = Database(...)
    #     >>> train, test = db.split([0.8, 0.2])
        
    #     """
    #     print(chunks)
    #     if all(isinstance(n, int) for n in chunks):
    #         if sum(chunks) != len(self):
    #             raise ValueError('chunk elements are incosistent')
            
    #         ids = [
    #             j for j, chunk in enumerate(chunks)
    #             for i in range(chunk)
    #         ] 

    #         return [self[ids == i] for i in range(chunks)]

    #     elif all(isinstance(n, int) for n in chunks):
    #         pass
    #     else:
    #         ValueError
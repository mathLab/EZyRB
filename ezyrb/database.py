"""Module for the snapshots database collected during the Offline stage."""

import numpy as np

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
    def __init__(self,
                 parameters=None,
                 snapshots=None,
                 scaler_parameters=None,
                 scaler_snapshots=None,
                 space=None):
        self._parameters = None
        self._snapshots = None
        self._space = None
        self.scaler_parameters = scaler_parameters
        self.scaler_snapshots = scaler_snapshots

        # if only parameters or snapshots are provided
        if (parameters is None) ^ (snapshots is None):
            raise RuntimeError(
                'Parameters and Snapshots are not both provided')

        if space is not None and snapshots is None:
            raise RuntimeError(
                'Snapshot data is not provided with Spatial data')

        if parameters is not None and snapshots is not None:
            if space is not None:
                self.add(parameters, snapshots, space)
            else:
                self.add(parameters, snapshots)


    @property
    def parameters(self):
        """
        The matrix containing the input parameters (by row).

        :rtype: numpy.ndarray
        """
        if self.scaler_parameters:
            return self.scaler_parameters.fit_transform(self._parameters)

        return self._parameters

    @property
    def snapshots(self):
        """
        The matrix containing the snapshots (by row).

        :rtype: numpy.ndarray
        """
        if self.scaler_snapshots:
            return self.scaler_snapshots.fit_transform(self._snapshots)

        return self._snapshots

    @property
    def space(self):
        """
        The matrix containing spatial information (by row).

        :rtype: numpy.ndarray
        """
        return self._space

    def __getitem__(self, val):
        """
        This method returns a new Database with the selected parameters and
        snapshots.

        .. warning:: The new parameters and snapshots are a view of the
            original Database.
        """
        if isinstance(val, int):
            if self._space is None:
                return Database(np.reshape(self._parameters[val],
                                    (1,len(self._parameters[val]))),
                                np.reshape(self._snapshots[val],
                                    (1,len(self._snapshots[val]))),
                                self.scaler_parameters,
                                self.scaler_snapshots)

            return Database(np.reshape(self._parameters[val],
                                (1,len(self._parameters[val]))),
                            np.reshape(self._snapshots[val],
                                (1,len(self._snapshots[val]))),
                            self.scaler_parameters,
                            self.scaler_snapshots,
                            np.reshape(self._space[val],
                                (1,len(self._space[val]))))

        if self._space is None:
            return Database(self._parameters[val],
                            self._snapshots[val],
                            self.scaler_parameters,
                            self.scaler_snapshots)

        return Database(self._parameters[val],
                        self._snapshots[val],
                        self.scaler_parameters,
                        self.scaler_snapshots,
                        self._space[val])

    def __len__(self):
        """
        This method returns the number of snapshots.

        :rtype: int
        """
        return len(self._snapshots)

    def add(self, parameters, snapshots, space=None):
        """
        Add (by row) new sets of snapshots and parameters to the original
        database.

        :param array_like parameters: the parameters to add.
        :param array_like snapshots: the snapshots to add.
        """
        if len(parameters) != len(snapshots):
            raise RuntimeError(
                'Different number of parameters and snapshots.')

        if self._space is not None and space is None:
            raise RuntimeError('No Spatial Value given')

        if (self._space is not None) or (space is not None):
            if space.shape != snapshots.shape:
                raise RuntimeError(
                    'shape of space and snapshots are different.')

        if self._parameters is None and self._snapshots is None:
            self._parameters = parameters
            self._snapshots = snapshots
            if self._space is None:
                self._space = space
        elif self._space is None:
            self._parameters = np.vstack([self._parameters, parameters])
            self._snapshots = np.vstack([self._snapshots, snapshots])
        else:
            self._parameters = np.vstack([self._parameters, parameters])
            self._snapshots = np.vstack([self._snapshots, snapshots])
            self._space = np.vstack([self._space, space])

        return self

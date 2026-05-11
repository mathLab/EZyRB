"""Module for the snapshots database collected during the Offline stage."""

import logging
import numpy as np

from .parameter import Parameter
from .snapshot import Snapshot

logger = logging.getLogger(__name__)


class Database:
    """
    Database class for storing parameter-snapshot pairs.

    :param array_like parameters: the input parameters
    :param array_like snapshots: the input snapshots
    :param array_like space: the input spatial data

    :Example:

        >>> import numpy as np
        >>> from ezyrb import Database, Parameter, Snapshot
        >>> params = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        >>> snapshots = np.random.rand(3, 100)
        >>> db = Database(params, snapshots)
        >>> print(len(db))
        3
        >>> print(db.parameters_matrix.shape)
        (3, 2)
        >>> print(db.snapshots_matrix.shape)
        (3, 100)
    """

    def __init__(self, parameters=None, snapshots=None, space=None):
        logger.debug(
            "Initializing Database with parameters=%s, "
            "snapshots=%s, space=%s",
            type(parameters),
            type(snapshots),
            type(space),
        )
        self._pairs = []

        if parameters is None and snapshots is None:
            logger.debug("Empty database created")
            return

        if parameters is None:
            n_snaps = len(snapshots) if snapshots is not None else 0
            parameters = [None] * n_snaps
            logger.debug(
                "Parameters were None, created %d None parameters", n_snaps
            )
        elif snapshots is None:
            n_params = len(parameters) if parameters is not None else 0
            snapshots = [None] * n_params
            logger.debug(
                "Snapshots were None, created %d None snapshots", n_params
            )

        if len(parameters) != len(snapshots):
            logger.error(
                "Mismatch: %d parameters vs %d snapshots",
                len(parameters),
                len(snapshots),
            )
            raise ValueError(
                "parameters and snapshots must have the same " "length"
            )

        logger.debug("Adding %d parameter-snapshot pairs", len(parameters))
        for param, snap in zip(parameters, snapshots):
            param = Parameter(param)
            if isinstance(space, dict):
                snap_space = space.get(tuple(param.values), None)
                # print('snap_space', snap_space)
            else:
                snap_space = space
            snap = Snapshot(snap, space=snap_space)

            self.add(param, snap)

        logger.info("Database initialized with %d snapshots", len(self))
        # TODO: eventually improve the `space` assignment in the snapshots,
        # snapshots can have different space coordinates

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
        if isinstance(val, np.ndarray):
            view = Database()
            for p, s in np.asarray(self._pairs)[val]:
                view.add(p, s)
        elif isinstance(val, (int, slice)):
            view = Database()
            view._pairs = self._pairs[val]
        return view

    def __len__(self):
        """
        This method returns the number of snapshots.

        :rtype: int
        """
        return len(self._pairs)

    def __str__(self):
        """Print minimal info about the Database"""
        s = "Database with {} snapshots and {} parameters".format(
            self.snapshots_matrix.shape[0], self.parameters_matrix.shape[1]
        )
        return s

    def add(self, parameter, snapshot):
        """
        Add (by row) new sets of snapshots and parameters to the original
        database.

        :param Parameter parameter: the parameter to add.
        :param Snapshot snapshot: the snapshot to add.
        """
        if not isinstance(parameter, Parameter):
            logger.error("Invalid parameter type: %s", type(parameter))
            raise TypeError(
                f"Expected a Parameter object, got {type(parameter)}"
            )

        if not isinstance(snapshot, Snapshot):
            logger.error("Invalid snapshot type: %s", type(snapshot))
            raise TypeError(f"Expected a Snapshot object, got {type(snapshot)}")

        self._pairs.append((parameter, snapshot))
        logger.debug(
            "Added parameter-snapshot pair. Total pairs: %d", len(self._pairs)
        )

        return self

    def split(self, chunks, seed=None):
        """

        >>> db = Database(...)
        >>> train, test = db.split([0.8, 0.2]) # ratio
        >>> train, test = db.split([80, 20])   # n snapshots

        """
        logger.debug("Splitting database with chunks=%s, seed=%s", chunks, seed)

        if seed is not None:
            np.random.seed(seed)
            logger.debug("Random seed set to %d", seed)

        if all(isinstance(n, int) for n in chunks):
            if sum(chunks) != len(self):
                logger.error(
                    "Sum of chunks %d != database size %d",
                    sum(chunks),
                    len(self),
                )
                raise ValueError("chunk elements are inconsistent")

            logger.debug("Splitting by absolute numbers: %s", chunks)
            ids = [j for j, chunk in enumerate(chunks) for i in range(chunk)]
            np.random.shuffle(ids)

        elif all(isinstance(n, float) for n in chunks):
            if not np.isclose(sum(chunks), 1.0):
                logger.error("Sum of chunk ratios %f != 1.0", sum(chunks))
                raise ValueError("chunk elements are inconsistent")

            logger.debug("Splitting by ratios: %s", chunks)
            cum_chunks = np.cumsum(chunks)
            cum_chunks = np.insert(cum_chunks, 0, 0.0)
            ids = np.ones(len(self)) * -1.0
            tmp = np.random.uniform(0, 1, size=len(self))
            for i in range(len(cum_chunks) - 1):
                is_between = np.logical_and(
                    tmp >= cum_chunks[i], tmp < cum_chunks[i + 1]
                )
                ids[is_between] = i

        else:
            logger.error("Invalid chunk type")
            raise TypeError(
                f"Invalid chunk type. Expected a list of integers or floats, but got {type(chunks)}."
            )

        new_database = [Database() for _ in range(len(chunks))]
        for i, chunk in enumerate(chunks):
            chunk_ids = np.array(ids) == i
            for p, s in np.asarray(self._pairs)[chunk_ids]:
                new_database[i].add(p, s)

        logger.info(
            "Database split into %d parts with sizes: %s",
            len(new_database),
            [len(db) for db in new_database],
        )

        return new_database

    def get_snapshot_space(self, index):
        """
        Get the space coordinates of a snapshot by its index.

        :param int index: The index of the snapshot.
        :return: The space coordinates of the snapshot.
        :rtype: numpy.ndarray
        """
        if index < 0 or index >= len(self._pairs):
            raise IndexError("Snapshot index out of range.")
        return self._pairs[index][1].space

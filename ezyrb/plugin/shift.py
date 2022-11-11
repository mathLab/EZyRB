""" Module for Scaler plugin """

from .plugin import Plugin


class ShiftSnapshots(Plugin):
    """
    The plugin implements the "shifting" preprocessing: it makes possible the
    rigid shift of the snapshots composing the database, such that the
    reduction method performs better, dipendentely by the problem at hand.
    The shift function has to be passed by the user, together with an
    `Approximation` class in order to evaluate the translated snapshots onto
    the space of a custom reference space.

    Reference: Reiss, J., Schulze, P., Sesterhenn, J., & Mehrmann, V. (2018).
    The shifted proper orthogonal decomposition: A mode decomposition for
    multiple transport phenomena. SIAM Journal on Scientific Computing, 40(3),
    A1322-A1344.

    :param callable shift_function: a user defined function that return the
        shifting quantity for any the snapshot, given the corresponding input
        parameter.
    :param Approximation interpolator: the interpolator to use to evaluate the
        shifted snapshots on some reference space.
    :param int parameter_index: in case of multi-dimensional parameter,
        indicate the index of the parameter component to pass to the shift
        function. Default is 0.
    :param int reference_index: indicate the index of the snapshots within the
        database whose space will be used as reference space. Default is 0.

    Example:


    >>> def shift(time):
    >>>     return time-0.5
    >>> pod = POD()
    >>> rbf = RBF()
    >>> db = Database()
    >>> for param in params:
    >>>     space, values = wave(param)
    >>>     snap = Snapshot(values=values, space=space)
    >>>     db.add(Parameter(param), snap)
    >>> rom = ROM(db, pod, rbf, plugins=[ShiftSnapshots(shift, RBF())])
    >>> rom.fit()

    """
    def __init__(self, shift_function, interpolator, parameter_index=0,
                 reference_index=0):
        super().__init__()

        self.__shift_function = shift_function
        self.interpolator = interpolator
        self.parameter_index = parameter_index
        self.reference_index = reference_index

    def fom_preprocessing(self, rom):
        db = rom._full_database

        reference_snapshot = db._pairs[self.reference_index][1]

        for param, snap in db._pairs:
            snap.space = reference_snapshot.space
            shift = self.__shift_function(param.values[self.parameter_index])
            self.interpolator.fit(
                snap.space.reshape(-1, 1) - shift,
                snap.values.reshape(-1, 1))

            snap.values = self.interpolator.predict(
                reference_snapshot.space.reshape(-1, 1)).flatten()

        rom._full_database = db

    def fom_postprocessing(self, rom):
        for param, snap in rom._full_database._pairs:
            snap.space = (
                rom.database._pairs[self.reference_index][1].space +
                self.__shift_function(param.values)
            )

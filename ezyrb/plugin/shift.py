""" Module for Scaler plugin """

from .plugin import Plugin
from .. import Database


class ShiftSnapshots(Plugin):
    """
    The plugin to rescale the database of the reduced order model. It uses a
    user defined `scaler`, which has to have implemented the `fit`, `transform`
    and  `inverse_trasform` methods (i.e. `sklearn` interface), to rescale
    the parameters and/or the snapshots. It can be applied at the full order
    (`mode='full'`), at the reduced one (`mode='reduced'`) or both of them
    (`mode='both'`).

    :param obj scaler: a generic object which has to have implemented the
        `fit`, `transform` and  `inverse_trasform` methods (i.e. `sklearn`
        interface).
    :param {'full', 'reduced'} mode: define if the rescaling has to be
        applied at the full order ('full') or at the reduced one ('reduced').
    :param {'parameters', 'snapshots'} params: define if the rescaling has to
        be applied to the parameters or to the snapshots.
    """
    def __init__(self, shift_function, interpolator, time_index=0,
                 reference_configuration=0):
        super().__init__()

        self.__shift_function = shift_function
        self.interpolator = interpolator
        self.time_index = time_index
        self.reference_configuration = reference_configuration

    def fom_preprocessing(self, rom):
        pass

    def fom_postprocessing(self, rom):
        pass

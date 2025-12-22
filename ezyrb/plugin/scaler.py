""" Module for Scaler plugin """

from .plugin import Plugin


class DatabaseScaler(Plugin):
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
    
    :Example:
    
        >>> from ezyrb import ReducedOrderModel as ROM
        >>> from ezyrb import POD, RBF, Database
        >>> from ezyrb.plugin import DatabaseScaler
        >>> from sklearn.preprocessing import StandardScaler
        >>> pod = POD(rank=10)
        >>> rbf = RBF()
        >>> db = Database(params, snapshots)
        >>> scaler = DatabaseScaler(StandardScaler(), 'full', 'snapshots')
        >>> rom = ROM(db, pod, rbf, plugins=[scaler])
        >>> rom.fit()
    """
    def __init__(self, scaler, mode, target) -> None:
        """
        Initialize the DatabaseScaler plugin.
        
        :param scaler: Scaler object with fit, transform, and inverse_transform methods.
        :param str mode: 'full' or 'reduced' - where to apply the scaling.
        :param str target: 'parameters' or 'snapshots' - what to scale.
        """
        super().__init__()

        self.scaler = scaler
        self.mode = mode
        self.target = target
        
    @property
    def target(self):
        """
        Get the type of scaling. See class documentation for more info.

        rtype: str
        """
        return self._target

    @target.setter
    def target(self, new_target):
        if new_target not in ['snapshots', 'parameters']:
            raise ValueError

        self._target = new_target

    @property
    def mode(self):
        """
        Get the type of scaling. See class documentation for more info.

        rtype: str
        """
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        if new_mode not in ['full', 'reduced']:
            raise ValueError

        self._mode = new_mode

    def _select_matrix(self, db):
        """
        Helper function to select the proper matrix to rescale.
        
        :param Database db: The database object.
        :return: The selected matrix (parameters or snapshots).
        """
        return getattr(db, f'{self.target}_matrix')

    def rom_preprocessing(self, rom):
        """
        Apply scaling to the reduced database before ROM processing.
        
        :param ReducedOrderModel rom: The ROM instance.
        """
        if self.mode != 'reduced':
            return

        db = rom._reduced_database

        self.scaler.fit(self._select_matrix(db))

        if self.target == 'parameters':
            new_db = type(db)(
                self.scaler.transform(self._select_matrix(db)),
                db.snapshots_matrix
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.transform(self._select_matrix(db)),
            )

        rom._reduced_database = new_db

    def fom_preprocessing(self, rom):
        if self.mode != 'full':
            return

        db = rom._full_database

        self.scaler.fit(self._select_matrix(db))

        if self.target == 'parameters':
            new_db = type(db)(
                self.scaler.transform(self._select_matrix(db)),
                db.snapshots_matrix
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.transform(self._select_matrix(db)),
            )

        rom._full_database = new_db

    def fom_postprocessing(self, rom):

        if self.mode != 'full':
            return

        db = rom._full_database

        if self.target == 'parameters':
            new_db = type(db)(
                self.scaler.inverse_transform(self._select_matrix(db)),
                db.snapshots_matrix
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.inverse_transform(self._select_matrix(db)),
            )

        rom._full_database = new_db

    def rom_postprocessing(self, rom):
        if self.mode != 'reduced':
            return

        db = rom._reduced_database

        if self.target == 'parameters':
            new_db = type(db)(
                self.scaler.inverse_transform(self._select_matrix(db)),
                db.snapshots_matrix
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.inverse_transform(self._select_matrix(db)),
            )
            
        rom._reduced_database = new_db

"""Module for Scaler plugin"""

import logging
from .plugin import Plugin

logger = logging.getLogger(__name__)


class DatabaseScaler(Plugin):
    """
    The plugin to rescale the database of the reduced order model. It uses a
    user defined `scaler`, which has to have implemented the `fit`, `transform`
    and  `inverse_transform` methods (i.e. `sklearn` interface), to rescale
    the parameters and/or the snapshots. It can be applied at the full order
    (`mode='full'`) or at the reduced one (`mode='reduced'`).

    :param obj scaler: a generic object which has to have implemented the
        `fit`, `transform` and  `inverse_transform` methods (i.e. `sklearn`
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
        logger.debug(
            "Initializing DatabaseScaler with mode=%s, target=%s", mode, target
        )

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
        if new_target not in ["snapshots", "parameters"]:
            error_msg = f"Invalid target: '{new_target}' must be 'snapshots' or 'parameters'."
            logger.error(error_msg)
            raise ValueError(error_msg)

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
        if new_mode not in ["full", "reduced"]:
            error_msg = f"Invalid mode: '{new_mode}' must be 'full' or 'reduced'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self._mode = new_mode


    def _select_matrix(self, db):
        """
        Helper function to select the proper matrix to rescale.

        :param Database db: The database object.
        :return: The selected matrix (parameters or snapshots).
        """
        return getattr(db, f"{self.target}_matrix")

    # =========================================================================
    # MODE = 'FULL' - Scaling applied at full order (before reduction or after prediction)
    # =========================================================================

    def fit_before_reduction(self, rom):
        """
        Apply scaling before POD reduction when mode='full'.
        Scales the full-order database before reduction.

        :param ReducedOrderModel rom: The ROM instance.
        """
        if self.mode != "full":
            return

        db = rom.train_full_database

        self.scaler.fit(self._select_matrix(db))

        if self.target == "parameters":
            new_db = type(db)(
                self.scaler.transform(self._select_matrix(db)),
                db.snapshots_matrix,
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.transform(self._select_matrix(db)),
            )

        rom.train_full_database = new_db

    def predict_postprocessing(self, rom):
        """
        Inverse transform scaled data after prediction when mode='full'.
        Restores original scale to the full-order predicted database.

        :param ReducedOrderModel rom: The ROM instance.
        """
        if self.mode != "full":
            return

        db = rom.predicted_full_database

        if self.target == "parameters":
            new_db = type(db)(
                self.scaler.inverse_transform(self._select_matrix(db)),
                db.snapshots_matrix,
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.inverse_transform(self._select_matrix(db)),
            )

        rom.predicted_full_database = new_db

    # =========================================================================
    # MODE = 'REDUCED' - Scaling applied at reduced order (before/after approximation)
    # =========================================================================

    def fit_before_approximation(self, rom):
        """
        Apply scaling before approximation training when mode='reduced'.
        Scales the reduced database before approximation training.

        :param ReducedOrderModel rom: The ROM instance.
        """
        if self.mode != "reduced":
            return

        db = rom.train_reduced_database

        self.scaler.fit(self._select_matrix(db))

        if self.target == "parameters":
            new_db = type(db)(
                self.scaler.transform(self._select_matrix(db)),
                db.snapshots_matrix,
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.transform(self._select_matrix(db)),
            )

        rom.train_reduced_database = new_db

    def predict_after_approximation(self, rom):
        """
        Inverse transform scaled data after approximation when mode='reduced'.
        Restores original scale to the reduced predicted database.

        :param ReducedOrderModel rom: The ROM instance.
        """
        if self.mode != "reduced":
            return

        db = rom.predict_reduced_database

        if self.target == "parameters":
            new_db = type(db)(
                self.scaler.inverse_transform(self._select_matrix(db)),
                db.snapshots_matrix,
            )
        else:
            new_db = type(db)(
                db.parameters_matrix,
                self.scaler.inverse_transform(self._select_matrix(db)),
            )

        rom.predict_reduced_database = new_db

    # =========================================================================
    # PREDICT - Scaling input parameters before approximation (both modes)
    # =========================================================================

    def predict_before_approximation(self, rom):
        """
        Transform (scale) input parameters before approximation if target='parameters'.
        This ensures parameters are scaled to match the training data.
        Applied during prediction for both 'full' and 'reduced' modes.

        :param ReducedOrderModel rom: The ROM instance.
        """
        if self.target != "parameters":
            return

        db = rom.predict_reduced_database
        transformed_params = self.scaler.transform(self._select_matrix(db))

        # During prediction, snapshots are None (not yet predicted)
        # Database constructor handles None snapshots: creates [None] * len(parameters)
        new_db = type(db)(transformed_params, None)

        rom.predict_reduced_database = new_db

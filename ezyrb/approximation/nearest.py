"""Wrapper for Nearest Neighbor Interpolator."""
import logging
import numpy as np
from scipy.interpolate import NearestNDInterpolator as NearestND
from scipy.interpolate import interp1d
from .approximation import Approximation

logger = logging.getLogger(__name__)

class Nearest(Approximation):
    """
    Nearest Neighbors interpolator for univariate and multivariate approximation.

    :param kwargs: arguments passed to the internal instance of 
        scipy.interpolate.NearestNDInterpolator or scipy.interpolate.interp1d.
    """
    def __init__(self, **kwargs):
        logger.debug("Initializing Nearest with kwargs: %s", kwargs)
        super().__init__()
        self.kwargs = kwargs
        self.interpolator = None

    def fit(self, points, values):
        as_np_array = np.array(points)
        
        if as_np_array.ndim == 1 or (as_np_array.ndim == 2 and as_np_array.shape[1] == 1):
            logger.debug("Using 1D nearest interpolation")
            self.interpolator = interp1d(
                np.squeeze(as_np_array), values, kind='nearest', 
                axis=0, bounds_error=False, fill_value="extrapolate"
            )
        else:
            logger.debug("Using ND nearest interpolation")
            self.interpolator = NearestND(as_np_array, values, **self.kwargs)
        
        logger.info("Nearest fitted successfully")

    def predict(self, new_point):
        return self.interpolator(new_point).squeeze()
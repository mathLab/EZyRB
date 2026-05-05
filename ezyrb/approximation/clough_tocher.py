"""Wrapper for Clough-Tocher 2D Interpolator."""

import logging
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator as CT

from .approximation import Approximation

logger = logging.getLogger(__name__)


class CloughTocher(Approximation):
    r"""
    :math:`C^1` smooth, piecewise cubic interpolator for 2D multivariate approximation.

    Note: This interpolator only supports 2-dimensional parameter spaces
    (i.e., mapping :math:`\mathbb{R}^2 \to \mathbb{R}^m`).

    :param kwargs: arguments passed to the internal instance of
        scipy.interpolate.CloughTocher2DInterpolator.
    """

    def __init__(self, **kwargs):
        logger.debug("Initializing CloughTocher with kwargs: %s", kwargs)
        super().__init__()
        self.kwargs = kwargs
        self.interpolator = None

    def fit(self, points, values):
        """
        Construct the interpolator given `points` and `values`.
        """
        as_np_array = np.array(points)

        # Mathematical constraint: CT only works in R^2
        if as_np_array.ndim != 2 or as_np_array.shape[1] != 2:
            logger.error(
                "CloughTocher requested for data with shape %s",
                as_np_array.shape,
            )
            raise ValueError(
                "CloughTocher interpolator only supports exactly 2D parameter spaces."
            )

        self.interpolator = CT(as_np_array, values, **self.kwargs)
        logger.info("CloughTocher fitted successfully")

    def predict(self, new_point):
        """
        Evaluate interpolator at given `new_points`.
        """
        return self.interpolator(new_point).squeeze()

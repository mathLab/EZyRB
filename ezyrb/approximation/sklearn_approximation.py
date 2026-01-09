"""
Wrapper for using any scikit-learn regressor in EZyRB.
"""

import logging
from .approximation import Approximation

logger = logging.getLogger(__name__)


class SklearnApproximation(Approximation):
    """
    Wrapper class to use any scikit-learn regressor as an approximation
    method in EZyRB.

    This class allows you to use any scikit-learn estimator that implements
    the fit/predict interface (regressors, etc.) within the EZyRB framework.

    :param sklearn_model: An instance of a scikit-learn estimator (e.g.,
        RandomForestRegressor, SVR, KNeighborsRegressor, etc.). The model
        should implement fit() and predict() methods.
    :param dict fit_params: Optional parameters to pass to the fit() method.
        Default is None.

    :Example:
        >>> from ezyrb import SklearnApproximation
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100, random_state=42)
        >>> approximation = SklearnApproximation(model)
        >>> approximation.fit(points, values)
        >>> predictions = approximation.predict(new_points)

    :Example:
        >>> from ezyrb import SklearnApproximation
        >>> from sklearn.svm import SVR
        >>> from sklearn.multioutput import MultiOutputRegressor
        >>> base_model = SVR(kernel='rbf', C=1.0)
        >>> model = MultiOutputRegressor(base_model)
        >>> approximation = SklearnApproximation(model)
        >>> approximation.fit(points, values)
    """

    def __init__(self, sklearn_model, fit_params=None):
        """
        Initialize the SklearnApproximation wrapper.

        :param sklearn_model: A scikit-learn estimator instance
        :param dict fit_params: Optional fit parameters
        """
        logger.debug(
            "Initializing SklearnApproximation with model: %s",
            type(sklearn_model).__name__
        )

        if not hasattr(sklearn_model, 'fit'):
            raise ValueError(
                "sklearn_model must have a 'fit' method"
            )
        if not hasattr(sklearn_model, 'predict'):
            raise ValueError(
                "sklearn_model must have a 'predict' method"
            )

        self.model = sklearn_model
        self.fit_params = fit_params if fit_params is not None else {}
        self._fitted = False

    def fit(self, points, values):
        """
        Fit the scikit-learn model.

        :param numpy.ndarray points: The input points (training data)
        :param numpy.ndarray values: The output values (targets)
        """
        logger.info(
            "Fitting %s with %d samples",
            type(self.model).__name__,
            points.shape[0]
        )
        logger.debug(
            "Input shape: %s, Output shape: %s",
            points.shape,
            values.shape
        )

        # Ensure 2D arrays
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        self.model.fit(points, values, **self.fit_params)
        self._fitted = True

        logger.debug("Model fitting completed")

    def predict(self, new_points):
        """
        Predict using the fitted scikit-learn model.

        :param numpy.ndarray new_points: The input points for prediction
        :return: The predicted values
        :rtype: numpy.ndarray
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before calling predict()"
            )

        logger.debug(
            "Predicting for %d new points",
            new_points.shape[0] if new_points.ndim > 1 else 1
        )

        # Ensure 2D array
        if new_points.ndim == 1:
            new_points = new_points.reshape(-1, 1)

        predictions = self.model.predict(new_points)

        logger.debug("Prediction completed, output shape: %s",
                     predictions.shape)

        return predictions

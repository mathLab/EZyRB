"""
Wrapper for using any scikit-learn dimensionality reduction method in EZyRB.
"""

import logging
from .reduction import Reduction

logger = logging.getLogger(__name__)


class SklearnReduction(Reduction):
    """
    Wrapper class to use any scikit-learn dimensionality reduction method
    in EZyRB.

    This class allows you to use any scikit-learn transformer that implements
    the fit/transform/inverse_transform interface (PCA, KernelPCA, NMF, etc.)
    within the EZyRB framework.

    :param sklearn_model: An instance of a scikit-learn transformer (e.g.,
        PCA, KernelPCA, FastICA, NMF, TruncatedSVD, etc.). The model should
        implement fit(), transform(), and optionally inverse_transform()
        methods.
    :param dict fit_params: Optional parameters to pass to the fit() method.
        Default is None.

    :Example:
        >>> from ezyrb import SklearnReduction
        >>> from sklearn.decomposition import PCA
        >>> model = PCA(n_components=5)
        >>> reduction = SklearnReduction(model)
        >>> reduction.fit(snapshots)
        >>> reduced = reduction.transform(snapshots)
        >>> reconstructed = reduction.inverse_transform(reduced)

    :Example:
        >>> from ezyrb import SklearnReduction
        >>> from sklearn.decomposition import KernelPCA
        >>> model = KernelPCA(n_components=10, kernel='rbf')
        >>> reduction = SklearnReduction(model)
        >>> reduction.fit(snapshots)

    :Example:
        >>> from ezyrb import SklearnReduction
        >>> from sklearn.decomposition import FastICA
        >>> model = FastICA(n_components=8, random_state=42)
        >>> reduction = SklearnReduction(model)
        >>> reduction.fit(snapshots)
    """

    def __init__(self, sklearn_model, fit_params=None):
        """
        Initialize the SklearnReduction wrapper.

        :param sklearn_model: A scikit-learn transformer instance
        :param dict fit_params: Optional fit parameters
        """
        logger.debug(
            "Initializing SklearnReduction with model: %s",
            type(sklearn_model).__name__,
        )

        if not hasattr(sklearn_model, "fit"):
            raise ValueError("sklearn_model must have a 'fit' method")
        if not hasattr(sklearn_model, "transform"):
            raise ValueError("sklearn_model must have a 'transform' method")

        self.model = sklearn_model
        self.fit_params = fit_params if fit_params is not None else {}
        self._fitted = False
        self._has_inverse = hasattr(sklearn_model, "inverse_transform")

        if not self._has_inverse:
            logger.warning(
                "%s does not have inverse_transform method. "
                "inverse_transform() will raise an error.",
                type(sklearn_model).__name__,
            )

    def fit(self, values):
        """
        Fit the scikit-learn dimensionality reduction model.

        :param numpy.ndarray values: The snapshots matrix (stored by column)
        """
        logger.info(
            "Fitting %s with %d snapshots",
            type(self.model).__name__,
            values.shape[0],
        )
        logger.debug("Input shape: %s", values.shape)

        # scikit-learn expects (n_samples, n_features)
        # EZyRB stores snapshots by column, so we transpose
        values_T = values.T

        self.model.fit(values_T, **self.fit_params)
        self._fitted = True

        logger.debug("Model fitting completed")

        # Log explained variance if available (e.g., PCA)
        if hasattr(self.model, "explained_variance_ratio_"):
            total_var = self.model.explained_variance_ratio_.sum()
            logger.info("Explained variance ratio: %.4f", total_var)

    def transform(self, values):
        """
        Reduce the dimensionality of the given snapshots.

        :param numpy.ndarray values: The snapshots matrix (stored by column)
        :return: The reduced representation
        :rtype: numpy.ndarray
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before calling transform()"
            )

        logger.debug("Transforming %d snapshots", values.shape[0])

        # Transpose for scikit-learn
        values_T = values.T
        reduced_T = self.model.transform(values_T)

        # Transpose back to EZyRB format
        reduced = reduced_T.T

        logger.debug(
            "Transformation completed, output shape: %s", reduced.shape
        )

        return reduced

    def inverse_transform(self, reduced_values):
        """
        Reconstruct the snapshots from reduced representation.

        :param numpy.ndarray reduced_values: The reduced representation
        :return: The reconstructed snapshots
        :rtype: numpy.ndarray
        """
        if not self._fitted:
            raise RuntimeError(
                "Model must be fitted before calling inverse_transform()"
            )

        if not self._has_inverse:
            raise NotImplementedError(
                f"{type(self.model).__name__} does not implement "
                "inverse_transform()"
            )

        logger.debug(
            "Inverse transforming %d reduced vectors", reduced_values.shape[0]
        )

        # Transpose for scikit-learn
        reduced_T = reduced_values.T
        reconstructed_T = self.model.inverse_transform(reduced_T)

        # Transpose back to EZyRB format
        reconstructed = reconstructed_T.T

        logger.debug(
            "Inverse transformation completed, output shape: %s",
            reconstructed.shape,
        )

        return reconstructed

    def reduce(self, X):
        """
        Alias for transform(). Kept for backward compatibility.

        :param numpy.ndarray X: The snapshots matrix
        :return: The reduced representation
        :rtype: numpy.ndarray
        """
        return self.transform(X)

    def expand(self, reduced):
        """
        Alias for inverse_transform(). Kept for backward compatibility.

        :param numpy.ndarray reduced: The reduced representation
        :return: The reconstructed snapshots
        :rtype: numpy.ndarray
        """
        return self.inverse_transform(reduced)

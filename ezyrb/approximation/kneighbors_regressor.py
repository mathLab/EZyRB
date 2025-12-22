"""Wrapper for K-Neighbors Regressor."""

from sklearn.neighbors import KNeighborsRegressor as Regressor

from .neighbors_regressor import NeighborsRegressor


class KNeighborsRegressor(NeighborsRegressor):
    """
    K-Neighbors Regressor for multidimensional approximation.

    :param kwargs: arguments passed to the internal instance of
        KNeighborsRegressor.
    
    :Example:
    
        >>> import numpy as np
        >>> from ezyrb import KNeighborsRegressor
        >>> x = np.random.uniform(-1, 1, size=(20, 2))
        >>> y = np.array([np.sin(x[:, 0]), np.cos(x[:, 1])]).T
        >>> knn = KNeighborsRegressor(n_neighbors=3)
        >>> knn.fit(x, y)
        >>> new_x = np.array([[0.5, 0.5]])
        >>> y_pred = knn.predict(new_x)
    """
    def __init__(self, **kwargs):
        """
        Initialize a K-Neighbors Regressor.
        
        :param kwargs: Arguments passed to sklearn's KNeighborsRegressor.
        """
        self.regressor = Regressor(**kwargs)

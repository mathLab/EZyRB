"""Wrapper for RadiusNeighborsRegressor."""

from sklearn.neighbors import RadiusNeighborsRegressor as Regressor

from .neighbors_regressor import NeighborsRegressor


class RadiusNeighborsRegressor(NeighborsRegressor):
    """
    Radius Neighbors Regressor for multidimensional approximation.

    :param kwargs: arguments passed to the internal instance of
        RadiusNeighborsRegressor.
    
    :Example:
    
        >>> import numpy as np
        >>> from ezyrb import RadiusNeighborsRegressor
        >>> x = np.random.uniform(-1, 1, size=(20, 2))
        >>> y = np.sin(x[:, 0]) * np.cos(x[:, 1])
        >>> rnn = RadiusNeighborsRegressor(radius=0.5)
        >>> rnn.fit(x, y)
        >>> new_x = np.array([[0.0, 0.0]])
        >>> y_pred = rnn.predict(new_x)
    """
    def __init__(self, **kwargs):
        """
        Initialize a Radius Neighbors Regressor.
        
        :param kwargs: Arguments passed to sklearn's RadiusNeighborsRegressor.
        """
        self.regressor = Regressor(**kwargs)

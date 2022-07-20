"""Wrapper for K-Neighbors Regressor."""

from sklearn.neighbors import KNeighborsRegressor as Regressor

from .neighbors_regressor import NeighborsRegressor


class KNeighborsRegressor(NeighborsRegressor):
    """
    K-Neighbors Regressor.

    :param kwargs: arguments passed to the internal instance of
        KNeighborsRegressor.
    """
    def __init__(self, **kwargs):
        self.regressor = Regressor(**kwargs)

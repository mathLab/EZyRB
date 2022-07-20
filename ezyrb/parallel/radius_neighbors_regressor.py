"""Wrapper for RadiusNeighborsRegressor."""

from sklearn.neighbors import RadiusNeighborsRegressor as Regressor

from .neighbors_regressor import NeighborsRegressor


class RadiusNeighborsRegressor(NeighborsRegressor):
    """
    Radius Neighbors Regressor.

    :param kwargs: arguments passed to the internal instance of
        RadiusNeighborsRegressor.
    """
    def __init__(self, **kwargs):
        self.regressor = Regressor(**kwargs)

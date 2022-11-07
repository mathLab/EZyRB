"""EZyRB package"""

__all__ = [
    'Approximation', 'RBF', 'Linear', 'GPR',
    'ANN', 'KNeighborsRegressor', 'RadiusNeighborsRegressor'
]

from .approximation import Approximation
from .rbf import RBF
from .linear import Linear
from .gpr import GPR
from .ann import ANN
from .kneighbors_regressor import KNeighborsRegressor
from .radius_neighbors_regressor import RadiusNeighborsRegressor

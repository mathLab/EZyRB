__all__ = [
    'database',
    'reduction', 'pod',
    'approximation', 'rbf', 'linear',

]

__project__ = 'EZyRB'
__title__ = "ezyrb"
__author__ = "Nicola Demo, Marco Tezzele"
__copyright__ = "Copyright 2016-2019, EZyRB contributors"
__license__ = "MIT"
__version__ = "1.0"
__mail__ = 'demo.nicola@gmail.com, marcotez@gmail.com'
__maintainer__ = __author__
__status__ = "Stable"

from .database import Database
from .reduction import Reduction
from .pod import POD
from .rbf import RBF
from .reducedordermodel import ReducedOrderModel

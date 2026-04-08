import pytest

import ezyrb
from ezyrb.parallel import ReducedOrderModel as ParallelROM
ezyrb.ReducedOrderModel = ParallelROM

from tests.test_shift import *

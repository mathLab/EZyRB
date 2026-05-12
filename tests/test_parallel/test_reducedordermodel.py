import pytest

import ezyrb
from ezyrb.parallel import ReducedOrderModel as ParallelROM
ezyrb.ReducedOrderModel = ParallelROM

# Explicitly import ONLY the original base tests, not the new extended ones
from tests.test_reducedordermodel import TestReducedOrderModel, test_invariant_pod
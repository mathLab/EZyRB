import pytest
import ezyrb
import torch
from ezyrb.parallel import ReducedOrderModel as ParallelROM

ezyrb.ReducedOrderModel = ParallelROM

from tests.test_automatic_shift import TestAutomaticShiftSnapshots
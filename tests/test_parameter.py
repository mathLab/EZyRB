from ezyrb import Parameter
import numpy as np

import pytest

test_value = [0.5, 0.78]

def test_costructor():
    param = Parameter(test_value)
    np.testing.assert_array_equal(param.values, test_value)

def test_values():
    param = Parameter(test_value)
    with pytest.raises(ValueError):
        param.values = [[0, 5]]




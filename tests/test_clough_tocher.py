import numpy as np
import pytest

from ezyrb import CloughTocher

np.random.seed(17)

def get_xy():
    npts = 20
    dinput = 2

    inp = np.random.uniform(-1, 1, size=(npts, dinput))
    out = np.array([
        np.sin(inp[:, 0]) + np.sin(inp[:, 1]**2),
        np.cos(inp[:, 0]) + np.cos(inp[:, 1]**2)
    ]).T

    return inp, out

class TestCloughTocher:
    def test_constructor_empty(self):
        model = CloughTocher()

    def test_fit(self):
        x, y = get_xy()
        approx = CloughTocher()
        approx.fit(x, y)

    def test_predict_01(self):
        x, y = get_xy()
        approx = CloughTocher() 
        approx.fit(x, y)
        test_y = approx.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=6)

    def test_wrong_dimensions(self):
        x = np.random.uniform(-1, 1, size=(10, 3))
        y = np.random.uniform(-1, 1, size=(10, 2))
        approx = CloughTocher()
        with pytest.raises(ValueError):
            approx.fit(x, y)

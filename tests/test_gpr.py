import numpy as np

from unittest import TestCase
from ezyrb import GPR
import GPy

np.random.seed(666)

def get_xy():
    npts = 20
    dinput = 4

    inp = np.random.uniform(-1, 1, size=(npts, dinput))
    out = np.array([
        np.sin(inp[:, 0]) + np.sin(inp[:, 1]**2),
        np.cos(inp[:, 2]) + np.cos(inp[:, 3]**2)
    ]).T

    return inp, out

class TestGPR(TestCase):
    def test_constructor_empty(self):
        gpr = GPR()

    def test_fit(self):
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y)
        assert isinstance(gpr.model, GPy.models.GPRegression)

    def test_predict(self):
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y, optimization_restart=10)
        test_y, covariance = gpr.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=6)

    def test_optimal_mu(self):
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y, optimization_restart=10)
        new_mu = gpr.optimal_mu(np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]]))
        np.testing.assert_array_equal(new_mu, np.array([[-1, 1, -1, -1]]))

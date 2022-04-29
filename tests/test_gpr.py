import numpy as np

from unittest import TestCase
from ezyrb import GPR
import sklearn

np.random.seed(17)

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

    def test_types(self):
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x[:, 0], y[:, 0])
        assert isinstance(gpr.model,
                          sklearn.gaussian_process.GaussianProcessRegressor)

    def test_predict_01(self):
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y, optimization_restart=50)
        test_y, variance = gpr.predict(x, return_variance=True)
        np.testing.assert_array_almost_equal(y, test_y, decimal=6)

    def test_predict_02(self):
        np.random.seed(42)
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y, optimization_restart=50)
        test_y, variance = gpr.predict(x[:4], return_variance=True)
        true_var = np.array([[5.761689e-06, 2.017326e-06],
                             [5.761686e-06, 2.017325e-06],
                             [5.761692e-06, 2.017327e-06],
                             [5.761695e-06, 2.017328e-06]])
        np.testing.assert_array_almost_equal(true_var, variance, decimal=6)

    def test_predict_03(self):
        np.random.seed(1)
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y, optimization_restart=50)
        test_y = gpr.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=6)

    def test_optimal_mu(self):
        x, y = get_xy()
        gpr = GPR()
        gpr.fit(x, y, optimization_restart=10)
        new_mu = gpr.optimal_mu(np.array([[-1, 1], [-1, 1], [-1, 1], [-1, 1]]))
        np.testing.assert_array_almost_equal(np.abs(new_mu), np.ones(4).reshape(1, -1))

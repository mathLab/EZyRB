import numpy as np

from ezyrb import (GPR, Linear, RBF, ANN, KNeighborsRegressor,
    RadiusNeighborsRegressor)
import sklearn
import pytest

import torch.nn as nn
np.random.seed(17)

def get_xy():
    npts = 10
    dinput = 4

    inp = np.random.uniform(-1, 1, size=(npts, dinput))
    out = np.array([
        np.sin(inp[:, 0]) + np.sin(inp[:, 1]**2),
        np.cos(inp[:, 2]) + np.cos(inp[:, 3]**2)
    ]).T

    return inp, out

@pytest.mark.parametrize("model,kwargs", [
    (GPR, {}),
    (ANN, {'layers': [20, 20], 'function': nn.Tanh(), 'stop_training': 1e-8, 'last_identity': True}),
    (KNeighborsRegressor, {'n_neighbors': 1}),
    (RadiusNeighborsRegressor, {'radius': 0.1}),
    (Linear, {}),
])
class TestApproximation:
    def test_constructor_empty(self, model, kwargs):
        model = model(**kwargs)

    def test_fit(self, model, kwargs):
        x, y = get_xy()
        approx = model(**kwargs)
        approx.fit(x[:, 0].reshape(-1, 1), y[:, 0].reshape(-1, 1))

        approx = model(**kwargs)
        approx.fit(x, y)

    def test_predict_01(self, model, kwargs):
        x, y = get_xy()
        approx = model(**kwargs) 
        approx.fit(x, y)
        test_y = approx.predict(x)
        if isinstance(approx, ANN):
            np.testing.assert_array_almost_equal(y, test_y, decimal=3)
        else:
            np.testing.assert_array_almost_equal(y, test_y, decimal=6)
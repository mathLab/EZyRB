import numpy as np
import torch.nn as nn

from unittest import TestCase
from ezyrb import ANN

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

class TestANN(TestCase):
    def test_constructor_empty(self):
        ann = ANN([10, 5], nn.Tanh(), 20000)

    def test_fit_mono(self):
        x, y = get_xy()
        ann = ANN([10, 5], nn.Tanh(), [20000, 1e-5])
        ann.fit(x[:, 0].reshape(-1,1), y[:, 0].reshape(-1,1))
        assert isinstance(ann.model, nn.Sequential)

    def test_fit_01(self):
        x, y = get_xy()
        ann = ANN([10, 5], nn.Tanh(), [20000, 1e-8])
        ann.fit(x, y)
        assert isinstance(ann.model, nn.Sequential)
        
    def test_fit_02(self):
        x, y = get_xy()
        ann = ANN([10, 5, 2], [nn.Tanh(), nn.Sigmoid(), nn.Tanh()], [20000, 1e-8])
        ann.fit(x, y)
        assert isinstance(ann.model, nn.Sequential)

    def test_predict_01(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], nn.Tanh(), 20)
        ann.fit(x, y)
        test_y = ann.predict(x)
        assert isinstance(test_y, np.ndarray)

    def test_predict_02(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], nn.Tanh(), [20000, 1e-8])
        ann.fit(x, y)
        test_y = ann.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=3)
        
    def test_predict_03(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], nn.Tanh(), 1e-8)
        ann.fit(x, y)
        test_y = ann.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=3)

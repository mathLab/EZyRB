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
        ann = ANN()

    def test_fit_mono(self):
        x, y = get_xy()
        ann = ANN()
        ann.fit(x[:, 0].reshape(len(x),1), y[:, 0].reshape(len(y),1))
        assert isinstance(ann.model, nn.Sequential)

    def test_fit(self):
        x, y = get_xy()
        ann = ANN()
        ann.fit(x, y)
        assert isinstance(ann.model, nn.Sequential)

    def test_predict_01(self):
        x, y = get_xy()
        ann = ANN()
        ann.fit(x, y)
        test_y = ann.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=3)

    def test_predict_02(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN()
        ann.fit(x, y)
        test_y = ann.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=3)
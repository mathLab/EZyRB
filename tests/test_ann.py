import numpy as np
import torch.nn as nn
from torch import Tensor, from_numpy

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

    def test_constrctor_loss_none(self):
        ann = ANN([10, 5], nn.Tanh(), 20000, loss=None)
        assert isinstance(ann.loss, nn.MSELoss)

    def test_constructor_single_function(self):
        passed_func = nn.Tanh()
        ann = ANN([10, 5], passed_func, 20000)

        assert isinstance(ann.function, list)
        for func in ann.function:
            assert func == passed_func

    def test_constructor_layers(self):
        ann = ANN([10, 5], nn.Tanh(), 20000)
        assert ann.layers == [10, 5]

    def test_constructor_stop_training(self):
        ann = ANN([10, 5], nn.Tanh(), 20000)
        assert isinstance(ann.stop_training, list)
        assert ann.stop_training == [20000]

    def test_constructor_fields_initialized(self):
        ann = ANN([10, 5], nn.Tanh(), 20000)
        assert ann.loss_trend == []
        assert ann.model is None

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

    def test_convert_numpy_to_torch(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]

        ann = ANN([10, 5], nn.Tanh(), 20000)

        value = ann._convert_numpy_to_torch(np.array(arr))
        assert isinstance(value, Tensor)
        for i in range(len(arr)):
            assert value[i] == arr[i]

    def test_convert_torch_to_numpy(self):
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        tensor = from_numpy(np.array(arr)).float()

        ann = ANN([10, 5], nn.Tanh(), 20000)

        value = ann._convert_torch_to_numpy(tensor)
        assert isinstance(value, np.ndarray)
        for i in range(len(arr)):
            assert value[i] == arr[i]

    def test_last_linear(self):
        passed_func = nn.Tanh()
        ann = ANN([10, 5, 2], passed_func, 20000, last_identity=True)
        ann._build_model(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]))
        last_layer = ann.model[-1]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.in_features == 2
        assert last_layer.out_features == 2

        passed_func = nn.Tanh()
        ann = ANN([10, 5, 2], passed_func, 20000, last_identity=False)
        ann._build_model(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]))
        last_layer = ann.model[-1]
        assert isinstance(last_layer, nn.Tanh)

        passed_func = [nn.Tanh()] * 3 + [nn.ReLU()]
        ann = ANN([10, 5, 2], passed_func, 20000)
        ann._build_model(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]))
        last_layer = ann.model[-1]
        assert isinstance(last_layer, nn.ReLU)

    def test_build_model(self):
        passed_func = nn.Tanh()
        ann = ANN([10, 5, 2], passed_func, 20000)

        ann._build_model(np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]]))

        assert len(ann.model) == 6 + 1
        for i in range(7):
            layer = ann.model[i]
            # the last layer, I keep the separated for clarity
            if i == 6:
                assert isinstance(layer, nn.Linear)
            elif i % 2 == 0:
                assert isinstance(layer, nn.Linear)
            else:
                assert layer == passed_func

        # check input and output
        assert ann.model[6].out_features == np.array([[5,6],[7,8]]).shape[1]
        assert ann.model[0].in_features == np.array([[1,2],[3,4]]).shape[1]

        for i in range(0, 5, 2):
            assert ann.model[i].out_features == ann.model[i+2].in_features

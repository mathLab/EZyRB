import numpy as np

from unittest import TestCase
from ezyrb import ANN

np.random.seed(17)


def get_xy():
    npts = 20
    dinput = 4

    inp = np.random.uniform(-1, 1, size=(npts, dinput))
    out = np.array(
        [
            np.sin(inp[:, 0]) + np.sin(inp[:, 1] ** 2),
            np.cos(inp[:, 2]) + np.cos(inp[:, 3] ** 2),
        ]
    ).T

    return inp, out


class TestANN(TestCase):
    def test_constructor_empty(self):
        ann = ANN([10, 5], activation='tanh', max_iter=20000)
        assert ann.layers == [10, 5]
        assert ann.activation == 'tanh'

    def test_constructor_activation(self):
        ann = ANN([10, 5], activation='relu', max_iter=20000)
        assert ann.activation == 'relu'

    def test_constructor_layers(self):
        ann = ANN([10, 5], activation='tanh', max_iter=20000)
        assert ann.layers == [10, 5]

    def test_constructor_max_iter(self):
        ann = ANN([10, 5], activation='tanh', max_iter=5000)
        assert ann.max_iter == 5000

    def test_constructor_fields_initialized(self):
        ann = ANN([10, 5], activation='tanh', max_iter=20000)
        assert ann.loss_trend == []
        assert ann.model is None

    def test_fit_mono(self):
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', max_iter=200)
        ann.fit(x[:, 0].reshape(-1, 1), y[:, 0].reshape(-1, 1))
        assert ann.model is not None
        assert len(ann.loss_trend) > 0

    def test_fit_01(self):
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', max_iter=200)
        ann.fit(x, y)
        assert ann.model is not None
        assert len(ann.loss_trend) > 0

    def test_fit_02(self):
        x, y = get_xy()
        ann = ANN([10, 5, 2], activation='tanh', max_iter=200)
        ann.fit(x, y)
        assert ann.model is not None
        assert len(ann.loss_trend) > 0

    def test_predict_01(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', max_iter=20)
        ann.fit(x, y)
        test_y = ann.predict(x)
        assert isinstance(test_y, np.ndarray)
        assert test_y.shape == y.shape

    def test_predict_02(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', max_iter=5000, tol=1e-10)
        ann.fit(x, y)
        test_y = ann.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=1)

    def test_predict_03(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', max_iter=5000, tol=1e-10)
        ann.fit(x, y)
        test_y = ann.predict(x)
        np.testing.assert_array_almost_equal(y, test_y, decimal=1)

    def test_loss_trend(self):
        np.random.seed(1)
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', max_iter=100)
        ann.fit(x, y)
        assert len(ann.loss_trend) > 0
        # Loss should generally decrease
        assert ann.loss_trend[-1] < ann.loss_trend[0]

    def test_different_activations(self):
        x, y = get_xy()
        
        for activation in ['relu', 'tanh', 'logistic', 'identity']:
            ann = ANN([10, 5], activation=activation, max_iter=100)
            ann.fit(x, y)
            test_y = ann.predict(x)
            assert test_y.shape == y.shape

    def test_solver_adam(self):
        x, y = get_xy()
        ann = ANN([10, 5], activation='tanh', solver='adam', max_iter=100)
        ann.fit(x, y)
        assert ann.model is not None

    def test_solver_sgd(self):
        x, y = get_xy()
        ann = ANN(
            [10, 5], activation='tanh', solver='sgd', 
            learning_rate_init=0.01, max_iter=100
        )
        ann.fit(x, y)
        assert ann.model is not None

    def test_regularization(self):
        x, y = get_xy()
        ann = ANN(
            [10, 5], activation='tanh', max_iter=100, 
            alpha=0.01  # L2 regularization
        )
        ann.fit(x, y)
        assert ann.model is not None

    def test_early_stopping(self):
        x, y = get_xy()
        ann = ANN(
            [10, 5], 
            activation='tanh', 
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2
        )
        ann.fit(x, y)
        # With early stopping, might not reach max_iter
        assert ann.model.n_iter_ <= ann.max_iter

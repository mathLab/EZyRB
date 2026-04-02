import numpy as np
from unittest import TestCase
from ezyrb import Nearest


class TestNearest(TestCase):
    def test_params(self):
        reg = Nearest(rescale=False)
        assert reg.rescale == False
        
    def test_default_params(self):
        reg = Nearest()
        assert reg.interpolator is None

    def test_predict1d(self):
        reg = Nearest()
        reg.fit([[1], [6], [8]], [[1, 0], [20, 5], [8, 6]])
        result = reg.predict([[1], [8], [6]])
        assert (result[0] == [1, 0]).all()
        assert (result[1] == [8, 6]).all()
        assert (result[2] == [20, 5]).all()

    def test_predict_multivariate(self):
        reg = Nearest()
        points = [[0, 0], [1, 1]]
        values = [[10, 10], [20, 20]]
        reg.fit(points, values)
        result = reg.predict([[0.1, 0.1]])
        assert (result == [10, 10]).all()

    def test_wrong_input_shape(self):
        with self.assertRaises(Exception):
            reg = Nearest()
            reg.fit([[1, 2], [6], [8, 9]], [[1, 0], [20, 5], [8, 6]])

    def test_wrong_sample_count(self):
        with self.assertRaises(Exception):
            reg = Nearest()
            reg.fit([[1, 2], [4, 5], [8, 9]], [[10, 10], [20, 20]])

    def test_batch_multivariate(self):
        """Test batch prediction with 2D input."""
        reg = Nearest()
        points = [[0, 0], [1, 1], [2, 2]]
        values = [[10, 10], [20, 20], [30, 30]]
        reg.fit(points, values)
        result = reg.predict([[0.1, 0.1], [0.9, 0.9]])
        assert result.shape == (2, 2)
        assert (result[0] == [10, 10]).all()
        assert (result[1] == [20, 20]).all()

    def test_scalar_output(self):
        """Test with scalar (1D) output values."""
        reg = Nearest()
        reg.fit([[1], [6], [8]], [10, 20, 30])
        result = reg.predict([[1], [6]])
        assert result.shape == (2,)
        assert (result == [10, 20]).all()
from unittest import TestCase
import unittest
import numpy as np
import scipy
import filecmp
import os
import sys
from ezyrb.points import Points
from ezyrb.filehandler import FileHandler


class TestPoints(TestCase):
    def test_points(self):
        pts = Points()

    def test_append(self):
        pts = Points()
        pts.append([0, 1, 2])
        expected_values = np.array([[0, 1, 2]]).T
        np.testing.assert_array_almost_equal(pts.values, expected_values)

    def test_append2(self):
        pts = Points()
        pts.append([0, 1, 2])
        pts.append([3, 4, 5])
        expected_values = np.array([[0, 1, 2], [3, 4, 5]]).T
        np.testing.assert_array_almost_equal(pts.values, expected_values)

    def test_size(self):
        pts = Points()
        pts.append([0, 1, 2])
        pts.append([3, 4, 5])
        assert pts.size == 2

    def test_dimension_1(self):
        pts = Points()
        pts.append([0, 1, 2])
        pts.append([3, 4, 5])
        assert pts.dimension == 3

    def test_dimension_2(self):
        pts = Points()
        pts.append([0, 1, 2])
        pts.append([3, 4, 5])
        pts.append([4, 4, 5])
        pts.append([6, 4, 5])
        pts.append([6, 7, 5])
        assert isinstance(pts.triangulation, scipy.spatial.qhull.Delaunay)

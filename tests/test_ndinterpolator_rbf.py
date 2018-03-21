from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
import sys
from ezyrb.ndinterpolator.rbf import RBFInterpolator


class TestRBFInterpolator(TestCase):
    def test_rbf(self):
        pts = np.eye(3)
        values = np.arange(15).reshape(3, 5)
        space = RBFInterpolator(pts, values)

    def test_points(self):
        pts = np.eye(3)
        values = np.arange(15).reshape(3, 5)
        space = RBFInterpolator(pts, values)
        np.testing.assert_array_equal(space.points, pts)

    def test_default_basis(self):
        pts = np.eye(3)
        values = np.arange(15).reshape(3, 5)
        space = RBFInterpolator(pts, values)
        assert space.basis == RBFInterpolator.multi_quadratic

    def test_default_radius(self):
        pts = np.eye(3)
        values = np.arange(15).reshape(3, 5)
        space = RBFInterpolator(pts, values)
        assert space.radius == 1.0

    def test_multi_quadratic(self):
        func_res = RBFInterpolator.multi_quadratic(np.linspace(0, 1, 5), .5)
        exct_res = np.array([0.5, 0.559017, 0.707107, 0.901388, 1.118034])
        np.testing.assert_array_almost_equal(func_res, exct_res, decimal=6)

    def test_default_call(self):
        pts = np.eye(3)
        values = np.arange(15).reshape(3, 5)
        space = RBFInterpolator(pts, values)
        exct_res = np.array(
            [[4.75195353, 5.70234424, 6.65273494, 7.60312565, 8.55351636]])
        np.testing.assert_array_almost_equal(
            space([[0, 0, 0]]), exct_res, decimal=6)

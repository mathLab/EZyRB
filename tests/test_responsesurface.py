from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
import sys
import scipy
from ezyrb.responsesurface import ResponseSurface
from ezyrb.points import Points
from ezyrb.snapshots import Snapshots


class TestResponseSurface(TestCase):
    def test_responsesurface(self):
        space = ResponseSurface()

    def test_generate(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = ResponseSurface()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        assert isinstance(space.interpolator,
                          scipy.interpolate.LinearNDInterpolator)

    def test_call(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = ResponseSurface()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        solution = space([0, 0])
        assert solution.shape == (1, 2500)

    def test_save(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = ResponseSurface()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        space.save("tests/test_datasets/ResponseSurface()space")
        assert os.path.isfile("tests/test_datasets/ResponseSurface()space")
        os.remove("tests/test_datasets/ResponseSurface()space")

    def test_loo_error(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = ResponseSurface()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        error = space.loo_error(mu, snap)
        assert error.shape == (4, )

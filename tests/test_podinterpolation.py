from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
import sys
from ezyrb.podinterpolation import PODInterpolation
from ezyrb.parametricspace import ParametricSpace
from ezyrb.points import Points
from ezyrb.snapshots import Snapshots
from scipy.interpolate import LinearNDInterpolator


class TestPODInterpolation(TestCase):
    def test_pod(self):
        space = PODInterpolation()

    def test_generate(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        assert space.pod_basis.shape == (2500, 4)

    def test_interpolator(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        assert isinstance(space.interpolator, LinearNDInterpolator)

    def test_call(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        #mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        #snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        solution = space([0, 0])
        assert solution.shape == (2500, 1)

    def test_save(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        space.save("tests/test_datasets/podspace")
        assert os.path.isfile("tests/test_datasets/podspace")


#os.remove("tests/test_datasets/podspace")

    def test_load(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        space.generate(mu, snap)
        space.save("tests/test_datasets/podspace")
        another_space = ParametricSpace.load("tests/test_datasets/podspace")
        assert another_space.pod_basis.shape == (2500, 4)
        os.remove("tests/test_datasets/podspace")

    def test_loo_error(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
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

    def test_loo_error2(self):
        mu = Points()
        snap = Snapshots(output_name="Pressure", dformat="point")
        space = PODInterpolation()
        mu.append([-.5, -.5])
        mu.append([.5, -.5])
        mu.append([.5, .5])
        mu.append([-.5, .5])
        snap.append("tests/test_datasets/matlab_00.vtk")
        snap.append("tests/test_datasets/matlab_01.vtk")
        snap.append("tests/test_datasets/matlab_02.vtk")
        snap.append("tests/test_datasets/matlab_03.vtk")
        error = space.loo_error(mu, snap)
        np.testing.assert_almost_equal(max(error), 0.149130165577, decimal=4)

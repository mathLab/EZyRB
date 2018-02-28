from unittest import TestCase
import unittest
from ezyrb.online import Online
from ezyrb.podinterpolation import PODInterpolation
import numpy as np
import filecmp
import os

space_pod_file = 'tests/test_datasets/podspace'


class TestOnline(TestCase):
    def test_online_attributes_01(self):
        output_name = 'Pressure'
        online_handler = Online(output_name, space_pod_file)
        assert online_handler.output_name == 'Pressure'

    def test_online_attributes_02(self):
        output_name = 'Pressure'
        online_handler = Online(output_name, space_pod_file)
        assert isinstance(online_handler.space, PODInterpolation)

    def test_online_run(self):
        mu_value = np.array([.0, .0])
        output_name = 'Pressure'
        online_handler = Online(output_name, space_pod_file)
        rb_solution = online_handler.run(mu_value)
        np.testing.assert_almost_equal(
            np.linalg.norm(rb_solution), 92.571669420, decimal=5)

    def test_online_run_and_store(self):
        mu_value = np.array([.0, .0])
        output_name = 'Pressure'
        filename = 'online_evaluation_vtk.vtk'
        online_handler = Online(output_name, space_pod_file)
        online_handler.run_and_store(
            mu_value,
            filename,
            geometry_file='tests/test_datasets/matlab_00.vtk')
        os.remove(filename)

from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
import sys
from ezyrb.offline import Offline
from ezyrb.filehandler import FileHandler


class TestOffline(TestCase):
    def test_offline(self):
        offline = Offline(output_name='Pressure')

    def test_init_database(self):
        mu_values = [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]
        files = [
            "tests/test_datasets/matlab_00.vtk",
            "tests/test_datasets/matlab_01.vtk",
            "tests/test_datasets/matlab_02.vtk",
            "tests/test_datasets/matlab_03.vtk"
        ]
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database(mu_values, files)
        expected_mu = np.array([[-.5, .5, .5, -.5], [-.5, -.5, .5, .5]])
        np.testing.assert_array_almost_equal(expected_mu, offline.mu.values)

    def test_init_database2(self):
        mu_values = [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]
        files = [
            "tests/test_datasets/matlab_00.vtk",
            "tests/test_datasets/matlab_01.vtk",
            "tests/test_datasets/matlab_02.vtk",
            "tests/test_datasets/matlab_03.vtk"
        ]
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database(mu_values, files)
        assert offline.snapshots.values.shape == (2500, 4)

    def test_init_database_from_file(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        expected_mu = np.array([[-.5, .5, .5, -.5], [-.5, -.5, .5, .5]])
        np.testing.assert_array_almost_equal(expected_mu, offline.mu.values)

    def test_init_database_from_file2(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        assert offline.snapshots.values.shape == (2500, 4)

    def test_init_database_from_file_nofile(self):
        conf_file = 'tests/test_datasets/notexisting_mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        with self.assertRaises(IOError):
            offline.init_database_from_file(conf_file)

    def test_init_database_from_file_wrongfile(self):
        conf_file = 'tests/test_datasets/wrongmu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        with self.assertRaises(ValueError):
            offline.init_database_from_file(conf_file)

    def test_add_snapshot(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        offline.add_snapshot([-0.29334384, -0.2312056],
                             "tests/test_datasets/matlab_04.vtk")
        expected_mu = np.array([[-.5, .5, .5, -.5, -0.29334384],
                                [-.5, -.5, .5, .5, -0.2312056]])
        np.testing.assert_array_almost_equal(expected_mu, offline.mu.values)

    def test_add_snapshot2(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        offline.add_snapshot([-0.29334384, -0.2312056],
                             "tests/test_datasets/matlab_04.vtk")
        assert offline.snapshots.values.shape == (2500, 5)

    def test_generate(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        offline.generate_rb_space()

    def test_save(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        offline.generate_rb_space()
        offline.save_rb_space('space')
        assert os.path.isfile('space')


#os.remove('space')

    def test_loo_error(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        error = offline.loo_error()
        assert error.shape == (4, )

    def test_loo_error2(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        max_error = np.max(offline.loo_error())
        assert isinstance(max_error, float)

    def test_optimal_mu(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        mu = offline.optimal_mu()
        expected_mu = np.array([-0.29334384, -0.23120563])
        np.testing.assert_array_almost_equal(mu[0], expected_mu)

    def test_optimal_mu2(self):
        conf_file = 'tests/test_datasets/mu.conf'
        offline = Offline(output_name='Pressure', dformat='point')
        offline.init_database_from_file(conf_file)
        mu = offline.optimal_mu(error=offline.loo_error())
        expected_mu = np.array([-0.29334384, -0.23120563])
        np.testing.assert_array_almost_equal(mu[0], expected_mu)

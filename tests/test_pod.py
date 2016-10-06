from unittest import TestCase
import unittest
import ezyrb.pod as pod
from ezyrb.filehandler import FileHandler
import numpy as np
import filecmp
import os
import sys
from matplotlib.testing.decorators import cleanup


class TestPod(TestCase):
	def test_pod_attributes_01(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_[0-3].vtk'
		pod_handler = pod.Pod(output_name, snapshot_files_regex)
		assert pod_handler.output_name == output_name

	def test_pod_attributes_03(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, weights_name, snapshot_file_regex=snapshot_files_regex
		)
		assert pod_handler.snapshot_files_regex == snapshot_files_regex

	def test_pod_attributes_03(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		assert pod_handler.snapshot_files_regex == snapshot_files_regex

	def test_pod_attributes_05(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		assert pod_handler.mu_values == None

	def test_pod_attributes_06(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		assert pod_handler.pod_basis == None

	def test_pod_attributes_07_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert isinstance(pod_handler.snapshots, np.ndarray)

	def test_pod_attributes_07_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert isinstance(pod_handler.snapshots, np.ndarray)

	def test_pod_attributes_08(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		assert pod_handler.weights == None

	def test_pod_start_mu_values_01(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert pod_handler.mu_values.shape == (2, 4)

	def test_pod_start_mu_values_02(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		np.testing.assert_array_almost_equal(
			pod_handler.mu_values, expected_mu_values
		)

	def test_pod_start_snapshots_01_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert pod_handler.snapshots.shape == (2500, 4)

	def test_pod_start_snapshots_01_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert pod_handler.snapshots.shape == (1024, 4)

	def test_pod_start_snapshots_02_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_snapshots = np.load(
			'tests/test_datasets/snapshots_test_vtk.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.snapshots, expected_snapshots
		)

	def test_pod_start_snapshots_02_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_snapshots = np.load(
			'tests/test_datasets/snapshots_test_mat.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.snapshots, expected_snapshots
		)

	def test_pod_start_snapshots_03(self):
		output_name = 'Velocity'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert pod_handler.snapshots.shape == (7500, 4)

	def test_pod_start_snapshots_04(self):
		output_name = 'Velocity'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_snapshots = np.load(
			'tests/test_datasets/snapshots_vectorial_test.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.snapshots, expected_snapshots
		)

	def test_pod_start_weights_01_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert (pod_handler.weights.shape == (2500, ))

	def test_pod_start_weights_01_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert (pod_handler.weights.shape == (1024, ))

	def test_pod_start_weights_02_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_weights = np.load('tests/test_datasets/weights_test.npy')
		np.testing.assert_array_almost_equal(
			pod_handler.weights, expected_weights
		)

	def test_pod_start_weights_02_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_weights = np.ones(np.shape(pod_handler.weights))
		np.testing.assert_array_almost_equal(
			pod_handler.weights, expected_weights
		)

	def test_pod_start_weights_03(self):
		output_name = 'Velocity'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		assert (pod_handler.weights.shape == (7500, ))

	def test_pod_start_weights_04(self):
		output_name = 'Velocity'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_weights = np.load(
			'tests/test_datasets/weights_vectorial_test.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.weights, expected_weights
		)

	def test_pod_start_weights_05(self):
		output_name = 'Velocity'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		expected_weights = np.load(
			'tests/test_datasets/weights_vectorial_test.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.weights, expected_weights
		)

	def test_pod_write_structures_01_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		pod_handler.write_structures()
		expected_pod_basis = np.load(
			'tests/test_datasets/pod_basis_test_vtk.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.pod_basis, expected_pod_basis
		)
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')

	def test_pod_write_structures_01_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		pod_handler.write_structures()
		expected_pod_basis = np.load(
			'tests/test_datasets/pod_basis_test_mat.npy'
		)
		np.testing.assert_array_almost_equal(
			pod_handler.pod_basis, expected_pod_basis
		)
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')

	'''def test_pod_write_structures_02(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		snapshot_files_regex = 'tests/test_datasets/matlab_0'
		
		pod_handler = pod.Pod(output_name, weights_name, snapshot_files_regex)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		pod_handler.write_structures()
		expected_outfilename = 'tests/test_datasets/expected_pod_basis_Pressure_python' + str(sys.version_info.major) + '.npy'
		self.assertTrue(filecmp.cmp('pod_basis_Pressure.npy', expected_outfilename))
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')
		
		
	def test_pod_write_structures_03(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		snapshot_files_regex = 'tests/test_datasets/matlab_0'
		
		pod_handler = pod.Pod(output_name, weights_name, snapshot_files_regex)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		pod_handler.write_structures()
		expected_outfilename = 'tests/test_datasets/expected_coefs_tria_Pressure_python' + str(sys.version_info.major) + '.npy'
		self.assertTrue(filecmp.cmp('coefs_tria_Pressure.npy', expected_outfilename))
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')'''

	def test_pod_print_info(self):
		output_name = 'Velocity'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		pod_handler.print_info()

	def test_pod_add_snapshot_01_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.vtk')
		mu_1 = np.array([-.5, .5, .5, -.5, -0.29334384])
		mu_2 = np.array([-.5, -.5, .5, .5, -0.2312056])
		expected_mu_values = np.array([mu_1, mu_2])
		np.testing.assert_array_almost_equal(
			pod_handler.mu_values, expected_mu_values
		)

	def test_pod_add_snapshot_01_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.mat')
		mu_1 = np.array([-.5, .5, .5, -.5, -0.3009776])
		mu_2 = np.array([-.5, -.5, .5, .5, -0.22145312])
		expected_mu_values = np.array([mu_1, mu_2])
		np.testing.assert_array_almost_equal(
			pod_handler.mu_values, expected_mu_values
		)

	def test_pod_add_snapshot_02_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.vtk')
		np.testing.assert_almost_equal(
			pod_handler.snapshots[0, -1], 12.1921539307
		)

	def test_pod_add_snapshot_02_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.mat')
		np.testing.assert_almost_equal(
			pod_handler.snapshots[0, -1], 12.1680652248
		)

	def test_pod_add_snapshot_03_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.vtk')
		np.testing.assert_almost_equal(
			pod_handler.snapshots[30, -1], 4.09717798233
		)

	def test_pod_add_snapshot_03_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.mat')
		np.testing.assert_almost_equal(
			pod_handler.snapshots[30, -1], -0.0695743079
		)

	def test_pod_add_snapshot_04_vtk(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].vtk'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.vtk')
		np.testing.assert_almost_equal(
			pod_handler.snapshots[-1, -1], 0.571366786957
		)

	def test_pod_add_snapshot_04_mat(self):
		output_name = 'Pressure'
		snapshot_files_regex = 'tests/test_datasets/matlab_0[0-3].mat'
		pod_handler = pod.Pod(
			output_name, snapshot_files_regex=snapshot_files_regex
		)
		pod_handler.read_config('tests/test_datasets/setting.conf')
		pod_handler.initialize_snapshot()
		new_mu = pod_handler.find_optimal_mu()
		pod_handler.add_snapshot(new_mu, 'tests/test_datasets/matlab_04.mat')
		np.testing.assert_almost_equal(
			pod_handler.snapshots[-1, -1], 0.5650054088
		)

from unittest import TestCase
import unittest
import ezyrb.interpolation as interp
import numpy as np
import filecmp
import os


class TestInterpolation(TestCase):
	def test_interp_attributes_01(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		assert interp_handler.output_name == output_name

	def test_interp_attributes_02(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		assert interp_handler.snapshot_files_regex == snapshot_files_regex

	def test_interp_attributes_04(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		assert interp_handler.mu_values == None

	def test_interp_attributes_05(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		assert isinstance(interp_handler.snapshots, np.ndarray)

	def test_interp_start_mu_values_01(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		assert interp_handler.mu_values.shape == (2, 4)

	def test_interp_start_mu_values_02(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		expected_mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		np.testing.assert_array_almost_equal(
			interp_handler.mu_values, expected_mu_values
		)

	def test_interp_start_snapshots_01(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		assert interp_handler.snapshots.shape == (1, 4)

	def test_interp_start_snapshots_02(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		expected_snapshots = np.load(
			'tests/test_datasets/snapshots_test_scalar.npy'
		)
		np.testing.assert_array_almost_equal(
			interp_handler.snapshots, expected_snapshots
		)

	'''def test_interp_write_structures_01(self):
		output_name = 'Pressure_drop'
		weights_name = 'Weights'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0'
		
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.start()
		interp_handler.write_structures()
		expected_outfilename = 'tests/test_datasets/expected_coefs_tria_Pressure.npy'
		self.assertTrue(filecmp.cmp('coefs_tria_Pressure.npy', expected_outfilename))
		os.remove('interp_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')'''

	def test_interp_write_structures_01(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		interp_handler.write_structures()
		os.remove('triangulation_scalar.npy')

	def test_interp_print_info(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		interp_handler.print_info()

	def test_interp_add_snapshot_01(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		new_mu = interp_handler.find_optimal_mu()
		interp_handler.add_snapshot(
			new_mu, 'tests/test_datasets/matlab_scalar_04.mat'
		)
		mu_1 = np.array([-.5, .5, .5, -.5, -0.22280828])
		mu_2 = np.array([-.5, -.5, .5, .5, -0.46651485])
		expected_mu_values = np.array([mu_1, mu_2])
		np.testing.assert_array_almost_equal(
			interp_handler.mu_values, expected_mu_values
		)

	def test_interp_add_snapshot_02(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		new_mu = interp_handler.find_optimal_mu()
		interp_handler.add_snapshot(
			new_mu, 'tests/test_datasets/matlab_scalar_04.mat'
		)
		np.testing.assert_almost_equal(
			interp_handler.snapshots[0, -1], 17.57596528
		)

	def test_interp_add_snapshot_03(self):
		output_name = 'Pressure_drop'
		snapshot_files_regex = 'tests/test_datasets/matlab_scalar_0[0-3].mat'
		interp_handler = interp.Interp(output_name, snapshot_files_regex)
		interp_handler.read_config('tests/test_datasets/setting.conf')
		interp_handler.initialize_snapshot()
		new_mu = interp_handler.find_optimal_mu()
		interp_handler.add_snapshot(
			new_mu, 'tests/test_datasets/matlab_scalar_04.mat'
		)
		np.testing.assert_almost_equal(
			interp_handler.snapshots[0, 0], 36.3507969378
		)

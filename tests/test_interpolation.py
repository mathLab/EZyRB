
from unittest import TestCase
import unittest
import ezyrb.interpolation as interp
import numpy as np
import filecmp
import os


class TestInterpolation(TestCase):


	def test_interp_attributes_01(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		assert interp_handler.output_name == output_name
		
		
	def test_interp_attributes_02(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		assert interp_handler.namefile_prefix == namefile_prefix
		
		
	def test_interp_attributes_03(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		assert interp_handler.file_format == file_format
		
	
	def test_interp_attributes_04(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		assert interp_handler.mu_values == None
		
		
	def test_interp_attributes_05(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		assert isinstance(interp_handler.snapshots, np.ndarray)
		
			
	def test_interp_start_mu_values_01(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		assert interp_handler.mu_values.shape == (2,4)
		
		
	def test_interp_start_mu_values_02(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		expected_mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		np.testing.assert_array_almost_equal(interp_handler.mu_values, expected_mu_values)
		
		
	def test_interp_start_snapshots_01(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		assert interp_handler.snapshots.shape == (1, 4)
		
		
	def test_interp_start_snapshots_02(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		expected_snapshots = np.load('tests/test_datasets/snapshots_test_scalar.npy')
		np.testing.assert_array_almost_equal(interp_handler.snapshots, expected_snapshots)
		
		
	'''def test_interp_write_structures_01(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		interp_handler.write_structures()
		expected_outfilename = 'tests/test_datasets/expected_coefs_tria_Pressure.npy'
		self.assertTrue(filecmp.cmp('coefs_tria_Pressure.npy', expected_outfilename))
		os.remove('interp_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')'''
		
		
	def test_interp_write_structures_01(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		interp_handler.write_structures()
		os.remove('triangulation_scalar.npy')
	
	
	def test_interp_print_info(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		interp_handler.print_info()
		
		
	def test_interp_add_snapshot_01(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		interp_handler.add_snapshot()
		mu_1 = np.array([-.5, .5,  .5, -.5, -0.22280828])
		mu_2 = np.array([-.5, -.5, .5,  .5, -0.46651485])
		expected_mu_values = np.array([mu_1, mu_2])
		np.testing.assert_array_almost_equal(interp_handler.mu_values, expected_mu_values)
		
		
	def test_interp_add_snapshot_02(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		interp_handler.add_snapshot()
		print interp_handler.snapshots[0,-1]
		np.testing.assert_almost_equal(interp_handler.snapshots[0,-1], 17.57596528)
		
			
	def test_interp_add_snapshot_03(self):
		output_name = 'Pressure'
		namefile_prefix = 'tests/test_datasets/matlab_scalar_0'
		file_format = '.mat'
		interp_handler = interp.Interp(output_name, namefile_prefix, file_format)
		interp_handler.start()
		interp_handler.add_snapshot()
		np.testing.assert_almost_equal(interp_handler.snapshots[0,0], 36.3507969378)
	

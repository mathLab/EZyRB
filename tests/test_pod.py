
from unittest import TestCase
import unittest
import ezyrb.pod as pod
import numpy as np
import filecmp
import os


class TestCvt(TestCase):


	def test_pod_attributes_01(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.output_name == output_name
		
		
	def test_pod_attributes_02(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.weights_name == weights_name
		
		
	def test_pod_attributes_03(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.namefile_prefix == namefile_prefix
		
		
	def test_pod_attributes_04(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.file_format == file_format
		
	
	def test_pod_attributes_05(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.mu_values == None
		
		
	def test_pod_attributes_06(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.pod_basis == None
		
		
	def test_pod_attributes_07(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.snapshots == None
		
		
	def test_pod_attributes_08(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		assert pod_handler.weights == None
		
		
	def test_pod_start_mu_values_01(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		assert pod_handler.mu_values.shape == (2,4)
		
		
	def test_pod_start_mu_values_02(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		expected_mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.mu_values, expected_mu_values)
		
		
	def test_pod_start_snapshots_01(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		assert pod_handler.snapshots.shape == (2500, 4)
		
		
	def test_pod_start_snapshots_02(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		expected_snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.snapshots, expected_snapshots)
		
		
	def test_pod_start_snapshots_03(self):
		output_name = 'Velocity'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		assert pod_handler.snapshots.shape == (7500, 4)
		
		
	def test_pod_start_snapshots_04(self):
		output_name = 'Velocity'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		expected_snapshots = np.load('tests/test_datasets/snapshots_vectorial_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.snapshots, expected_snapshots)
		
		
	def test_pod_start_weights_01(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		assert pod_handler.weights.shape == (2500,)
		
		
	def test_pod_start_weights_02(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		expected_weights = np.load('tests/test_datasets/weights_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.weights, expected_weights)
		
		
	def test_pod_start_weights_03(self):
		output_name = 'Velocity'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		assert pod_handler.weights.shape == (7500,)
		
		
	def test_pod_start_weights_04(self):
		output_name = 'Velocity'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		expected_weights = np.load('tests/test_datasets/weights_vectorial_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.weights, expected_weights)
		
		
	def test_pod_start_weights_05(self):
		output_name = 'Velocity'
		weights_name = 'fake_weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		expected_weights = np.load('tests/test_datasets/weights_vectorial_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.weights, expected_weights)
		
		
	def test_pod_write_structures_01(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		pod_handler.write_structures()
		expected_pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		np.testing.assert_array_almost_equal(pod_handler.pod_basis, expected_pod_basis)
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')
		
		
	'''def test_pod_write_structures_02(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		pod_handler.write_structures()
		expected_outfilename = 'tests/test_datasets/expected_pod_basis_Pressure.npy'
		self.assertTrue(filecmp.cmp('pod_basis_Pressure.npy', expected_outfilename))
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')
		
		
	def test_pod_write_structures_03(self):
		output_name = 'Pressure'
		weights_name = 'Weights'
		namefile_prefix = 'tests/test_datasets/matlab_0'
		file_format = '.vtk'
		pod_handler = pod.Pod(output_name, weights_name, namefile_prefix, file_format)
		pod_handler.start()
		pod_handler.write_structures()
		expected_outfilename = 'tests/test_datasets/expected_coefs_tria_Pressure.npy'
		self.assertTrue(filecmp.cmp('coefs_tria_Pressure.npy', expected_outfilename))
		os.remove('pod_basis_Pressure.npy')
		os.remove('coefs_tria_Pressure.npy')'''
		
	
	
	
	

	

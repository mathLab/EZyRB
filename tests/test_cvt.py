
from unittest import TestCase
import unittest
import ezyrb.cvt as cvt
import numpy as np
import filecmp
import os


class TestCvt(TestCase):


	def test_cvt_attributes_01(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		np.testing.assert_array_almost_equal(cvt_handler.mu_values, mu_values)
		
		
	def test_cvt_attributes_02(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		np.testing.assert_array_almost_equal(cvt_handler.pod_basis, pod_basis)
		

	def test_cvt_attributes_03(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		np.testing.assert_array_almost_equal(cvt_handler.snapshots, snapshots)
	
		
	def test_cvt_attributes_04(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		np.testing.assert_array_almost_equal(cvt_handler.weights, weights)
		
		
	def test_cvt_attributes_05(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		assert cvt_handler.dim_out == 2500
		
		
	def test_cvt_attributes_06(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		assert cvt_handler.dim_db == 4
		
		
	def test_cvt_attributes_07(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		
		expected_rel_error = 780.142129403
		
		np.testing.assert_almost_equal(cvt_handler.rel_error, expected_rel_error)
		
		
	def test_cvt_attributes_08(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		assert cvt_handler.max_error == None
		
		
	def test_cvt_attributes_09(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		assert cvt_handler.dim_mu == 2


	def test_cvt_compute_volume_1(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		
		points_x = np.array([0., 1., 0.])
		points_y = np.array([0., 0., 1.])
		simplex_vertices = np.array([points_x, points_y])
		
		volume = cvt_handler._compute_simplex_volume(simplex_vertices)
		
		assert volume == 0.5
		
		
	def test_cvt_compute_leave_one_out_error(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		
		error = cvt_handler.loo_error()
		expected_error = np.array([0.14913012, 0.05875263, 0.04603026, 0.07641862])
		
		np.testing.assert_array_almost_equal(error, expected_error)
		
		
	def test_cvt_compute_new_point(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		
		cvt_handler.add_new_point()
		expected_value = np.array([-0.29334384, -0.2312056])
		
		np.testing.assert_array_almost_equal(cvt_handler.mu_values[:,-1], expected_value)
		
		
	def test_cvt_compute_max_error(self):
		pod_basis = np.load('tests/test_datasets/pod_basis_test.npy')
		mu_values = np.load('tests/test_datasets/mu_values_test.npy')
		snapshots = np.load('tests/test_datasets/snapshots_test.npy')
		weights   = np.load('tests/test_datasets/weights_test.npy')
		cvt_handler = cvt.Cvt(mu_values, pod_basis, snapshots, weights)
		
		cvt_handler.add_new_point()
		expected_value = 0.14913012395372877
		
		np.testing.assert_almost_equal(cvt_handler.max_error, expected_value)
	

	

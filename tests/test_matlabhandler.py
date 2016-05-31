
from unittest import TestCase
import unittest
import ezyrb.matlabhandler as mh
import numpy as np
import filecmp
import os


class TestStlHandler(TestCase):


	def test_mat_instantiation(self):
		mat_handler = mh.MatlabHandler()
	

	def test_mat_default_infile_member(self):
		mat_handler = mh.MatlabHandler()
		assert mat_handler.infile == None


	def test_mat_default_outfile_member(self):
		mat_handler = mh.MatlabHandler()
		assert mat_handler.outfile == None


	def test_mat_default_extension_member(self):
		mat_handler = mh.MatlabHandler()
		assert mat_handler.extension == '.mat'
	

	def test_mat_parse_failing_filename_type(self):
		mat_handler = mh.MatlabHandler()
		with self.assertRaises(TypeError):
			output = mat_handler.parse(5.2)
			
	
	def test_mat_parse_failing_output_name_type(self):
		mat_handler = mh.MatlabHandler()
		with self.assertRaises(TypeError):
			output = mat_handler.parse('tests/test_datasets/matlab_field_test.vtk', 5.2)

	
	def test_mat_parse_failing_check_extension(self):
		mat_handler = mh.MatlabHandler()
		with self.assertRaises(ValueError):
			output = mat_handler.parse('tests/test_datasets/matlab_field_test.vtk')


	def test_mat_parse_infile(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		assert mat_handler.infile == 'tests/test_datasets/matlab_output_test.mat'


	def test_mat_parse_shape(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		assert output.shape == (64, 1)


	def test_mat_parse_coords_1(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		np.testing.assert_almost_equal(output[33][0], 16.8143769)


	def test_mat_parse_coords_2(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		np.testing.assert_almost_equal(output[0][0], 149.0353302)	


	def test_mat_write_failing_filename_type(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		with self.assertRaises(TypeError):
			mat_handler.write(output, 4.)


	def test_mat_write_failing_check_extension(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		with self.assertRaises(ValueError):
			mat_handler.write(output, 'tests/test_datasets/matlab_output_test_out.vtk')


	def test_mat_write_failing_infile_instantiation(self):
		mat_handler = mh.MatlabHandler()
		output = np.zeros((40, 3))
		with self.assertRaises(RuntimeError):
 			mat_handler.write(output, 'tests/test_datasets/matlab_output_test_out.mat')


	'''def test_mat_write_outfile(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		outfilename = 'tests/test_datasets/matlab_output_test_out.mat'
		mat_handler.write(output, outfilename)
		assert mat_handler.outfile == outfilename
		os.remove(outfilename)'''


	def test_stl_write_comparison(self):
		mat_handler = mh.MatlabHandler()
		output = mat_handler.parse('tests/test_datasets/matlab_output_test.mat')
		output[0] = [1.1]
		output[1] = [1.1]
		output[2] = [1.1]
		output[11] = [1.1]
		output[12] = [1.1]
		output[13] = [1.1]
		output[30] = [1.1]
		output[31] = [1.1]
		output[32] = [1.1]

		outfilename = 'tests/test_datasets/matlab_output_test_out.mat'

		mat_handler.write(output, outfilename)
		os.remove(outfilename)
		

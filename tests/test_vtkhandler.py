from unittest import TestCase
import unittest
import ezyrb.vtkhandler as vh
import numpy as np
import filecmp
import os


class TestVtkHandler(TestCase):
	def test_vtk_instantiation(self):
		vtk_handler = vh.VtkHandler()

	def test_vtk_default_infile_member(self):
		vtk_handler = vh.VtkHandler()
		assert vtk_handler.infile == None

	def test_vtk_default_extension_member(self):
		vtk_handler = vh.VtkHandler()
		assert vtk_handler.extension == '.vtk'

	def test_vtk_parse_failing_filename_type(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(TypeError):
			output = vtk_handler.parse(5.2)

	def test_vtk_parse_failing_output_name_type(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(TypeError):
			output = vtk_handler.parse(
				'tests/test_datasets/matlab_output_test.mat', 5.2
			)

	def test_vtk_parse_failing_check_extension(self):
		vtk_handler = vh.VtkHandler()
		with self.assertRaises(ValueError):
			output = vtk_handler.parse(
				'tests/test_datasets/matlab_output_test.mat'
			)

	def test_vtk_parse_infile(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		assert vtk_handler.infile == 'tests/test_datasets/matlab_field_test_bin.vtk'

	def test_vtk_parse_shape(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		assert output.shape == (2500, 1)

	def test_vtk_parse_check_data_format_1(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		assert vtk_handler.cell_data == False

	def test_vtk_parse_check_data_format_2(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/openfoam_output_test.vtk', 'p'
		)
		assert vtk_handler.cell_data == True

	def test_vtk_parse_coords_1(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		np.testing.assert_almost_equal(output[33][0], 3.7915385)

	def test_vtk_parse_coords_2(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		np.testing.assert_almost_equal(output[0][0], 8.2308226)

	def test_vtk_write_failing_filename_type(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		with self.assertRaises(TypeError):
			vtk_handler.write(output, 4.)

	def test_vtk_write_failing_check_extension(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		with self.assertRaises(ValueError):
			vtk_handler.write(
				output, 'tests/test_datasets/matlab_output_test_out.mat'
			)

	def test_vtk_write_failing_infile_instantiation(self):
		vtk_handler = vh.VtkHandler()
		output = np.zeros((40, 3))
		with self.assertRaises(RuntimeError):
			vtk_handler.write(
				output, 'tests/test_datasets/matlab_field_test_out.vtk'
			)

	def test_vtk_write_default_output_name(self):
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse(
			'tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure'
		)
		outfilename = 'tests/test_datasets/matlab_field_test_out_bin.vtk'
		vtk_handler.write(output, outfilename, write_bin=True)
		os.remove(outfilename)

	'''
	def test_vtk_write_comparison_bin_1(self):
		import vtk
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse('tests/test_datasets/matlab_field_test_bin.vtk', 'Pressure')
		output[0] = [1.1]
		output[1] = [1.1]
		output[2] = [1.1]
		output[11] = [1.1]
		output[12] = [1.1]
		output[13] = [1.1]
		output[30] = [1.1]
		output[31] = [1.1]
		output[32] = [1.1]

		outfilename = 'tests/test_datasets/matlab_field_test_out_bin.vtk'

		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/matlab_field_test_out_true_bin_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/matlab_field_test_out_true_bin_version6.vtk'
			
		vtk_handler.write(output, outfilename, 'Pressure', write_bin=True)
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
		
		
	def test_vtk_write_comparison_bin_ascii(self):
		import vtk
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse('tests/test_datasets/openfoam_output_test.vtk', 'p')
		output[0] = [1.1]
		output[1] = [1.1]
		output[2] = [1.1]
		output[11] = [1.1]
		output[12] = [1.1]
		output[13] = [1.1]
		output[30] = [1.1]
		output[31] = [1.1]
		output[32] = [1.1]

		outfilename = 'tests/test_datasets/openfoam_output_test_out.vtk'

		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/openfoam_output_test_out_true_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/openfoam_output_test_out_true_version6.vtk'
			
		vtk_handler.write(output, outfilename, 'p')
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
		
	
	def test_vtk_write_comparison_ascii(self):
		import vtk
		vtk_handler = vh.VtkHandler()
		output = vtk_handler.parse('tests/test_datasets/matlab_field_test_ascii.vtk', 'Pressure')
		output[0] = [1.1]
		output[1] = [1.1]
		output[2] = [1.1]
		output[11] = [1.1]
		output[12] = [1.1]
		output[13] = [1.1]
		output[30] = [1.1]
		output[31] = [1.1]
		output[32] = [1.1]

		outfilename = 'tests/test_datasets/matlab_field_test_out_ascii.vtk'

		if vtk.VTK_MAJOR_VERSION <= 5:
			outfilename_expected = 'tests/test_datasets/matlab_field_test_out_true_ascii_version5.vtk'
		else:
			outfilename_expected = 'tests/test_datasets/matlab_field_test_out_true_ascii_version6.vtk'
		
		vtk_handler.write(output, outfilename, 'Pressure')
		self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
		os.remove(outfilename)
	'''

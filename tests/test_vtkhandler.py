from unittest import TestCase
import unittest
import ezyrb.filehandler as fh
import ezyrb.vtkhandler as vh
import numpy as np
import filecmp
import os

vtk_file = "tests/test_datesets/test_sphere.vtk"


class TestVtkHandler(TestCase):
    def test_vtk_instantiation(self):
        vtk_handler = vh.VtkHandler(vtk_file)
        assert (isinstance(vtk_handler, vh.VtkHandler))

    def test_vtk_instantiation2(self):
        vtk_handler = fh.FileHandler(vtk_file)
        assert (isinstance(vtk_handler, vh.VtkHandler))

    def test_vtk_parse_failing_filename_type(self):
        with self.assertRaises(TypeError):
            handler = fh.FileHandler(5.2)

    def test_vtk_parse_failing_output_name_type(self):
        with self.assertRaises(TypeError):
            output = fh.FileHandler(
                'tests/test_datasets/openfoam_output_test.vtk').get_dataset(5.2)

    def test_vtk_get_geometry_wrong_filename(self):
        with self.assertRaises(RuntimeError):
            mesh_points = fh.FileHandler('nonexistent.vtk').get_geometry()

    def test_vtk_get_dataset_wrong_filename(self):
        with self.assertRaises(RuntimeError):
            mesh_points = fh.FileHandler('nonexistent.vtk').get_dataset(
                'output')

    def test_vtk_get_dataset_wrong_datatype(self):
        with self.assertRaises(ValueError):
            mesh_points = fh.FileHandler(
                'tests/test_datasets/matlab_field_test_bin.vtk').get_dataset(
                    'Pressure', 'cells')

    def test_vtk_set_dataset_wrong_datatype(self):
        mesh_points = fh.FileHandler(
            'tests/test_datasets/matlab_field_test_bin.vtk').get_dataset(
                'Pressure')
        with self.assertRaises(ValueError):
            fh.FileHandler(
                'tests/test_datasets/matlab_field_test_bin.vtk').set_dataset(
                    mesh_points, 'Pressure', 'faces')

    def test_vtk_parse_shape(self):
        output = fh.FileHandler('tests/test_datasets/matlab_field_test_bin.vtk'
                                ).get_dataset('Pressure')
        assert output.shape == (2500, 1)

    def test_vtk_parse_check_data_format_2(self):
        output = fh.FileHandler(
            'tests/test_datasets/openfoam_output_test.vtk').get_dataset(
                'p', 'cell')
        assert output.shape == (400, 1)

    def test_vtk_parse_coords_1(self):
        output = fh.FileHandler('tests/test_datasets/matlab_field_test_bin.vtk'
                                ).get_dataset('Pressure')
        np.testing.assert_almost_equal(output[33, 0], 3.7915385)

    def test_vtk_parse_coords_2(self):
        output = fh.FileHandler('tests/test_datasets/matlab_field_test_bin.vtk'
                                ).get_dataset('Pressure')
        np.testing.assert_almost_equal(output[0], 8.2308226)

    def test_vtk_write_failing_filename_type(self):
        handler = fh.FileHandler(
            'tests/test_datasets/matlab_field_test_bin.vtk')
        output = handler.get_dataset('Pressure')
        with self.assertRaises(TypeError):
            handler.set_dataset(output, 4.)

    def test_vtk_write0(self):
        outfilename = 'tests/test_datasets/matlab_field_test_bin_write.vtk'
        read_handl = fh.FileHandler(
            'tests/test_datasets/matlab_field_test_bin.vtk')
        points, cells = read_handl.get_geometry(get_cells=True)

        write_handl = fh.FileHandler(outfilename)
        write_handl.set_geometry(points, cells)
        os.remove(outfilename)

    def test_vtk_write1(self):
        outfilename = 'tests/test_datasets/matlab_field_test_bin_write.vtk'
        read_handl = fh.FileHandler(
            'tests/test_datasets/matlab_field_test_bin.vtk')
        output = read_handl.get_dataset('Pressure')
        points, cells = read_handl.get_geometry(get_cells=True)

        write_handl = fh.FileHandler(outfilename)
        write_handl.set_geometry(points, cells)
        write_handl.set_dataset(output, 'Pressure', write_bin=True)
        os.remove(outfilename)

    def test_vtk_list_output_point(self):
        read_handl = fh.FileHandler(
            'tests/test_datasets/matlab_field_test_bin.vtk')
        out_point, out_cell = read_handl.get_all_output_names()
        expected_out = ['Velocity', 'Pressure']
        assert (out_point == expected_out)

    def test_vtk_list_output_cell(self):
        read_handl = fh.FileHandler(
            'tests/test_datasets/openfoam_output_test.vtk')
        out_point, out_cell = read_handl.get_all_output_names()
        expected_out = ['cellID', 'p', 'U']
        assert (out_cell == expected_out)

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

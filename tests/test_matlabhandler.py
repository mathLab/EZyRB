from unittest import TestCase
import unittest
import ezyrb.matlabhandler as mh
import ezyrb.filehandler as fh
import numpy as np
import filecmp
import os

mat_file = "tests/test_datasets/matlab_output_test.mat"


class TestMatlabHandler(TestCase):
    def test_mat_instantiation(self):
        mat_handler = mh.MatlabHandler(mat_file)
        assert (isinstance(mat_handler, mh.MatlabHandler))

    def test_mat_instantiation1(self):
        mat_handler = fh.FileHandler(mat_file)
        assert (isinstance(mat_handler, mh.MatlabHandler))

    def test_mat_parse_failing_filename_type(self):
        with self.assertRaises(TypeError):
            fh.FileHandler(5.2)

    def test_mat_parse_failing_output_name_type(self):
        with self.assertRaises(TypeError):
            output = fh.FileHandler(mat_file).get_dataset(5.2)

    def test_mat_get_dataset_wrong_filename(self):
        with self.assertRaises(RuntimeError):
            mesh_points = fh.FileHandler('nonexistent.mat').get_dataset(
                'output')

    def test_mat_get_dataset_shape(self):
        output = fh.FileHandler(mat_file).get_dataset('output')
        assert output.shape == (64, 1)

    def test_mat_get_dataset_coords_1(self):
        output = fh.FileHandler(mat_file).get_dataset('output')
        np.testing.assert_almost_equal(output[33][0], 16.8143769)

    def test_mat_get_dataset_coords_2(self):
        output = fh.FileHandler(mat_file).get_dataset('output')
        np.testing.assert_almost_equal(output[0][0], 149.0353302)

    def test_stl_write_comparison(self):
        output = fh.FileHandler(mat_file).get_dataset('output')
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

        fh.FileHandler(outfilename).set_dataset(output, 'output')
        os.remove(outfilename)

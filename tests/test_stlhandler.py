from unittest import TestCase
import unittest
import ezyrb.stlhandler as sh
import ezyrb.filehandler as fh
import numpy as np
import filecmp
import os

stl_file = 'tests/test_datasets/test_sphere.stl'
stl_bin_file = 'tests/test_datasets/test_sphere_bin.stl'


class TestStlHandler(TestCase):
    def test_stl_instantiation_1(self):
        stl_handler = sh.StlHandler(stl_file)
        assert (isinstance(stl_handler, sh.StlHandler))

    def test_stl_instantiation_2(self):
        stl_handler = fh.FileHandler(stl_file)
        assert (isinstance(stl_handler, sh.StlHandler))

    def test_stl_failing_filename_type(self):
        with self.assertRaises(TypeError):
            fh.FileHandler(5.2)

    def test_stl_shape_point(self):
        mesh_points = fh.FileHandler(stl_file).get_geometry()[0]
        assert mesh_points.shape == (197, 3)

    def test_stl_shape_cell(self):
        cells = fh.FileHandler(stl_file).get_geometry(get_cells=True)[1]
        assert len(cells) == 390

    def test_stl_coords_1(self):
        mesh_points = fh.FileHandler(stl_file).get_geometry()[0]
        np.testing.assert_almost_equal(mesh_points[33, 0], 0.198186)

    def test_stl_get_geometry_wrong_filename(self):
        with self.assertRaises(RuntimeError):
            mesh_points = fh.FileHandler('nonexistent.stl').get_geometry()

    def test_stl_get_geometry_coords_2(self):
        mesh_points = fh.FileHandler(stl_file).get_geometry()[0]
        expected_coordinates = np.array([-0.25, -0.433012992, 0])
        np.testing.assert_array_almost_equal(mesh_points[147],
                                             expected_coordinates)

    def test_stl_get_geometry_coords_3(self):
        cells = fh.FileHandler(stl_file).get_geometry(get_cells=True)[1]
        expected_list = [19, 17, 20]
        assert (expected_list == cells[17])

    def test_stl_get_geometry_coords_5_bin(self):
        mesh_points = fh.FileHandler(stl_file).get_geometry()[0]
        np.testing.assert_almost_equal(mesh_points[-1, 2], -0.45048400)

    def test_stl_set_geometry_comparison(self):
        mesh_points, cells = fh.FileHandler(stl_file).get_geometry(
            get_cells=True)
        mesh_points[0] = [-40.2, -20.5, 60.9]
        mesh_points[1] = [-40.2, -10.5, 60.9]
        mesh_points[2] = [-40.2, -10.5, 60.9]
        mesh_points[50] = [-40.2, -20.5, 60.9]
        mesh_points[51] = [-40.2, -10.5, 60.9]
        mesh_points[52] = [-40.2, -10.5, 60.9]
        mesh_points[126] = [-40.2, -20.5, 60.9]
        mesh_points[127] = [-40.2, -10.5, 60.9]
        mesh_points[128] = [-40.2, -10.5, 60.9]

        outfilename = 'tests/test_datasets/test_sphere_out.stl'
        outfilename_expected = 'tests/test_datasets/test_sphere_out_true.stl'

        fh.FileHandler(outfilename).set_geometry(mesh_points, cells)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    def test_stl_set_geometry_binary_from_binary(self):
        mesh_points, cells = fh.FileHandler(stl_bin_file).get_geometry(
            get_cells=True)
        mesh_points[0] = [-40.2, -20.5, 60.9]
        mesh_points[1] = [-40.2, -10.5, 60.9]
        mesh_points[2] = [-40.2, -10.5, 60.9]
        mesh_points[50] = [-40.2, -20.5, 60.9]
        mesh_points[51] = [-40.2, -10.5, 60.9]
        mesh_points[52] = [-40.2, -10.5, 60.9]
        mesh_points[126] = [-40.2, -20.5, 60.9]
        mesh_points[127] = [-40.2, -10.5, 60.9]
        mesh_points[128] = [-40.2, -10.5, 60.9]

        outfilename = 'tests/test_datasets/test_sphere_out_bin.stl'
        outfilename_expected = 'tests/test_datasets/test_sphere_out_bin_true.stl'

        fh.FileHandler(outfilename).set_geometry(
            mesh_points, cells, write_bin=True)
        self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
        os.remove(outfilename)

    # TODO Check these test: it fails but it shouldn't do

    # def test_stl_set_geometry_binary_from_ascii(self):
    #     mesh_points, cells = fh.FileHandler(stl_file).get_geometry(get_cells=True)
    #     mesh_points[0] = [-40.2, -20.5, 60.9]
    #     mesh_points[1] = [-40.2, -10.5, 60.9]
    #     mesh_points[2] = [-40.2, -10.5, 60.9]
    #     mesh_points[50] = [-40.2, -20.5, 60.9]
    #     mesh_points[51] = [-40.2, -10.5, 60.9]
    #     mesh_points[52] = [-40.2, -10.5, 60.9]
    #     mesh_points[126] = [-40.2, -20.5, 60.9]
    #     mesh_points[127] = [-40.2, -10.5, 60.9]
    #     mesh_points[128] = [-40.2, -10.5, 60.9]

    #     outfilename = 'tests/test_datasets/test_sphere_out_bin.stl'
    #     outfilename_expected = 'tests/test_datasets/test_sphere_out_bin_true.stl'

    #     fh.FileHandler(outfilename).set_geometry(mesh_points, cells, write_bin=True)
    #     self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
    #     os.remove(outfilename)

    # def test_stl_set_geometry_ascii_from_binary(self):
    #     stl_handler = sh.StlHandler(stl_bin_file)
    #     mesh_points, cells = stl_handler.get_geometry(get_cells=True)
    #     mesh_points[0] = [-40.2, -20.5, 60.9]
    #     mesh_points[1] = [-40.2, -10.5, 60.9]
    #     mesh_points[2] = [-40.2, -10.5, 60.9]
    #     mesh_points[50] = [-40.2, -20.5, 60.9]
    #     mesh_points[51] = [-40.2, -10.5, 60.9]
    #     mesh_points[52] = [-40.2, -10.5, 60.9]
    #     mesh_points[126] = [-40.2, -20.5, 60.9]
    #     mesh_points[127] = [-40.2, -10.5, 60.9]
    #     mesh_points[128] = [-40.2, -10.5, 60.9]

    #     outfilename = 'tests/test_datasets/test_sphere_out.stl'
    #     outfilename_expected = 'tests/test_datasets/test_sphere_out_true.stl'

    #     stl_handler.set_geometry(outfilename, mesh_points, cells, write_bin=False)
    #     mesh_points2, cells2 = stl_handler.get_geometry(outfilename_expected, get_cells=True)
    #     self.assertTrue(filecmp.cmp(outfilename, outfilename_expected))
    #     os.remove(outfilename)

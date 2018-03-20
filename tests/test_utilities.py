from unittest import TestCase
import unittest
import ezyrb.utilities
import numpy as np
from ezyrb.filehandler import FileHandler


class TestPod(TestCase):
    def test_normal(self):
        p0 = np.array([1, 0, 0])
        p1 = np.array([0, 1, 0])
        p2 = np.array([0, 0, 1])
        normal = ezyrb.utilities.normal(p0, p1, p2)
        real_normal = np.array([1, 1, 1])
        np.testing.assert_array_almost_equal(normal, real_normal)

    def test_normalize(self):
        array = np.array([2, 3, 5, 7])
        normalized = ezyrb.utilities.normalize(array)
        np.testing.assert_almost_equal(1.0, np.linalg.norm(normalized))

    def test_polygon_area(self):
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        area = ezyrb.utilities.polygon_area(p)
        real_area = 0.866025403784
        np.testing.assert_almost_equal(area, real_area)

    def test_polygon_area2(self):
        p = np.array([[1, 0, 0], [0, 0, 1]])
        area = ezyrb.utilities.polygon_area(p)
        real_area = 0.0
        np.testing.assert_almost_equal(area, real_area)

    def test_simplex_volume(self):
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        volume = ezyrb.utilities.simplex_volume(p)
        real_volume = 1. / 6.
        np.testing.assert_almost_equal(volume, real_volume)

    def test_compute_area(self):
        area = ezyrb.utilities.compute_area(
            'tests/test_datasets/test_sphere.stl')
        np.testing.assert_almost_equal(3.076495, area.sum(), decimal=6)

    def test_compute_normals_points(self):
        normals = ezyrb.utilities.compute_normals(
            'tests/test_datasets/test_sphere.stl', datatype='point')
        np.testing.assert_array_almost_equal(
            [0, 0, 0], normals.sum(axis=0), decimal=4)

    def test_compute_normals_cell(self):
        normals = ezyrb.utilities.compute_normals(
            'tests/test_datasets/test_sphere.stl')
        np.testing.assert_array_almost_equal(
            [0, 0, 0], normals.sum(axis=0), decimal=4)

    def test_write_area(self):
        ezyrb.utilities.write_area('tests/test_datasets/test_sphere.vtk')
        area = FileHandler('tests/test_datasets/test_sphere.vtk').get_dataset(
            'Area', 'cell')
        np.testing.assert_almost_equal(3.118406, area.sum(), decimal=6)

    def test_write_normals(self):
        ezyrb.utilities.write_normals('tests/test_datasets/test_sphere.vtk')
        normals = FileHandler(
            'tests/test_datasets/test_sphere.vtk').get_dataset(
                'Normals', 'cell')
        np.testing.assert_array_almost_equal(
            [0, 0, 0], normals.sum(axis=0), decimal=4)

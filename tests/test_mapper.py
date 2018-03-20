from unittest import TestCase
import unittest
import ezyrb.mapper as ma
from ezyrb.filehandler import *
import numpy as np
import filecmp
import os

stl_file = 'tests/test_datasets/test_sphere.stl'
vtk_file = 'tests/test_datasets/test_sphere.vtk'
map_file = 'tests/test_datasets/test_sphere_mapped.vtk'


class TestMapper(TestCase):
    def test_mapper_instantiation(self):
        mapper = ma.Mapper()

    def test_mapper_set_output(self):
        mapper = ma.Mapper()
        names = ["this", "is", "a", "test"]
        mapper.output_name = names
        assert (names == mapper.output_name)

    def test_mapper_set_output_wrong_type(self):
        mapper = ma.Mapper()
        names = ["this", "is", 5.2, "test"]
        with self.assertRaises(TypeError):
            mapper.output_name = names

    def test_mapper_set_output_append2(self):
        mapper = ma.Mapper()
        names = ["this", "is"]
        names2 = ["a", "test"]
        mapper.output_name = names
        mapper.output_name = mapper.output_name + names2
        assert ((names + names2) == mapper.output_name)

    def test_mapper_set_output_append3(self):
        mapper = ma.Mapper()
        names = ["this", "is"]
        names2 = "a"
        mapper.output_name = names
        mapper.output_name = mapper.output_name + [names2]
        assert (["this", "is", "a"] == mapper.output_name)

    def test_mapper_set_output_append4(self):
        mapper = ma.Mapper()
        names = "this"
        names2 = "a"
        mapper.output_name = names
        mapper.output_name = mapper.output_name + [names2]
        assert (["this", "a"] == mapper.output_name)

    def test_interpolate_function_setter(self):
        def custom_interp(values, dist):
            return values[np.argmin(dist)]

        mapper = ma.Mapper()
        mapper.interpolate_function = custom_interp

    def test_interpolate_function_setter_wrongtype(self):
        mapper = ma.Mapper()
        with self.assertRaises(TypeError):
            mapper.interpolate_function = 3

    def test_interpolation_mode_setter_wrongtype(self):
        mapper = ma.Mapper()
        with self.assertRaises(TypeError):
            mapper.interpolation_mode = 5

    def test_interpolation_mode_setter_wrongvalue(self):
        mapper = ma.Mapper()
        with self.assertRaises(ValueError):
            mapper.interpolation_mode = 'node'

    def test_mapper_number_neighbors(self):
        mapper = ma.Mapper()
        mapper.number_neighbors = 3
        assert (3 == mapper.number_neighbors)

    def test_mapper_number_neighbors_setter_wrongtype(self):
        mapper = ma.Mapper()
        with self.assertRaises(TypeError):
            mapper.number_neighbors = np.array([3])

    def test_find_neighbour(self):
        mapper = ma.Mapper()
        with self.assertRaises(RuntimeError):
            mapper._find_neighbour([0, 0, 0])

    def test_mapper_callable_build_neighbour_locator(self):
        mapper = ma.Mapper()
        points = np.eye(3)
        mapper._build_neighbour_locator(points)

    def test_mapper_find_neighbour_coord(self):
        mapper = ma.Mapper()
        points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        mapper._build_neighbour_locator(points)
        neighbour = mapper._find_neighbour([1, 0, 0])
        closest = points[neighbour[:, 0].astype(int)]
        np.testing.assert_array_almost_equal(closest, [points[0]])

    def test_mapper_find_neighbour_dist(self):
        mapper = ma.Mapper()
        points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        mapper._build_neighbour_locator(points)
        neighbour = mapper._find_neighbour([1, 0, 0])
        closest = neighbour[:, mapper.number_neighbors:]
        np.testing.assert_almost_equal(1., closest)

    def test_mapper_find_neighbour_dist2(self):
        mapper = ma.Mapper()
        points = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        mapper._build_neighbour_locator(points)
        mapper.number_neighbors = 2
        neighbour = mapper._find_neighbour([1, 0, 0])
        closest = neighbour[:, mapper.number_neighbors:]
        np.testing.assert_almost_equal(np.array([[1., 2**0.5]]), closest)

    def test_mapper_map_solution_name(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.map_solution(map_file, vtk_file, stl_file)
        point_output_name = FileHandler(map_file).get_all_output_names()[0]
        os.remove(map_file)
        assert (point_output_name == ["test"])

    def test_mapper_map_solution_wrong_name(self):
        mapper = ma.Mapper()
        mapper.output_name = "testtest"
        with self.assertRaises(RuntimeError):
            mapper.map_solution(map_file, vtk_file, stl_file)

    def test_mapper_map_solution_point(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.map_solution(map_file, vtk_file, stl_file)
        point_map = FileHandler(map_file).get_geometry()[0]
        point_stl = FileHandler(stl_file).get_geometry()[0]
        os.remove(map_file)
        np.testing.assert_array_almost_equal(point_map, point_stl)

    def test_mapper_map_solution_cell(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.map_solution(map_file, vtk_file, stl_file)
        cell_map = FileHandler(map_file).get_geometry(get_cells=True)[1]
        cell_stl = FileHandler(stl_file).get_geometry(get_cells=True)[1]
        os.remove(map_file)
        assert (cell_map == cell_stl)

    def test_mapper_map_solution(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.map_solution(map_file, vtk_file, stl_file)
        test_map = FileHandler(map_file).get_dataset("test")
        os.remove(map_file)
        assert (test_map.shape == (197, 1))

    def test_mapper_map_solution_nogeometryfile(self):
        mapper = ma.Mapper()
        mapper.output_name = "Pressure"
        mapper.interpolation_mode = 'point'
        mapper.map_solution(vtk_file, 'tests/test_datasets/matlab_00.vtk')

    def test_mapper_map_solution2(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.interpolation_mode = 'cell'
        mapper.map_solution(map_file, vtk_file, stl_file)
        test_map = FileHandler(map_file).get_dataset("test", datatype='cell')
        os.remove(map_file)
        assert (test_map.shape == (390, 1))

    def test_mapper_map_solution3(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.map_solution(map_file, vtk_file, stl_file)
        test_map = FileHandler(map_file).get_dataset("test")
        os.remove(map_file)
        np.testing.assert_almost_equal(test_map[117, 0], 0.71308929, decimal=3)

    def test_mapper_map_solution4(self):
        mapper = ma.Mapper()
        mapper.output_name = "test"
        mapper.interpolation_mode = 'cell'
        mapper.map_solution(map_file, vtk_file, stl_file)
        test_map = FileHandler(map_file).get_dataset("test", datatype='cell')
        np.testing.assert_almost_equal(test_map[253, 0], 766.0)
        os.remove(map_file)

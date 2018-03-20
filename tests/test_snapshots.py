from unittest import TestCase
import unittest
import numpy as np
import filecmp
import os
import sys
from ezyrb.snapshots import Snapshots


class TestSnapshots(TestCase):
    def test_snapshots(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")

    def test_append(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        assert snaps.values.shape == (2500, 1)

    def test_append2(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        snaps.append("tests/test_datasets/matlab_01.vtk")
        assert snaps.weighted.shape == (2500, 2)

    def test_append_weights(self):
        snaps = Snapshots(
            output_name='Weights', weight_name='Weights', dformat="cell")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        assert snaps.weights.shape == (2401, 1)

    def test_append_wrongname(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        with self.assertRaises(TypeError):
            snaps.append(3.6)

    def test_size(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        snaps.append("tests/test_datasets/matlab_01.vtk")
        assert snaps.size == 2

    def test_dimension(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        snaps.append("tests/test_datasets/matlab_01.vtk")
        assert snaps.dimension == 2500

    def test_files(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        snaps.append("tests/test_datasets/matlab_01.vtk")
        expected_files = [
            "tests/test_datasets/matlab_00.vtk",
            "tests/test_datasets/matlab_01.vtk"
        ]
        assert snaps.files == expected_files

    def test_getitem(self):
        snaps = Snapshots(output_name='Pressure', dformat="point")
        snaps.append("tests/test_datasets/matlab_00.vtk")
        snaps.append("tests/test_datasets/matlab_01.vtk")
        snaps.append("tests/test_datasets/matlab_02.vtk")
        snaps.append("tests/test_datasets/matlab_03.vtk")
        expected_files = [
            "tests/test_datasets/matlab_00.vtk",
            "tests/test_datasets/matlab_01.vtk"
        ]
        assert snaps[0:2].files == expected_files

from unittest import TestCase
import unittest
import ezyrb.filehandler as fh
import numpy as np
import filecmp
import os


class TestFilehandler(TestCase):
    def test_file_filename(self):
        handler = fh.FileHandler('inexistent_file.vtk')
        assert (handler._filename == 'inexistent_file.vtk')

    def test_file_wrong_type_filename(self):
        with self.assertRaises(TypeError):
            handler = fh.FileHandler([3])

    def test_file_available_ext(self):
        with self.assertRaises(TypeError):
            handler = fh.FileHandler('inexistent_file.pdf')

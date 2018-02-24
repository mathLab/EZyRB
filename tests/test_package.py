from unittest import TestCase
import unittest
import pkgutil
from os import walk
from os import path


class TestPackage(TestCase):
    def test_import_ez_1(self):
        import ezyrb as ez
        fh = ez.filehandler.FileHandler('inexistent.vtk')

    def test_import_ez_2(self):
        import ezyrb as ez
        mh = ez.matlabhandler.MatlabHandler('inexistent.mat')

    def test_import_ez_3(self):
        import ezyrb as ez
        vh = ez.vtkhandler.VtkHandler('inexistent.vtk')

    def test_modules_name(self):
        # it checks that __all__ includes all the .py files in ezyrb folder
        import ezyrb
        package = ezyrb

        f_aux = []
        for (__, __, filenames) in walk('ezyrb'):
            f_aux.extend(filenames)

        f = []
        for i in f_aux:
            file_name, file_ext = path.splitext(i)
            if file_name != '__init__' and file_ext == '.py':
                f.append(file_name)

        print(f)
        print(package.__all__)
        assert (sorted(package.__all__) == sorted(f))

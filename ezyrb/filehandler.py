"""
Base module with the base class for reading and writing different files.
"""
import os
import ezyrb.stlhandler
import ezyrb.vtkhandler
import ezyrb.matlabhandler


class FileHandler(object):
    """
    A base class for file handling.
    """

    def __new__(cls, filename):
        """
        Generic file handler. When you create a new instance of FilaHandler, it
        returns specialized file handler (`MatlabHandler`, `VtkHandler`, ...)
        instance.

        :param str filename: name of file
        """
        if not isinstance(filename, str):
            raise TypeError("Filename must be a string")

        ext = {
            ".vtk": ezyrb.vtkhandler.VtkHandler,
            ".stl": ezyrb.stlhandler.StlHandler,
            ".mat": ezyrb.matlabhandler.MatlabHandler,
        }

        specialized_handler = ext.get(os.path.splitext(filename)[1])

        if specialized_handler is None:
            raise TypeError("Not supported file")

        return specialized_handler(filename)

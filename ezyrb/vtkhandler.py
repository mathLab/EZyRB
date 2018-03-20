"""
Derived module from filehandler.py to handle Vtk files.
"""
import os
import numpy as np
import vtk
import vtk.util.numpy_support as ns


class VtkHandler(object):
    """
    Vtk format file handler class.
    You are NOT supposed to call directly this class constructor (use
    :class:`.FileHandler` constructor instead)

    :param str filename: the name of the file to handle.

    :cvar str _filename: name of file to handle
    :cvar vtkPolyData _cached_data: private attribute to store the last polydata
        processed
    """

    def __init__(self, filename):

        self._filename = filename
        self._cached_data = None

    def _read_polydata(self):
        """
        This private method reads the given `filename` and return a vtkPolyData
        object containing all informations about file; to avoid useless IO
        operation on the same file, it stores polydata of the last file parsed
        and if user ask for this file, the polydata previously stored is
        returned.

        :return: polydata containing information about file.
        :rtype: vtkPolyData
        """
        # Polydata from `filename` is allready loaded; return it
        if self._cached_data is not None:
            return self._cached_data

        if not os.path.isfile(self._filename):
            raise RuntimeError("{0!s} doesn't exist".format(
                os.path.abspath(self._filename)))

        reader = vtk.vtkDataSetReader()
        reader.SetFileName(self._filename)
        reader.Update()
        data = reader.GetOutput()

        self._cached_data = data

        return data

    def _save_polydata(self, data, write_bin=False):
        """
        This private method saves into `filename` the `data`. `data` is a
        vtkPolydata. It is possible to specify format for `filename`: if
        `write_bin` is True, file is written in binary format, otherwise in
        ASCII format. This method save cached polydata to reduce number of IO
        operations.

        :param vtkPolyData data: polydatat to save.
        :param bool write_bin: for binary format file.
        """
        self._cached_data = data

        writer = vtk.vtkDataSetWriter()

        if write_bin:
            writer.SetFileTypeToBinary()

        writer.SetFileName(self._filename)

        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(data)
        else:
            writer.SetInputData(data)
        writer.Write()

    def get_geometry(self, get_cells=False):
        """
        This method reads the given filename and returns points and cells of
        file. If get_cells is True, it computes the list that contain index of
        points defining cells, otherwise the list is not computed and set to
        None (less expensive).

        :param bool get_cells: flag to compute cells list or not. Default is
            false.

        :return: the `n_points`-by-3 matrix containing the coordinates of the
            points, the `n_cells` list containing, for each cell, the iindex
            of the points that define the cell (if computed).
        :rtype: numpy.ndarray, list(numpy.ndarray)
        """
        data = self._read_polydata()

        n_points = data.GetNumberOfPoints()
        n_cells = data.GetNumberOfCells()

        points = np.array([data.GetPoint(i) for i in np.arange(n_points)])
        if get_cells:
            cells = [[
                data.GetCell(i).GetPointIds().GetId(idx)
                for idx in np.arange(data.GetCell(i).GetNumberOfPoints())
            ] for i in np.arange(n_cells)]
        else:
            cells = None

        return points, cells

    def set_geometry(self, points, cells, write_bin=False):
        """
        This method writes to `filename` a new data defined by `points` and
        `cells`.

        :param numpy.ndarray points: matrix *n_points*-by-3 containing
            coordinates of all points.
        :param list(array_like) cell: list that contains for each cell the list
            of the indices of points that define the cell.
        :param bool write_bin: flag to write in the binary format. Default is
            false.
        """
        data = vtk.vtkPolyData()
        vtk_points = vtk.vtkPoints()
        vtk_cells = vtk.vtkCellArray()

        for i in np.arange(points.shape[0]):
            vtk_points.InsertNextPoint(points[i])

        for i in np.arange(len(cells)):
            vtk_cells.InsertNextCell(len(cells[i]), cells[i])

        data.SetPoints(vtk_points)
        data.SetPolys(vtk_cells)

        self._save_polydata(data, write_bin)

    def get_dataset(self, output_name, datatype='point'):
        """
        This method reads the given `filename` and returns a numpy array
        containing `output_name` field for each point (if datatype is 'point')
        or for each cell (if datatype is 'cell').
        If `output_name` is not a field in file, it raises exception.

        :param str output_name: the name of the output of interest to extract
            from file.
        :param str datatype: a string to specify if point data or cell data
            should be returned. Default value is `point`.
        :return: the matrix `n_components`-by-`n_elements` containing the
            extracted output.
        :rtype: numpy.ndarray
        """
        if datatype not in ['cell', 'point']:
            raise ValueError("datatype MUST be 'cell' or 'point'")

        data = self._read_polydata()

        if datatype == 'point':
            extracted_data = data.GetPointData().GetArray(output_name)
        else:
            extracted_data = data.GetCellData().GetArray(output_name)

        if extracted_data is None:
            raise RuntimeError(
                datatype + " data has no " + output_name + " field.")

        output_values = ns.vtk_to_numpy(extracted_data)

        # Make 1D array a column array
        # (400, ) --> (400,1)
        try:
            cols = output_values.shape[1]
        except IndexError:
            cols = 1
        return output_values.reshape((-1, cols))

    def set_dataset(self,
                    output_values,
                    output_name,
                    datatype='point',
                    write_bin=False):
        """
        Writes to filename the given output. `output_values` is a matrix that
        contains the new values of output to write, `output_name` is a string
        that indicates name of output to write.

        :param numpy.ndarray output_values: the
            *n_points*-by-*n_components* matrix containing the output values.
        :param str output_name: name of the output.
        :param str datatype: a string to specify if point data or cell data
            should be returned. Default value is 'point'.
        :param bool write_bin: flag to write in the binary format. Default is
            false.
        """
        if datatype not in ['cell', 'point']:
            raise ValueError("datatype MUST be 'cell' or 'point'")

        data = self._read_polydata()

        output_array = ns.numpy_to_vtk(
            num_array=output_values, array_type=vtk.VTK_DOUBLE)
        output_array.SetName(output_name)

        if datatype == 'point':
            data.GetPointData().AddArray(output_array)
        else:
            data.GetCellData().AddArray(output_array)

        self._save_polydata(data, write_bin)

    def get_all_output_names(self):
        """
        Return the list of all the output names by point and the list of all the
        output names by cell.

        :return: point output name, cell output name
        :rtype: tuple(list(str), list(str))
        """

        data = self._read_polydata()

        point_name = [
            data.GetPointData().GetArrayName(i)
            for i in np.arange(data.GetPointData().GetNumberOfArrays())
        ]

        cell_name = [
            data.GetCellData().GetArrayName(i)
            for i in np.arange(data.GetCellData().GetNumberOfArrays())
        ]

        return point_name, cell_name

"""
Utilities to map solution on a specific geometry
"""
import sys
import numpy as np
from ezyrb.filehandler import FileHandler
import scipy


class Mapper(object):
    """
    Documentation

    :cvar list(str) _output_name: the list of output names to map.
    :cvar str _mode: indicate if new file will be interpolated using point
        data or cell data.
    :cvar int _n_neighbors: number of neighbors to use to interpolate new
        value.
    :cvar function _interpolate_func: function to interpolate new value
        starting from neighbors value.
    """

    def __init__(self):
        self._output_name = []
        self._neighbour_locator = None
        self._mode = "point"
        self._n_neighbors = 1
        self.__interpolate_func = Mapper.default_interpolate_func

    @staticmethod
    def get_cell_centroid(vertices):
        """
        Compute the centroid of the cell defined by `vertices`; the centroid is
        computed as the `vertices` coordinates average.

        :param numpy.ndarray vertices: a *n_vertices*-by-3 matrix where vertices
            coordinates are store by row.
        :return: cell centroid coordinates
        :rtype: numpy.ndarray
        """
        centroid = np.mean(vertices, axis=0)
        return centroid

    @staticmethod
    def default_interpolate_func(values, distance):
        """
        Default function for the point interpolation. Take as argument
        the neighbors values and the respective distance between
        to-interpolate point and neighbors, and return the values weighted
        average (weights are the reciprocal of the distances). If
        to-interpolate point coincides with one of the neighbors, return that
        neighbour value.

        :param numpy.ndarray values: a matrix *number_of_neighbors* -by-
            *number_of_components* containing neighbors values.
        :param numpy.narray distance: a vector containing distance between
            to-interpolate point and neighbors.
        """
        if not distance.all():
            return values[np.where(distance == 0)]

        return np.average(values, weights=np.reciprocal(distance), axis=0)

    @property
    def output_name(self):
        """
        The names of outputs you wanna interpolate from the original solution
        file to the mapped one.

        :getter: Returns the output names
        :setter: Sets the output names
        :type: list(str)
        """
        return self._output_name

    @output_name.setter
    def output_name(self, output_name):

        if not isinstance(output_name, list):
            output_name = [output_name]

        for name in output_name:
            if not isinstance(name, str):
                raise TypeError

        self._output_name = output_name

    @property
    def interpolate_function(self):
        """
        The function to interpolate value of a new point using neighbors value.
        It has to take as arguments two `numpy.array`: the first indicates
        neighbors value (one row for each value) and the second indicates
        distance from interpolated point.

        :getter: gets the interpolated function
        :setter: sets the interpolation function
        :type: function
        """
        return self.__interpolate_func

    @interpolate_function.setter
    def interpolate_function(self, func):

        if not callable(func):
            raise TypeError

        self.__interpolate_func = func

    @property
    def interpolation_mode(self):
        """
        Indicate if solution will be mapped using the selected output point data
        or the selected output cell data. Valid value is 'point' or 'cell'.

        :getter: gets current mode
        :setter: sets the mode
        :type: string
        """
        return self._mode

    @interpolation_mode.setter
    def interpolation_mode(self, mode):

        if not isinstance(mode, str):
            raise TypeError

        if mode not in ['cell', 'point']:
            raise ValueError

        self._mode = mode

    def _build_neighbour_locator(self, points):
        """
        Construct kdtree to nearest neighbors search.

        :param numpy.ndarray points: a *n_points* -by- 3 where are stored
            coordinates of points by row
        """
        try:
            self._neighbour_locator = scipy.spatial.cKDTree(points)
        except RuntimeError:
            # The maximum recursion limit can be exceeded for large data sets.
            sys.setrecursion(points.shape[0])
            self._neighbour_locator = scipy.spatial.cKDTree(points)

    def _find_neighbour(self, coordinates):
        """
        This method looks for the nearest neighbors to a given point; the
        locator have to build before looking for operation.

        :param numpy.array coordinates: coordinates of the point to query
        :return: a 2 *n_neighbors*-by-*n_query_points*; it contains in the
            first `number_neighbors` columns the index of neighbors , and in
            last *number_neighbors* columns the distance from query points to
            neighbors.
        :rtype: numpy.ndarray
        """
        if self._neighbour_locator is None:
            raise RuntimeError("Neighbour locator seems not initialized")

        distance, id_neighbour = self._neighbour_locator.query(
            coordinates, self._n_neighbors)

        return np.concatenate(
            (np.array(id_neighbour).reshape(
                (-1, self._n_neighbors)), np.array(distance).reshape(
                    (-1, self._n_neighbors))),
            axis=1)

    @property
    def number_neighbors(self):
        """
        The number of neighbors to use for new value interpolation.

        :getter: Sets number of neighbors
        :setter: Return number of neighbors
        :type: int
        """
        return self._n_neighbors

    @number_neighbors.setter
    def number_neighbors(self, n):

        if not isinstance(n, int):
            raise TypeError

        self._n_neighbors = n

    def map_solution(self, output_file, solution_file, geometry_file=None):
        """
        This method maps the solution on a new geometry. To be more
        specific, for each point (or cell) of the given geometry looks for the
        nearest solution points (or cells) and interpolate the selected output
        using neighbors values.It is possible select number of neighbors and the
        interpolate function. If geometry_file is not specified, to-interpolate
        points are taken from output_file, so in this case output_file must
        contain geometry information; if geometry_file is specified,
        to-interpolate points are taken from geometry_file and the interpolated
        solution is write on output_file.

        :param str output_file: name of file where interpolated solution will
            be stored. Only vtk file are supported.
        :param str solution_file: name of file where real solution is stored.
        :param str geometry_file: name of file where geometry information is
            stored.
        """

        if not geometry_file:
            geometry_file = output_file

        geo_file_handler = FileHandler(geometry_file)
        sol_file_handler = FileHandler(solution_file)
        out_file_handler = FileHandler(output_file)

        #####
        # Read points and cell from geometry file and solution file
        #
        point_sol, cell_sol = sol_file_handler.get_geometry(get_cells=True)
        point_map, cell_map = geo_file_handler.get_geometry(get_cells=True)

        #####
        # Only if geomtry file is not output file, I have to copy geomtry
        # to output file
        if geometry_file != output_file:
            out_file_handler.set_geometry(point_map, cell_map)

        #####
        # Only for 'cell' mapping
        #
        # if mapping using cell, compute cells centroid and use them
        # as dataset for neighbors locator to find closest cell
        if self.interpolation_mode == "cell":
            centroid_sol = np.array(
                [self.get_cell_centroid(point_sol[cell]) for cell in cell_sol])
            centroid_map = np.array(
                [self.get_cell_centroid(point_map[cell]) for cell in cell_map])

            point_sol = centroid_sol
            point_map = centroid_map

        #####
        # Build locator using correct points
        self._build_neighbour_locator(point_sol)

        #####
        # For each point find closest points in solution geometry
        neighbors = np.array([
            self._find_neighbour(query_point).ravel()
            for query_point in point_map
        ])

        #####
        # Here the real mapping of new points on solution points
        for output_name in self._output_name:
            output_array = sol_file_handler.get_dataset(
                output_name, datatype=self._mode)

            index_near = neighbors[:, 0:self.number_neighbors].astype(int)
            distance = neighbors[:, self.number_neighbors:]

            mapped_output = np.array([
                self.interpolate_function(output_array[index_near[id_neigh]],
                                          distance[id_neigh])
                for id_neigh in np.arange(neighbors.shape[0])
            ])

            out_file_handler.set_dataset(
                mapped_output, output_name, datatype=self._mode)

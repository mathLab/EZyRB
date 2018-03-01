"""
Module implementing some utilities for EZyRB
"""
import math
import numpy as np
from ezyrb.filehandler import FileHandler


def normal(p0, p1, p2):
    """
    Compute the surface normal of the surface indicated by three points.

    :param array_like p0: first point
    :param array_like p1: second point
    :param array_like p2: third point
    :return: the normal vector
    :rtype: numpy.ndarray
    """
    return np.cross(p1 - p0, p2 - p0)


def normalize(v):
    """
    Normalize a vector.

    :param array_like v: the vector to normalize
    :return: the normalized vector
    :rtype: numpy.ndarray
    """
    return v / np.linalg.norm(v, 2)


def polygon_area(points):
    """
    Compute the area of a planar non-self-intersecting polygon defined by
    `points`.

    :param numpy.ndarray points: a matrix that contains the vertices coordinates
        stored by row.
    :return: the area of the polygon
    :rtype: float
    """
    num_points = points.shape[0]

    if num_points < 3:
        return 0.0

    total = np.sum([
        np.cross(points[i], points[(i + 1) % num_points])
        for i in np.arange(num_points)
    ],
                   axis=0)

    unit_vector = normalize(normal(*points[0:3]))

    return np.abs(np.dot(total, unit_vector) * .5)


def simplex_volume(vertices):
    """
    Method implementing the computation of the volume of a N dimensional
    simplex.
    Source from: `wikipedia.org/wiki/Simplex
    <https://en.wikipedia.org/wiki/Simplex>`_.

    :param numpy.ndarray simplex_vertices: Nx3 array containing the
        parameter values representing the vertices of a simplex. N is the
        dimensionality of the parameters.

    :return: N dimensional volume of the simplex.
    :rtype: float
    """
    distance = np.transpose([vertices[0] - vi for vi in vertices[1:]])
    return np.abs(np.linalg.det(distance) / math.factorial(vertices.shape[1]))


def compute_area(filename):
    """
    Given a file, this method computes the area for each cell of the mesh stored
    in the file and returns it. It uses :func:`polygon_area`.

    :param str filename: the name of the file to parse in order to extract
        the necessary information about the cells.
    :return: the array that contains the area of each cells.
    :rtype: numpy.ndarray
    """
    points, cells = FileHandler(filename).get_geometry(get_cells=True)
    return np.array([polygon_area(points[cell]) for cell in cells])


def compute_normals(filename, datatype='cell'):
    """
    Given a file, this method computes the surface normals of the mesh stored
    in the file. It allows to compute the normals of the cells or of the points.
    The normal computed in a point is the interpolation of the cell normals of
    the cells adiacent to the point.

    :param str filename: the name of the file to parse in order to extract
        the geometry information.
    :param str datatype: indicate if the normals have to be computed for the
        points or the cells. The allowed values are: 'cell', 'point'. Default
        value is 'cell'.
    :return: the array that contains the normals.
    :rtype: numpy.ndarray
    """
    points, cells = FileHandler(filename).get_geometry(get_cells=True)
    normals = np.array(
        [normalize(normal(*points[cell][0:3])) for cell in cells])

    if datatype == 'point':
        normals_cell = np.empty((points.shape[0], 3))
        for i_point in np.arange(points.shape[0]):
            cell_adiacent = [cells.index(c) for c in cells if i_point in c]
            normals_cell[i_point] = normalize(
                np.mean(
                    normals[cell_adiacent], axis=0))
        normals = normals_cell

    return normals


def write_area(filename, output_name='Area'):
    """
    Given a file, this method computes the area for each cell of the mesh stored
    in the file and save it as new dataset.

    :param str filename: the name of the file to parse in order to extract
        the geometry information.
    :param str output_name: the name of the new dataset that contains the
        cells area.
    """
    FileHandler(filename).set_dataset(
        compute_area(filename), output_name, datatype='cell')


def write_normals(filename, output_name='Normals', datatype='cell'):
    """
    Given a file, this method computes the surface normals of the mesh stored
    in the file and save it as new dataset.

    :param str filename: the name of the file to parse in order to extract
        the geometry information.
    :param str output_name: the name of the new dataset that contains the
        normals.
    :param str datatype: indicate if the normals have to be computed for the
        points or the cells. The allowed values are: 'cell', 'point'. Default
        value is 'cell'.
    """
    FileHandler(filename).set_dataset(
        compute_normals(filename), output_name, datatype=datatype)

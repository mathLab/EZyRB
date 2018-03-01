"""
Class for computation of the Offline part. It provides methods for:
    - import the `snapshots` and the parameter values
      correlated(:func:`init_database <ezyrb.offline.Offline.init_database>` ,
      :func:`init_database_from_file
      <ezyrb.offline.Offline.init_database_from_file>`);
    - generate the reduced space with the method chosen by the user;
    - estimate the error using the `leave-one-out` strategy;
    - save the reduced space to a specific file.
"""

import os
import numpy as np
from ezyrb.podinterpolation import PODInterpolation
from ezyrb.points import Points
from ezyrb.snapshots import Snapshots
from ezyrb.utilities import simplex_volume


class Offline(object):
    """
    Offline phase.

    :param ParametricSpace spacetype: the method used for the reduced space
        generation. Default is :class:`.PODInterpolation`.
    :param str output_name: the name of the output of interest.
    :param str weight_name: the name of the output to consider as weight.
    :param str dformat: the data format to extract from the snapshot files:
        if the parameter is "cell", the snapshot values refer to the cell data,
        if the parameter is "point", the snapshot values refer to the point
        data. These are the only options available.

    :cvar Points mu_values: the object that contains the parameter values
    :cvar Snapshots snapshots: the object that contains the snapshots extracted
        from the chosen files.
    :cvar ParametricSpace spacetype: the method used for
        the reduced space generation.
    """

    def __init__(self,
                 output_name,
                 space_type=PODInterpolation,
                 weight_name=None,
                 dformat='cell'):

        self.mu = Points()
        self.snapshots = Snapshots(output_name, weight_name, dformat)
        self.space = space_type()

    def init_database(self, values, files):
        """
        Initialize the database with the passed parameter values and snapshot
        files: the *i*-th parameter has to be the parametric point of the
        solution stored in the *i*-th file.

        :param array_like values: the list of parameter values.
        :param array_like files: the list of the solution files.
        """
        for mu in values:
            self.mu.append(mu)
        for fl in files:
            self.snapshots.append(fl)

    def init_database_from_file(self, filename):
        """
        Initialize the database by reading the parameter values and the
        snapshot files from a given filename; this file has to be format as
        following: first *N* columns indicate the parameter values, the *N+1*
        column indicates the corresponding snapshot file.

        Example of a generic file::

            par1    par2    ...     solution_file1
            par1    par2    ...     solution_file2
            par1    par2    ...     solution_file3

        :param str filename: name of file where parameter values and snapshot
            files are stored.
        """
        if not os.path.isfile(filename):
            raise IOError("File {0!s} not found".format(
                os.path.abspath(filename)))

        matrix = np.genfromtxt(filename, dtype=None)
        num_cols = len(matrix.dtype)

        if num_cols < 2:
            raise ValueError("Not valid number of columns in database file")

        snapshots_files = matrix[matrix.dtype.names[num_cols - 1]].astype(str)
        mu_values = np.array([
            matrix[name].astype(float)
            for name in matrix.dtype.names[0:num_cols - 1]
        ])

        self.init_database(mu_values.T, snapshots_files)

    def add_snapshot(self, new_mu, new_file):
        """
        This methos adds the new solution to the database and the new parameter
        values to the parameter points; this can be done only after the new
        solution has be computed and placed in the proper directory.

        :param array_like new_mu: the parameter value to add to database.
        :param str new_file: the name of snapshot file to add to
            database.
        """
        self.mu.append(new_mu)
        self.snapshots.append(new_file)

    def generate_rb_space(self):
        """
        Generate the reduced basis space by combining the snapshots. It
        uses the chosen method for the generation.
        """
        self.space.generate(self.mu, self.snapshots)

    def save_rb_space(self, filename):
        """
        Save the reduced basis space to `filename`.

        :param str filename: the file where the space will be stored.
        """
        self.space.save(filename)

    def loo_error(self, func=np.linalg.norm):
        """
        Estimate the approximation error using *leave-one-out* strategy. The
        main idea is to create several reduced spaces by combining all the
        snapshots except one. The error vector is computed as the difference
        between the removed snapshot and the projection onto the properly
        reduced space. The procedure repeats for each snapshot in the database.
        The `func` is applied on each vector of error to obtained a float
        number.

        :param function func: the function used to assign at each vector of
            error a float number. It has to take as input a 'numpy.ndarray` and
            returns a float. Default value is the L2 norm.
        :return: the vector that contains the errors estimated for all
            parametric points.
        :rtype: numpy.ndarray
        """
        return self.space.loo_error(self.mu, self.snapshots, func)

    def optimal_mu(self, error=None, k=1):
        """
        Return the parametric points where new high-fidelity solutions have to
        be computed in ordere to globaly reduce the estimated error. These
        points are the barycentric center of the region (simplex) with higher
        error.

        :param numpy.ndarray error: the estimated error evaluated for each
            snapshot; if error array is not passed, it is computed using
            :func:`loo_error` with the default function. Default value is None.
        :param int k: the number of optimal points to return. Default value is
            1.
        :return: the optimal points
        :rtype: list(numpy.ndarray)
        """
        if error is None:
            error = self.loo_error()

        tria = self.mu.triangulation

        error_on_simplex = np.array([
            np.sum(error[smpx]) * simplex_volume(self.mu.values.T[smpx])
            for smpx in tria.simplices
        ])

        barycentric_point = []
        for index in np.argpartition(error_on_simplex, -k)[-k:]:
            worst_tria_pts = self.mu.values.T[tria.simplices[index]]
            worst_tria_err = error[tria.simplices[index]]

            barycentric_point.append(
                np.average(
                    worst_tria_pts, axis=0, weights=worst_tria_err))

        return barycentric_point

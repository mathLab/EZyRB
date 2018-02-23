"""
Utilities for the online evaluation of the output of interest
"""
import numpy as np
import os
from ezyrb.filehandler import FileHandler
from ezyrb.parametricspace import ParametricSpace

class Online(object):
    """
    Online phase
    
    :param string output_name: the name of the output of interest.
    :param string space_filename: the name of the file where the space has
        been saved.
    
    :cvar string output_name: the name of the output of interest.
    :cvar ezyrb.space space_type: the type of space used for the online phase.
    """

    def __init__(self, output_name, space_type, space_filename):
        self.output_name = output_name
        self.space = ParametricSpace.load(space_filename)

    def run(self, value):
        """
        This method evaluates the new point `value` in the parametric space and
        returns the approximated solution.

        :param array_like value: the point where the approximated solution has
            to be evaluated.

        :return: the approximated solution.
        :rtype: numpy.ndarray
        """
        return self.space(value)

    def run_and_store(self, value, filename, geometry_file=None):
        """
        This method evaluates the new point `value` in the parametric space and
        save the approximated solution on `filename`. It is possible to pass as
        optional argument the `geometry_file` that contains the topology on
        which the solution is projected.

        :param array_like value: the point where the approximated solution has
            to be evaluated.
        :param string filename: the file where the approximated solution is
            projected.
        :param string geometry_filename: the file that contains the topology to
            use for the solution projection.
        """
        output = self.space(value)
        writer = FileHandler(filename)
        if geometry_file:
            points, cells = FileHandler(geometry_file).get_geometry(True)
            writer.set_geometry(points, cells)

        writer.set_dataset(output, self.output_name)

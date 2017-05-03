"""
Utilities for the online evaluation of the output of interest
"""
import numpy as np
import os
from ezyrb.filehandler import FileHandler


class Online(object):
	"""
	Documentation
	
	:param numpy.ndarray mu_value: values of the parameters for the new evaluation
		of the output.
	:param string output_name: suffix of the files containing the structures for
		the online phase.
	:param string directory: directory where the structures are stored.
	:param bool is_scalar: boolean to set if the output of interest is a scalar
		or a field.
	
	:cvar numpy.ndarray mu_value: values of the parameters for the new
		evaluation of the output.
	:cvar string output_name: suffix of the files containing the structures for
		the online phase.
	:cvar string directory: directory where the structures are stored.
	:cvar bool is_scalar: boolean to set if the output of interest is a scalar
		or a field.
	:cvar numpy.ndarray output: new evaluation of the output of interest.
	:cvar string file_extension: extension of the output_file.

	"""

	def __init__(self, output_name, space_type, rb_space_filename):
		self.output_name = output_name
		self.space = space_type()
		self.space.load(rb_space_filename)

	def run(self, value):
		"""
		This method runs the online evaluation.
		"""
		return self.space(value)

	def run_and_store(self, value, filename, geometry_file=None):
		"""
		This method runs the online evaluation.
		"""
		output = self.space(value)
		writer = FileHandler(filename)
		if geometry_file:
			points, cells = FileHandler(geometry_file).get_geometry(True)
			writer.set_geometry(points, cells)

		writer.set_dataset(output, self.output_name)

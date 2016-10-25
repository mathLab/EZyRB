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

	def __init__(self, mu_value, output_name, directory='./', is_scalar=True):
		self.mu_value = mu_value
		self.directory = directory
		self.is_scalar = is_scalar
		self.output_name = output_name
		self.output = None

	def perform_scalar(self):
		"""
		This method performs the online evaluation of the output if it is a
		scalar.
		"""

		## TODO Check if encoding option works well
		hyper_surf = np.load(
			self.directory + 'triangulation_scalar.npy', encoding='latin1'
		)
		surf = hyper_surf.all()

		self.output = surf(self.mu_value)

	def perform_field(self):
		"""
		This method performs the online evaluation of the output if it is a
		field.
		"""

		hyper_surf = np.load(
			self.directory + 'coefs_tria_' + self.output_name + '.npy',
			encoding='latin1'
		)
		pod_basis = np.load(
			self.directory + 'pod_basis_' + self.output_name + '.npy',
			encoding='latin1'
		)

		new_coefs = np.array([surf(self.mu_value)[0] for surf in hyper_surf])

		self.output = np.dot(pod_basis, new_coefs)

	def run(self):
		"""
		This method runs the online evaluation.
		"""

		if self.is_scalar is True:
			self.perform_scalar()
		else:
			self.perform_field()

	def write_file(self, filename, geometry_file=None):
		"""
		This method writes out the solution in the proper format. In this way,
		you can view the results with the viewer you like.
		
		:param string filename: name of the output file.
		:param string geometry_file: name of file from which get the geometry.
		"""

		writer = FileHandler(filename)
		if geometry_file:
			points, cells = FileHandler(geometry_file).get_geometry(True)
			writer.set_geometry(points, cells)

		writer.set_dataset(self.output, self.output_name)

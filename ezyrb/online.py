"""
Utilities for the online evaluation of the output of interest
"""
import numpy as np
import ezyrb.vtkhandler as vh
import ezyrb.matlabhandler as mh
import os


class Online(object):
	"""
	Documentation
	
	:param numpy.ndarray mu_value: values of the parameters for the new evaluation of the output.
	:param string output_name: suffix of the files containing the structures for the online phase.
	:param string directory: directory where the structures are stored.
	:param bool is_scalar: boolean to set if the output of interest is a scalar or a field.
	
	:cvar numpy.ndarray mu_value: values of the parameters for the new evaluation of the output.
	:cvar string output_name: suffix of the files containing the structures for the online phase.
	:cvar string directory: directory where the structures are stored.
	:cvar bool is_scalar: boolean to set if the output of interest is a scalar or a field.
	:cvar numpy.ndarray output: new evaluation of the output of interest.
	:cvar string file_extension: extension of the output_file.

	"""

	def __init__(self, mu_value, output_name, directory='./', is_scalar=True):
		self.mu_value = mu_value
		self.directory = directory
		self.is_scalar = is_scalar
		self.output_name = output_name
		self.output = None
		self.file_extension = None

	def perform_scalar(self):
		"""
		This method performs the online evaluation of the output if it is a scalar.
		"""

		## TODO Check if encoding option works well
		hyper_surf = np.load(
			self.directory + 'triangulation_scalar.npy', encoding='latin1'
		)
		surf = hyper_surf.all()

		self.output = surf.__call__([self.mu_value])

	def perform_field(self):
		"""
		This method performs the online evaluation of the output if it is a field.
		"""

		hyper_surf = np.load(
			self.directory + 'coefs_tria_' + self.output_name + '.npy',
			encoding='latin1'
		)
		pod_basis = np.load(
			self.directory + 'pod_basis_' + self.output_name + '.npy',
			encoding='latin1'
		)

		n_coefs = hyper_surf.shape[0]
		new_coefs = np.zeros(n_coefs)

		for i in range(0, n_coefs):
			surf = hyper_surf[i]
			new_coefs[i] = surf.__call__([self.mu_value])

		self.output = np.dot(pod_basis, new_coefs)

	def run(self):
		"""
		This method runs the online evaluation.
		"""

		if self.is_scalar is True:
			self.perform_scalar()
		else:
			self.perform_field()

	def write_file(self, filename, infile):
		"""
		This method writes out the solution in the proper format. In this way, you can view the results
		with the viewer you like.
		
		:param string filename: name of the output file.
		:param string infile: name of the input file. For the mat file this is simply necessary for the parse function.
			for the vtk is necessary for having the correct grid for the new output field.
		
		"""
		__, file_ext = os.path.splitext(filename)

		if file_ext == '.mat':
			mat_handler = mh.MatlabHandler()
			mat_handler.parse(infile, self.output_name)
			mat_handler.write(self.output, filename)
		elif file_ext == '.vtk':
			vtk_handler = vh.VtkHandler()
			vtk_handler.parse(infile, self.output_name)
			vtk_handler.write(
				self.output, filename, output_name=self.output_name
			)
		else:
			raise NotImplementedError(
				file_ext + " file extension is not implemented yet."
			)

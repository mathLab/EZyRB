"""
Utilities for the online evaluation of the output of interest
"""
import numpy as np

class Online(object):
	"""
	Documentation
	
	:cvar numpy.ndarray mu_value: values of the parameters for the new evaluation of the output.
	:cvar string output_name: suffix of the files containing the structures for the online phase.
	:cvar string directory: directory where the structures are stored.
	:cvar bool is_scalar: boolean to set if the output of interest is a scalar or a field.
	:cvar numpy.ndarray output: new evaluation of the output of interest.

	"""
	
	def __init__(self, mu_value, output_name, directory='./', is_scalar=True):
		self.mu_value = mu_value
		self.directory = directory
		self.is_scalar = is_scalar
		self.output_name  = output_name
		self.output = None
		

	def perform_scalar(self):
		"""
		This method performs the online evaluation of the output if it is a scalar.
		"""
		
		hyper_surf  = np.load(self.directory + 'triangulation_scalar.npy')
		surf = hyper_surf.all()

		self.output = surf.__call__([self.mu_value])
			
			
	def perform_field(self):
		"""
		This method performs the online evaluation of the output if it is a field.
		"""
		
		hyper_surf = np.load(self.directory + 'coefs_tria_' + self.output_name + '.npy')
		pod_basis  = np.load(self.directory + 'pod_basis_' + self.output_name + '.npy')
		
		n_coefs = hyper_surf.shape[0]
		new_coefs  = np.zeros(n_coefs)
	
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
		
		
	

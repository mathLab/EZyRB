"""
Class for the Proper Orthogonal Decomposition
"""

import numpy as np
import ezyrb.vtkhandler as vh
import ezyrb.matlabhandler as mh
import ezyrb.cvt as cvt
import matplotlib.pyplot as plt
from scipy import interpolate

class Pod(object):
	"""
	Documentation
	
	:param string output_name: name of the variable (or output) we want to extract from the solution file.
	:param string weights_name: name of the weights to be extracted  from the solution file for the computation
		of the errors. If the solution files does not contain any weight (like volume or area of the cells) 
		the weight is set to 1 for all the cells.
	:param string namefile_prefix: path and prefix of the solution files. The files are supposed to be named with
		the same prefix, plus an increasing numeration (from 0) in the same order as the parameter points.
	:param string file_format: format of the solution files.
	
	:cvar string output_name: name of the variable (or output) we want to extract from the solution file.
	:cvar string weights_name: name of the weights to be extracted  from the solution file for the computation
		of the errors. If the solution files does not contain any weight (like volume or area of the cells) 
		the weight is set to 1 for all the cells.
	:cvar string namefile_prefix: path and prefix of the solution files. The files are supposed to be named with
		the same prefix, plus an increasing numeration (from 0) in the same order as the parameter points.
	:cvar string file_format: format of the solution files.
	:cvar numpy.ndarray mu_values: values of the parameters representing the vertices 
		of the triangulation of the parametric domain.
	:cvar numpy.ndarray pod_basis: basis extracted from the proper orthogonal decomposition.
	:cvar numpy.ndarray snapshots: database of the output of interest.
	:cvar numpy.ndarray weights: array of the weights for the computation of the error between
		high fidelity and reconstructed error. Tipically, it is the area/volume of each cell of
		the domain.
	:cvar Cvt cvt_handler: handler for the tesselation.
	:cvar FileHandler file_hanldler: handler for the file to be read and written.
	
	.. warning::
			The files containing the snapshots must be stored in the same directory and must have
			the same prefix, with a increasing numeration (from 0) in the same order as the parameter points.
			For example, in the directory tests/test_datasets/ you can find the files (matlab_00.vtk, 
			matlab_01.vtk, ... , matlab_05.vtk)
	
	"""
	
	def __init__(self, output_name, weights_name, namefile_prefix, file_format):
		self.output_name  = output_name
		self.weights_name = weights_name
		self.namefile_prefix = namefile_prefix
		self.file_format  = file_format
		self.mu_values = None
		self.pod_basis = None
		
		self.file_handler = vh.VtkHandler()
		
		if self.file_format == '.vtk':
			self.file_handler = vh.VtkHandler()
		elif self.file_format == '.mat':
			self.file_handler = mh.MatlabHandler()
			
		aux_snapshot = self.file_handler.parse(self.namefile_prefix + '0' + self.file_format, self.output_name)
		self.snapshots = aux_snapshot.reshape(aux_snapshot.shape[0],1)
		self.weights   = None
		self.cvt_handler = None
		
		
	def start(self):
		"""
		Compute the starting parameter sample and snapshots.		
		"""
	
		mu_1 = np.array([-.5, .5,  .5, -.5])
		mu_2 = np.array([-.5, -.5, .5,  .5])
		self.mu_values = np.array([mu_1, mu_2])
		
		dim_set = self.mu_values.shape[1]

		# TODO: insert an assert if the number of dim_set is different from the number of files for the extraction of the output
		for i in range(1,dim_set):
			aux_snapshot = self.file_handler.parse(self.namefile_prefix + str(i) + self.file_format, self.output_name)
			snapshot = aux_snapshot.reshape(aux_snapshot.shape[0],1)
			self.snapshots = np.append(self.snapshots, snapshot, 1)
		
		try:		
			weights = self.file_handler.parse(self.namefile_prefix + '0' + self.file_format, self.weights_name)
			
			if weights.shape[0] != snapshots.shape[0]: #vectorial field: to be improved for n-dimensional fields
				weights = np.append(weights, np.append(weights,weights,0), 0)
		except:
			weights = 0.*snapshot + 1.
			
		self.weights = weights[:,0]
		
		self.print_info()
	

	def print_info(self):
		"""
		This method compute and print the new parameter point for the next simulation and the maximum error
		in the tesselation.
		"""
		
		weighted_snapshots = np.sqrt(self.weights)*self.snapshots.T
		eigenvectors,eigenvalues,__ = np.linalg.svd(weighted_snapshots.T, full_matrices=False)
		self.pod_basis = np.transpose(np.power(self.weights,-0.5)*eigenvectors.T)
		
		self.cvt_handler = cvt.Cvt(self.mu_values, self.snapshots, self.pod_basis, self.weights)
		self.cvt_handler.add_new_point()
			
		print ('Maximum error on the tassellation: ' + str(self.cvt_handler.max_error))
		print ('New baricentric parameter value added to the triangulation ' + str(self.cvt_handler.mu_values[:,-1]) + '\n')
				
				
	def add_snapshot(self):
		"""
		This methos adds the new solution to the database and the new parameter values to the parameter points. 
		This can be done only after the new solution has be computed and placed in the proper directory.
		"""

		self.mu_values = self.cvt_handler.mu_values
		dim_mu = self.mu_values.shape[1]
		aux_snapshot = self.file_handler.parse(self.namefile_prefix + str(dim_mu-1) + self.file_format, self.output_name)
		snapshot = aux_snapshot.reshape(aux_snapshot.shape[0],1)
		self.snapshots = np.append(self.snapshots, snapshot, 1)
		
		self.print_info()
	
	
	def write_structures(self, plot_singular_values=False, directory='./'):
		"""
		This method and the offline step and writes out the structures necessary for the online step, that is,
		the pod basis and the triangulations for the coefficients for the interpolation of the pod basis.
		
		:param bool plot_singular_values: boolean to decide if we want to plot the singular values or not. 
			The default is false.
		:param string directory: directory where to save the pod basis and the coefficients.
			The default is the current directory.	
		"""
		
		weighted_snapshots = np.sqrt(self.weights)*self.snapshots.T
		eigenvectors,eigenvalues,__ = np.linalg.svd(weighted_snapshots.T, full_matrices=False)
		self.pod_basis = np.transpose(np.power(self.weights,-0.5)*eigenvectors.T)

		if plot_singular_values == True:
			plt.semilogy(np.linspace(0,eigenvalues.shape[0],eigenvalues.shape[0]), eigenvalues/eigenvalues[0])
			plt.show()

		#nBasis = input('Chose the number of basis functions ')	
		#u = u[:,:nBasis]
		
		n_points = self.mu_values.shape[1]
		n_basis  = self.pod_basis.shape[1]
		coefs_tria = np.array([])
		coefs = np.zeros([n_basis,n_points])

		for i in range(0,n_points):
			coefs[:,i] = np.dot(np.transpose(self.pod_basis), self.snapshots[:,i]*self.weights)

		for i in range(0,n_basis):
			coefs_surf = interpolate.LinearNDInterpolator(np.transpose(self.mu_values),coefs[i,:])
			coefs_tria = np.append(coefs_tria, coefs_surf)

		np.save(directory + 'coefs_tria_' + self.output_name, coefs_tria)
		np.save(directory + 'pod_basis_' + self.output_name, self.pod_basis)

		
		
		


"""
Class for the Interpolation of the scalar output of interest on the parameter space
"""

import numpy as np
import ezyrb.matlabhandler as mh
import ezyrb.cvt as cvt
from scipy import interpolate

class Interp(object):
	"""
	Documentation
	
	:cvar string output_name: name of the variable (or output) we want to extract from the solution file.
	:cvar string namefile_prefix: path and prefix of the solution files. The files are supposed to be named with
		the same prefix, plus an increasing numeration (from 0) in the same order as the parameter points.
	:cvar string file_format: format of the solution files.
	:cvar numpy.ndarray mu_values: values of the parameters representing the vertices 
		of the triangulation of the parametric domain.
	:cvar numpy.ndarray snapshots: database of the output of interest.
	
	.. warning::
			The files containing the snapshots must be stored in the same directory and must have
			the same prefix, with a increasing numeration (from 0) in the same order as the parameter points.
			For example, in the directory tests/test_datasets/ you can find the files (matlab_00.vtk, 
			matlab_01.vtk, ... , matlab_05.vtk)
	
	"""
	
	def __init__(self, output_name, namefile_prefix, file_format):
		self.output_name  = output_name
		self.namefile_prefix = namefile_prefix
		self.file_format  = file_format
		self.mu_values = None
		
		matlab_handler = mh.MatlabHandler()
		self.snapshots = matlab_handler.parse(self.namefile_prefix + '0' + self.file_format, self.output_name)
		
		
	def start(self):
		"""
		Compute the starting parameter sample and snapshots.		
		"""
	
		mu_1 = np.array([-.5, .5,  .5, -.5])
		mu_2 = np.array([-.5, -.5, .5,  .5])
		self.mu_values = np.array([mu_1, mu_2])
		
		dim_set = self.mu_values.shape[1]
		
		matlab_handler = mh.MatlabHandler()

		# TODO: insert an assert if the number of dim_set is different from the number of files for the extraction of the output
		for i in range(1,dim_set):
			snapshot = matlab_handler.parse(self.namefile_prefix + str(i) + self.file_format, self.output_name)
			self.snapshots = np.append(self.snapshots, snapshot, 1)
		
		self.print_info()
	

	def print_info(self):
		"""
		This method compute and print the new parameter point for the next simulation and the maximum error
		in the tesselation.
		"""
		
		self.cvt_handler = cvt.Cvt(self.mu_values, self.snapshots)
		self.cvt_handler.add_new_point()
			
		print ('Maximum error on the tassellation: ' + str(self.cvt_handler.max_error))
		print ('New baricentric parameter value added to the triangulation ' + str(self.cvt_handler.mu_values[:,-1]) + '\n')
				
				
	def add_snapshot(self):
		"""
		This methos adds the new solution to the database and the new parameter values to the parameter points. 
		This can be done only after the new solution has be computed and placed in the proper directory.
		"""
		
		matlab_handler = mh.MatlabHandler()
		self.mu_values = self.cvt_handler.mu_values
		dim_mu = self.mu_values.shape[1]
		snapshot = matlab_handler.parse(self.namefile_prefix + str(dim_mu-1) + self.file_format, self.output_name)
		
		self.snapshots = np.append(self.snapshots, snapshot, 1)
		
		self.print_info()
	
	
	def write_structures(self, directory='./'):
		"""
		This method and the offline step and writes out the structures necessary for the online step, that is,
		the pod basis and the triangulations for the coefficients for the interpolation of the pod basis.
		
		:param string directory: directory where to save the pod basis and the coefficients.
			The default is the current directory.	
		"""
		
		tria_surf = interpolate.LinearNDInterpolator(np.transpose(self.mu_values[:,:]), self.snapshots[0,:])
		
		np.save(directory + 'triangulation_scalar', tria_surf)

		
		
		


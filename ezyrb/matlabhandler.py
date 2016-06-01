"""
Derived module from filehandler.py to handle Matlab files.
"""
import numpy as np
import scipy.io as sio
import ezyrb.filehandler as fh	


class MatlabHandler(fh.FileHandler):
	"""
	Matlab format file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string extension: extension of the input/output files. It is equal to '.mat'.
	:cvar string output_name: name of the output of interest inside the mat file. Default value is output.
	
	..todo:
		Exception if output name is not a variable of mat file.
	"""
	def __init__(self):
		super(MatlabHandler, self).__init__()
		self.extension = '.mat'
		self.output_name = 'output'


	def parse(self, filename, output_name=None):
		"""
		Method to parse the `filename`. It returns a vector (matrix with one column) with all the values of the chosen output.
		If `output_name` is not given it is set to the default value.

		:param string filename: name of the input file.
		:param string output_name: name of the output of interest inside the mat file. 
			If it is not passed, it is equal to self.output_name.
		
		:return: output_values: it is a `n_points`-by-1 matrix containing the values of the chosen output
		:rtype: numpy.ndarray
		"""
		
		if output_name is None:
			output_name = self.output_name
		else:
			self._check_filename_type(output_name)
		
		self._check_filename_type(filename)
		self._check_extension(filename)

		self.infile = filename

		loaded_struct = sio.loadmat(self.infile)
		output_values = np.array(loaded_struct[output_name])

		return output_values


	def write(self, output_values, filename):
		"""
		Writes a mat file, called filename. output_values is a matrix that contains the new values of the output 
		to write in the mat file.

		:param numpy.ndarray output_values: it is a `n_points`-by-1 matrix containing the values of the chosen output.
		:param string filename: name of the output file.
		"""
		
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)
		
		sio.savemat(filename, dict(output=output_values))
		

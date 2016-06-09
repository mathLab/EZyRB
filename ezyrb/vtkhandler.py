"""
Derived module from filehandler.py to handle Vtk files.
"""
import numpy as np
import scipy.io as sio
import ezyrb.filehandler as fh
import vtk
import vtk.util.numpy_support as ns


class VtkHandler(fh.FileHandler):
	"""
	Vtk format file handler class

	:cvar string infile: name of the input file to be processed.
	:cvar string extension: extension of the input/output files. It is equal to '.vtk'.
	:cvar string output_name: name of the output of interest inside the mat file. Default value is output.
	
	..todo:
		Exception if output name is not a variable of vtk file.
	"""
	def __init__(self):
		super(VtkHandler, self).__init__()
		self.extension = '.vtk'
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

		reader = vtk.vtkDataSetReader()
		reader.SetFileName(filename)
		reader.ReadAllVectorsOn()
		reader.ReadAllScalarsOn()
		reader.Update()
		data = reader.GetOutput()

		# TODO: insert a switch to decide if we have cell data oppure point data
		#extracted_data = data.GetCellData().GetArray(output_name)
		extracted_data = data.GetPointData().GetArray(output_name)
		
		# TODO: check if the output is a scalar or vector
		data_dim = extracted_data.GetSize()
		output_values = np.zeros([data_dim,1])

		for i in range (0,data_dim):
			output_values[i] = extracted_data.GetValue(i)

		return output_values


	def write(self, output_values, filename, output_name=None, write_bin=False):
		"""
		Writes a mat file, called filename. output_values is a matrix that contains the new values of the output 
		to write in the mat file.

		:param numpy.ndarray output_values: it is a `n_points`-by-1 matrix containing the values of the chosen output.
		:param string filename: name of the output file.
		:param string output_name: name of the output of interest inside the mat file. 
			If it is not passed, it is equal to self.output_name.
		:param boolean write_bin: flag to write in the binary format. Default is False.
		"""
		
		self._check_filename_type(filename)
		self._check_extension(filename)
		self._check_infile_instantiation(self.infile)
		
		if output_name is None:
			output_name = self.output_name
		else:
			self._check_filename_type(output_name)
		
		reader = vtk.vtkDataSetReader()
		reader.SetFileName(self.infile)
		reader.ReadAllVectorsOn()
		reader.ReadAllScalarsOn()
		reader.Update()
		data = reader.GetOutput()
	
		output_array = ns.numpy_to_vtk(num_array=output_values,array_type=vtk.VTK_DOUBLE)
		output_array.SetName(output_name)
		# TODO: insert a switch to decide if we have cell data oppure point data
		#data.GetCellData().AddArray(output_array)
		data.GetPointData().AddArray(output_array)
	
	
		writer = vtk.vtkDataSetWriter()
		
		if write_bin:
			writer.SetFileTypeToBinary()
		
		writer.SetFileName(filename)
		
		if vtk.VTK_MAJOR_VERSION <= 5:
			writer.SetInput(data)
		else:
			writer.SetInputData(data)
	
		writer.Write()
		

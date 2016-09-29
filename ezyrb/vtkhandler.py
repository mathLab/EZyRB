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
	:cvar bool cell_data: boolean that is True if the output of interest is a cell data array in the vtk file. 
		If it is False, it is a point data array.
	
	.. todo::
		Exception if output name is not a variable of vtk file.
	"""

	def __init__(self):
		super(VtkHandler, self).__init__()
		self.extension = '.vtk'
		self.output_name = 'output'
		self.cell_data = None

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

		extracted_data_cell = data.GetCellData().GetArray(output_name)
		extracted_data_point = data.GetPointData().GetArray(output_name)

		if extracted_data_cell is None:
			extracted_data = extracted_data_point
			self.cell_data = False
		else:
			extracted_data = extracted_data_cell
			self.cell_data = True

		# TODO: check if the output is a scalar or vector
		output_values = ns.vtk_to_numpy(extracted_data).reshape((-1, 1))

		return output_values

	def write(self, output_values, filename, output_name=None, write_bin=False):
		"""
		Writes a mat file, called filename. output_values is a matrix that contains the new values of the output 
		to write in the mat file.

		:param numpy.ndarray output_values: it is a `n_points`-by-1 matrix containing the values of the chosen output.
		:param string filename: name of the output file.
		:param string output_name: name of the output of interest inside the mat file. 
			If it is not passed, it is equal to self.output_name.
		:param bool write_bin: flag to write in the binary format. Default is False.
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

		output_array = ns.numpy_to_vtk(
			num_array=output_values, array_type=vtk.VTK_DOUBLE
		)
		output_array.SetName(output_name)

		if self.cell_data is True:
			data.GetCellData().AddArray(output_array)
		else:
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

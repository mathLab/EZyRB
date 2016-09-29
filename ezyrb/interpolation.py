"""
Class for the Proper Orthogonal Decomposition
"""

import os
import numpy as np
import ezyrb.matlabhandler as mh
import ezyrb.cvt as cvt
from scipy import interpolate

##
## Python 2 vs Python 3 conflicts
try:
	import configparser
except:
	import ConfigParser as configparser


class Interp(object):
	"""
	Documentation

	:param string output_name: name of the variable (or output) we want to
		extract from the solution file.
	:param string namefile_prefix: path and prefix of the solution files.
		The files are supposed to be named with the same prefix, plus an
		increasing numeration (from 0) in the same order as the parameter
		points.
	:param string file_format: format of the solution files.

	:cvar string output_name: name of the variable (or output) we want to
		extract from the solution file.
	:cvar string namefile_prefix: path and prefix of the solution files.
		The files are supposed to be named with the same prefix, plus an
		increasing numeration (from 0) in the same order as the parameter
		points.
	:cvar string file_format: format of the solution files.
	:cvar numpy.ndarray mu_values: values of the parameters representing the
		vertices of the triangulation of the parametric domain.
	:cvar numpy.ndarray snapshots: database of the output of interest.
	:cvar Cvt cvt_handler: handler for the tesselation.
	:cvar FileHandler file_hanldler: handler for the file to be read and
		written.

	.. warning::
		The files containing the snapshots must be stored in the same directory
		and must have the same prefix, with a increasing numeration (from 0) in
		the same order as the parameter points. For example, in the directory
		tests/test_datasets/ you can find the files (matlab_00.vtk,
		matlab_01.vtk, ..., matlab_05.vtk)

	"""

	def __init__(self, output_name, namefile_prefix, file_format):

		self._config_file = './setting.conf'
		self._num_mu = 0
		self._dimension_mu = 0	# dimension of mu values
		self._mu_file = './mu.conf'
		self._num_mu_values = 0
		self._snapshot_files = None
		self._new_optimal_mu = None
		self._max_error = None
		self.output_name = output_name
		self.namefile_prefix = namefile_prefix
		self.file_format = file_format
		self.mu_values = None

		self.file_handler = mh.MatlabHandler()

		self.snapshots = None
		self.cvt_handler = None

	def read_config(self, config_file=None):
		"""
		Read the configuration file; if is not specified the file,
		file parsed is ./setting.conf

		:param string config_file: configuration file to parse
		"""

		# take default file if no one is given
		if not config_file:
			config_file = self._config_file

		if not os.path.isfile(config_file):
			raise RuntimeError(
				"File {0!s} not found".format(os.path.abspath(config_file))
			)

		config = configparser.RawConfigParser()
		config.read(config_file)

		if config.has_option('Default', 'parameter dimension'):
			self._dimension_mu = config.getint('Default', 'parameter dimension')

		if config.has_option('Default', 'mu file'):
			self._mu_file = config.get('Default', 'mu file')

		if config.has_option('Default', 'namefile prefix'):
			self.namefile_prefix = config.get('Default', 'namefile prefix')

	def initialize_snapshot(self):
		"""
		Compute the starting parameter sample and snapshots.
		"""

		if not os.path.isfile(self._mu_file):
			raise RuntimeError(
				"File {0!s} not found".format(os.path.abspath(self._mu_file))
			)

		# Read the first `self._dimension_mu` columns to build
		# ndarray containing mu_values
		mu_col = tuple(range(self._dimension_mu))
		self.mu_values = np.genfromtxt(
			self._mu_file, dtype=np.float, usecols=mu_col
		).T

		self._num_mu_values = self.mu_values.shape[1]

		# if last column contains name of snapshot files, take them;
		# if not, create a sequence using namefile prefix
		try:
			self._snapshot_files = np.genfromtxt(
				self._mu_file, usecols=tuple(self._dimension_mu)
			)
		except:
			self._snapshot_files = np.array([
				self.namefile_prefix + str(i) + self.file_format
				for i in np.arange(self._num_mu_values)
			])

		aux_snapshot = self.file_handler.parse(
			self._snapshot_files[0], self.output_name
		)
		self.snapshots = aux_snapshot.reshape(aux_snapshot.shape[0], 1)

		for i in np.arange(1, self._num_mu_values):
			aux_snapshot = self.file_handler.parse(
				self._snapshot_files[i], self.output_name
			)
			snapshot = aux_snapshot.reshape(aux_snapshot.shape[0], 1)
			self.snapshots = np.append(self.snapshots, snapshot, 1)

	def get_cvt_handler(self):
		"""
		Return Central Voronoi Tasselation object; if object is not
		initialize yet, create it.
		"""
		if not self.cvt_handler:

			# TODO Add some checks
			self.cvt_handler = cvt.Cvt(self.mu_values, self.snapshots)

		return self.cvt_handler

	def find_optimal_mu(self):
		"""
		This method compute the new parameter point for the next simulation.
		"""
		if not self._new_optimal_mu:
			self._new_optimal_mu = self.get_cvt_handler().get_optimal_new_mu()
		return self._new_optimal_mu

	def find_max_error(self):
		"""
		This method compute the maximum error in tasselation.
		"""
		if not self._max_error:
			self._max_error = self.get_cvt_handler().get_max_error()
		return self._max_error

	def print_info(self):
		"""
		Simply print all information of class
		"""
		print(
			'Maximum error on the tassellation: {0!s}'.
			format(self.find_max_error())
		)

		print(
			'New baricentric parameter value added to the triangulation',
			self.cvt_handler.get_optimal_new_mu()
		)

	def add_snapshot(self, new_mu=None, snapshot_file=None):
		"""
		This methos adds the new solution to the database and the new parameter
		values to the parameter points; this can be done only after the new
		solution has be computed and placed in the proper directory.

		:param nem_mu numpy.ndarray
		:param snapshot_file string
		"""

		if new_mu is None:
			new_mu = self.find_optimal_mu()

		# mu_values are stored by column, so need to transpose it
		new_mu = new_mu.reshape((-1, 1))

		if snapshot_file is None:
			if self.namefile_prefix:
				snapshot_file = self.namefile_prefix + str(
					self._num_mu_values
				) + self.file_format
			else:
				raise RuntimeError(
					"You need to specify a namefile prefix"
					" or specific file for new snapshot"
				)

		if not os.path.isfile(snapshot_file):
			raise RuntimeError(
				"File {0!s} not found".format(os.path.abspath(snapshot_file))
			)

		# Add snapshot
		aux_snapshot = self.file_handler.parse(snapshot_file, self.output_name)
		snapshot = aux_snapshot.reshape(aux_snapshot.shape[0], 1)
		if self.snapshots is not None:
			self.snapshots = np.append(self.snapshots, snapshot, axis=1)
		else:
			self.snapshots = snapshot

		# Add mu_values
		if self.mu_values is not None:
			self.mu_values = np.append(self.mu_values, new_mu, axis=1)
		else:
			self.mu_values = new_mu

		# Whenever add a snapshot, everything need to be recomputed
		self.cvt_handler = None
		self._max_error = 0
		self._new_optimal_mu = None

	def write_structures(self, directory='./'):
		"""
		This method and the offline step and writes out the structures necessary
		for the online step, that is, the pod basis and the triangulations for
		the coefficients for the interpolation of the pod basis.

		:param bool plot_singular_values: boolean to decide if we want to plot
			the singular values or not. The default is false.
		:param string directory: directory where to save the pod basis and the
			coefficients. The default is the current directory.
		"""
		tria_surf = interpolate.LinearNDInterpolator(
			np.transpose(self.mu_values[:, :]), self.snapshots[0, :]
		)

		np.save(directory + 'triangulation_scalar', tria_surf)

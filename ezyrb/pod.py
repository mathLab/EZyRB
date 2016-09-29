"""
Class for the Proper Orthogonal Decomposition
"""

import os
import numpy as np
import ezyrb.vtkhandler as vh
import ezyrb.matlabhandler as mh
import ezyrb.cvt as cvt
import matplotlib.pyplot as plt
from scipy import interpolate

##
## Python 2 vs Python 3 conflicts
try:
	import configparser
except:
	import ConfigParser as configparser


class Pod(object):
	"""
	Documentation

	:param string output_name: name of the variable (or output) we want to
		extract from the solution file.
	:param string weights_name: name of the weights to be extracted	 from the
		solution file for the computation of the errors. If the solution files
		does not contain any weight (like volume or area of the cells) the
		weight is set to 1 for all the cells.
	:param string namefile_prefix: path and prefix of the solution files.
		The files are supposed to be named with the same prefix, plus an
		increasing numeration (from 0) in the same order as the parameter
		points.
	:param string file_format: format of the solution files.

	:cvar string output_name: name of the variable (or output) we want to
		extract from the solution file.
	:cvar string weights_name: name of the weights to be extracted	from the
		solution file for the computation of the errors. If the solution files
		does not contain any weight (like volume or area of the cells) the
		weight is set to 1 for all the cells.
	:cvar string namefile_prefix: path and prefix of the solution files.
		The files are supposed to be named with the same prefix, plus an
		increasing numeration (from 0) in the same order as the parameter
		points.
	:cvar string file_format: format of the solution files.
	:cvar numpy.ndarray mu_values: values of the parameters representing the
		vertices of the triangulation of the parametric domain.
	:cvar numpy.ndarray pod_basis: basis extracted from the proper orthogonal
		decomposition.
	:cvar numpy.ndarray snapshots: database of the output of interest.
	:cvar numpy.ndarray weights: array of the weights for the computation of
		the error between high fidelity and reconstructed error. Tipically, it
		is the area/volume of each cell of the domain.
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

	def __init__(self, output_name, weights_name, namefile_prefix, file_format):

		self._config_file = './setting.conf'
		self._num_mu = 0
		self._dimension_mu = 0	# dimension of mu values
		self._mu_file = './mu.conf'
		self._num_mu_values = 0
		self._snapshot_files = None
		self._new_optimal_mu = None
		self._max_error = None
		self.output_name = output_name
		self.weights_name = weights_name
		self.namefile_prefix = namefile_prefix
		self.file_format = file_format
		self.mu_values = None
		self.pod_basis = None

		if self.file_format == '.vtk':
			self.file_handler = vh.VtkHandler()
		elif self.file_format == '.mat':
			self.file_handler = mh.MatlabHandler()
		else:
			raise RuntimeError("file_format is not supported")

		self.snapshots = None
		self.weights = None
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

		try:
			weights = self.file_handler.parse(
				self._snapshot_files[0], self.weights_name
			)

			#vectorial field: to be improved for n-dimensional fields
			if weights.shape[0] != snapshots.shape[0]:
				weights = np.append(weights, np.append(weights, weights, 0), 0)

		except:
			weights = 0. * snapshot + 1.

		self.weights = weights[:, 0]

	def compute_pod_basis(self):
		"""
		Return POD basis; if basis is not computed yes, compute them
		"""
		weighted_snapshots = np.sqrt(self.weights) * self.snapshots.T
		eigenvectors, eigenvalues, __ = np.linalg.svd(
			weighted_snapshots.T, full_matrices=False
		)
		self.pod_basis = np.transpose(
			np.power(self.weights, -0.5) * eigenvectors.T
		)

		return eigenvectors, eigenvalues

	def get_cvt_handler(self):
		"""
		Return Central Voronoi Tasselation object; if object is not
		initialize yet, create it.
		"""
		if not self.cvt_handler:
			self.compute_pod_basis()

			# TODO Add some checks
			self.cvt_handler = cvt.Cvt(
				self.mu_values, self.snapshots, self.pod_basis, self.weights
			)

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
		self.pod_basis = None
		self.cvt_handler = None
		self._max_error = 0
		self._new_optimal_mu = None

	def write_structures(self, plot_singular_values=False, directory='./'):
		"""
		This method and the offline step and writes out the structures necessary
		for the online step, that is, the pod basis and the triangulations for
		the coefficients for the interpolation of the pod basis.

		:param bool plot_singular_values: boolean to decide if we want to plot
			the singular values or not. The default is false.
		:param string directory: directory where to save the pod basis and the
			coefficients. The default is the current directory.
		"""

		__, eigenvalues = self.compute_pod_basis()

		if plot_singular_values:
			plt.semilogy(
				np.linspace(0, eigenvalues.shape[0], eigenvalues.shape[0]),
				eigenvalues / eigenvalues[0]
			)
			plt.show()

		n_points = self.mu_values.shape[1]
		n_basis = self.pod_basis.shape[1]
		coefs_tria = np.array([])
		coefs = np.zeros([n_basis, n_points])

		for i in range(0, n_points):
			coefs[:, i] = np.dot(
				np.transpose(self.pod_basis),
				self.snapshots[:, i] * self.weights
			)

		for i in range(0, n_basis):
			coefs_surf = interpolate.LinearNDInterpolator(
				np.transpose(self.mu_values), coefs[i, :]
			)
			coefs_tria = np.append(coefs_tria, coefs_surf)

		np.save(directory + 'coefs_tria_' + self.output_name, coefs_tria)
		np.save(directory + 'pod_basis_' + self.output_name, self.pod_basis)

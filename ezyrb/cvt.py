"""
Class for the Centroidal Voronoi Tesseletion

.. todo::
	- add the show method

"""
import math
import numpy as np
from scipy.spatial import Delaunay
from scipy import interpolate


class Cvt(object):
	"""
	Documentation

	:param numpy.ndarray mu_values: values of the parameters representing the
		vertices of the triangulation of the parametric domain.
	:param numpy.ndarray pod_basis: basis extracted from the proper orthogonal
		decomposition.
	:param numpy.ndarray snapshots: database of the output of interest.
	:param numpy.ndarray weights: array of the weights for the computation of
		the error between high fidelity and reconstructed error. Tipically,
		it is the area/volume of each cell of the domain.

	:cvar numpy.ndarray mu_values: values of the parameters representing the
		vertices of the triangulation of the parametric domain.
	:cvar numpy.ndarray pod_basis: basis extracted from the proper orthogonal
		decomposition.
	:cvar numpy.ndarray snapshots: database of the output of interest.
	:cvar numpy.ndarray weights: array of the weights for the computation of
		the error between high fidelity and reconstructed error. Tipically,
		it is the area/volume of each cell of the domain.
	:cvar int dim_out: dimension of the output of interest.
	:cvar int dim_db: number of the output of interest on the snapshot matrix
		(dimension of the database).
	:cvar float rel_error: coefficient to make the computed errors relative to
		the magnitude of the output. If the output is a field, it is computed
		as the L2 value of the first snapshot, if the output is a scalar, it is
		the absolute value of the first snapshot.
	:cvar float max_error: max error of the leave-one-out strategy on the
		present outputs in the snapshots array.
	:cvar int dim_mu: dimension of the parametric space.
	:cvar boolean _offline_using_pod: indicate which method was used
		(Pod/Interpolation) to perform offline phase
	:cvar private __error_estimator_func: the function to estimate error

	"""

	def __init__(self, mu_values, snapshots, pod_basis=None, weights=None):
		self.mu_values = mu_values
		self.pod_basis = pod_basis
		self.snapshots = snapshots
		self.weights = weights
		self.dim_out, self.dim_db = self.snapshots.shape
		self.__error_estimator_func = lambda x: np.linalg.norm(x, 2)

		ref_solution = self.snapshots[:, 0]

		self._offline_using_pod = False if self.weights is None else True

		if self._offline_using_pod:
			self.rel_error = np.linalg.norm(ref_solution * self.weights, 2)
		else:  # offline_using_interpolation
			self.rel_error = np.abs(ref_solution)

		self.dim_mu = mu_values.shape[0]

	@staticmethod
	def _compute_simplex_volume(simplex_vertices):
		"""
		Method implementing the computation of the volume of a N dimensional
		simplex.
		Source from wikipedia https://en.wikipedia.org/wiki/Simplex

		:param numpy.ndarray simplex_vertices: Nx3 array containing the
			parameter values representing the vertices of a simplex. N is the
			dimensionality of the parameters.

		:return: volume: N dimensional volume of the simplex.
		:rtype: float
		"""

		vertex_0 = simplex_vertices[:, 0]
		vertex_0 = np.array([vertex_0])

		simplex_vertices = np.delete(simplex_vertices, 0, 1)
		par_dim = simplex_vertices.shape[1]

		distance = simplex_vertices - np.dot(vertex_0.T, np.ones([1, par_dim]))
		volume = np.abs(np.linalg.det(distance) / math.factorial(par_dim))

		return volume

	@property
	def error_estimator(self):
		"""
		Function to estimate error in compute_error method

		:setter: Returns error estimator function
		:getter: Sets error estimator function
		:type: function
		"""
		return self.__error_estimator_func

	@error_estimator.setter
	def error_estimator(self, func):

		if not callable(func):
			raise TypeError("Not valid estimator error function.")

		self.__error_estimator_func = func

	def compute_errors_pod(self):
		"""
		Compute the error for each parametric point as projection of the
		snapshot onto the POD basis with a leave-one-out (loo) strategy.
		To use when offline phase was computed using Pod class.

		:return: error: error array of the leave-one-out strategy.
		:rtype: numpy.ndarray
		"""
		loo_error = np.zeros(self.dim_db)

		for j in np.arange(self.dim_db):

			remaining_snaps = np.delete(self.snapshots, j, 1)

			weighted_remaining_snaps = np.sqrt(self.weights) * remaining_snaps.T
			eigenvectors, __, __ = np.linalg.svd(
				weighted_remaining_snaps.T, full_matrices=False
			)
			loo_basis = np.transpose(
				np.power(self.weights, -0.5) * eigenvectors.T
			)

			projection = np.zeros(self.dim_out)
			snapshot = self.snapshots[:, j]

			for i in range(0, self.dim_db - 1):
				projection += np.dot(snapshot * self.weights,
									 loo_basis[:, i]) * loo_basis[:, i]

			error = (snapshot - projection) * self.weights
			loo_error[j] = self.__error_estimator_func(error) / self.rel_error

		return loo_error

	def compute_errors_interpolation(self):
		"""
		Compute the error for each parametric point as projection 
		interpolated using snapshots with a leave-one-out (loo) strategy.
		To use when offline phase was computed using Interpolation class.

		:return: error: error array of the leave-one-out strategy.
		:rtype: numpy.ndarray

		"""

		loo_error = np.zeros(self.dim_db)

		for j in np.arange(self.dim_db):

			remaining_snaps = np.delete(self.snapshots, j, 1)
			remaining_mu = np.delete(self.mu_values, j, 1)
			remaining_tria = interpolate.LinearNDInterpolator(
				np.transpose(remaining_mu[:, :]), remaining_snaps[0, :]
			)

			projection = remaining_tria.__call__(self.mu_values[:, j])

			if projection is not float:
				projection = np.sum(remaining_snaps) / (self.dim_db - 1)
			loo_error[j] = np.abs(self.snapshots[:, j] - projection
								  ) / self.rel_error

		return loo_error

	def compute_errors(self):
		"""
		Compute the error for each parametric point using the appropriate
		method according to the offline phase.

		:return: error: error array of the leave-one-out strategy.
		:rtype: numpy.ndarray
		"""
		if self._offline_using_pod:
			func = self.compute_errors_pod
		else:  # offline_using_interpolation
			func = self.compute_errors_interpolation

		return func()

	def get_max_error(self):
		"""
		Return the max error using leave-one-out

		:return: max_error: maximum error
		:rtype: numpy.ndarray

		"""
		max_error = np.max(self.compute_errors())
		return max_error

	def get_optimal_new_mu(self):
		"""
		This method return the baricentric point of the worst approximated
		simplex by the already computed pod basis.

		:return: new_point: Baricentric point of worst approximated simplex.
		:rtype: numpy.array
		"""

		error = self.compute_errors()

		tria = Delaunay(np.transpose(self.mu_values))
		simplex = tria.simplices
		simp_dim_n = simplex.shape[0]
		error_on_simplex = np.zeros(simp_dim_n)

		for i in np.arange(simp_dim_n):
			points_of_simplex = self.mu_values[:, simplex[i]]
			volume = self._compute_simplex_volume(points_of_simplex)
			error_on_simplex[i] = np.sum(error[simplex[i]]) * volume

		worst_tria_ind = np.argmax(error_on_simplex)
		worst_tria_points = self.mu_values[:, simplex[worst_tria_ind]]
		worst_tria_err = error[simplex[worst_tria_ind]]
		new_point = np.zeros(self.dim_mu)

		for i in range(0, self.dim_mu):
			new_point[i] = np.sum(
				np.dot(worst_tria_points[i, :], worst_tria_err)
			) / np.sum(worst_tria_err)

		return new_point

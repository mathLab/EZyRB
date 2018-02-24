import numpy as np
from scipy.spatial.distance

class RBFInterpolator:

    @staticmethod
    def multi_quadratic(X, r):
        return np.sqrt(r ** 2 + X ** 2)

    def __init__(self, points, values, radius=1.0, norm='euclidean', basis=None):

        self.basis = basis
        self.points = points
        self.radius = radius

        if self.basis is None:
            self.basis = self.multi_quadratic

        distance_matrix = self.basis(scipy.spatial.distance.cdist(points, points, metric=norm), self.radius)
        self.weights = np.linalg.solve(distance_matrix, values)

    def __call__(self, new_points):
        distance_matrix = scipy.spatial.distance.cdist(new_points, self.points)
        return self.basis(distance_matrix, self.radius).dot(self.weights)

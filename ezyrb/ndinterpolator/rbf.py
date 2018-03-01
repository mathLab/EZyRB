import numpy as np
from scipy.spatial.distance import cdist as distance_matrix


class RBFInterpolator:
    @staticmethod
    def multi_quadratic(X, r):
        return np.sqrt(r**2 + X**2)

    def __init__(self, points, values, radius=1.0, norm='euclidean',
                 basis=None):
        self.basis = basis
        self.points = points
        self.radius = radius

        if self.basis is None:
            self.basis = self.multi_quadratic

        self.weights = np.linalg.solve(
            self.basis(
                distance_matrix(points, points, metric=norm), self.radius),
            values)

    def __call__(self, new_points):
        return self.basis(
            distance_matrix(new_points, self.points), self.radius).dot(
                self.weights)

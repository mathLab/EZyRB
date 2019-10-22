"""
"""
import numpy as np

from .reduction import Reduction


class POD(Reduction):
    def __init__(self, method='svd', **kwargs):
        """
        """
        available_methods = {
            'svd': (self._svd, {
                'rank': -1
            }),
            'randomized_svd': (self._rsvd, {
                'rank': -1
            }),
            'correlation_matrix': (self._corrm, {
                'rank': -1,
                'save_memory': False
            }),
        }

        self._modes = None
        self._singular_values = None

        method = available_methods.get(method)
        if method is None:
            raise RuntimeError(
                "Invalid method for POD. Please chose one among {}".format(
                    ', '.join(available_methods)))

        self.__method, default_args = method
        kwargs.update(default_args)

        for hyperparam, value in kwargs.items():
            setattr(self, hyperparam, value)

    @property
    def modes(self):
        """
        The POD modes.

        :type: numpy.ndarray
        """
        return self._modes

    @property
    def singular_values(self):
        """
        The singular values  

        :type: numpy.ndarray
        """
        return self._singular_values

    def reduce(self, X):
        """
        Reduces the parameter Space by using the specified reduction method (default svd).

        :type: numpy.ndarray
        """
        self._modes, self._singular_values = self.__method(X)
        return self.modes.T.dot(X)

    def expand(self, X):
        """
        Projects a reduced to full order solution.

        :type: numpy.ndarray
        """
        return np.sum(self.modes * X, axis=1)

    def _truncation(self, X):
        """
        Return the number of modes to select according to the `rank` parameter.
        See POD.__init__ for further info.

        :return: the number of modes
        :rtype: int
        """
        if self.rank is 0:
            omega = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif self.rank > 0 and svd_rank < 1:
            cumulative_energy = np.cumsum(s**2 / (s**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif self.rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        return rank

    def _svd(self, X):
        """
        Truncated Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s = np.linalg.svd(X, full_matrices=False)[:2]

        rank = self._truncation(X)
        return U[:, :rank], s[:rank]

    def _rsvd(self, X):
        """
        Truncated randomized Singular Value Decomposition.
        
        :param numpy.ndarray X: the matrix to decompose.
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        P = np.random.rand(X.shape[1], X.shape[0])
        Q = np.linalg.qr(X.dot(P))[0]

        Y = Q.T.conj().dot(X)

        Uy, s = np.linalg.svd(Y, full_matrices=False)[:2]
        U = Q.dot(Uy)

        rank = self._truncation(X)
        return U[:, :rank], s[:rank]

    def _corrm(self, X):
        """
        Truncated Singular Value Decomposition. calculated with correlation matrix.
        
        :param numpy.ndarray X: the matrix to decompose.
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """

        if self.save_memory:
            corr = np.empty(size=(X.shape[1], X.shape[1]))
            for i, i_snap in enumerate(X.T):
                for j, k_snap in enumerate(X.T):
                    corr[i, j] = np.inner(i_snap, k_snap)

        else:
            corr = X.T.dot(X)

        s, U = np.linalg.eig(corr)
        U = X.dot(U) / np.sqrt(s)
        rank = self._truncation(X)

        return U[:, :rank], s[:rank]

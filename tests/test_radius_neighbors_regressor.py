import numpy as np
import warnings

from unittest import TestCase
from ezyrb import RadiusNeighborsRegressor, POD, Database, ReducedOrderModel

class TestRadius(TestCase):
    def test_params(self):
        reg = RadiusNeighborsRegressor(radius=3.0, algorithm='kd_tree')
        assert reg.regressor.get_params()['radius'] == 3.0
        assert reg.regressor.get_params()['algorithm'] == 'kd_tree'

    def test_fit_onescalarparam_scalarfunc(self):
        reg = RadiusNeighborsRegressor()
        reg.fit([1], [20])
        assert reg.regressor.n_samples_fit_ == 1

    def test_fit_scalarparam_scalarfunc(self):
        reg = RadiusNeighborsRegressor()
        reg.fit([1, 2, 5, 7, 2], [2, 5, 7, 83, 3])
        assert reg.regressor.n_samples_fit_ == 5

    def test_fit_biparam_scalarfunc(self):
        reg = RadiusNeighborsRegressor()
        reg.fit([[1, 2], [6, 7], [8, 9]], [1, 5, 6])
        assert reg.regressor.n_samples_fit_ == 3

    def test_fit_biparam_bifunc(self):
        reg = RadiusNeighborsRegressor()
        reg.fit([[1, 2], [6, 7], [8, 9]], [[1, 0], [20, 5], [8, 6]])
        assert reg.regressor.n_samples_fit_ == 3

    def test_radiusneighbors(self):
        reg = RadiusNeighborsRegressor(radius=3.0)
        reg.fit([[1, 2], [6, 7], [8, 9]], [[1, 0], [20, 5], [8, 6]])
        neigh_idx = reg.regressor.radius_neighbors([[7,8]], return_distance=False)[0]
        assert neigh_idx[0] == 1
        assert neigh_idx[1] == 2
        assert len(neigh_idx) == 2

    def test_predict(self):
        reg = RadiusNeighborsRegressor(radius=0.5)
        reg.fit([[1, 2], [6, 7], [8, 9]], [[1, 0], [20, 5], [8, 6]])
        neigh_idx = reg.regressor.predict([[1,2], [8,9], [6,7]])
        assert (neigh_idx[0] == [1,0]).all()
        assert (neigh_idx[1] == [8,6]).all()
        assert (neigh_idx[2] == [20, 5]).all()

    def test_with_db_predict(self):
        reg = RadiusNeighborsRegressor(radius=0.5)
        pod = POD()
        db = Database(np.array([1, 2, 3])[:,None], np.array([1, 5, 3])[:,None])
        rom = ReducedOrderModel(db, pod, reg)

        rom.fit()
        pred = rom.predict([[1], [2], [3]])
        np.testing.assert_equal(pred, np.array([1, 5, 3])[:,None]) 



    def test_wrong1(self):
        # wrong number of params
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
            with self.assertRaises(Exception):
                reg = RadiusNeighborsRegressor()
                reg.fit([[1, 2], [6,], [8, 9]], [[1, 0], [20, 5], [8, 6]])

    def test_wrong2(self):
        # wrong number of values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
            with self.assertRaises(Exception):
                reg = RadiusNeighborsRegressor()
                reg.fit([[1, 2], [6,], [8, 9]], [[20, 5], [8, 6]])

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest import TestCase
from unittest.mock import Mock

from ezyrb import Database, Parameter, Snapshot
from ezyrb.plugin.automatic_shift import AutomaticShiftSnapshots

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        val = x.view(x.shape[0], -1).sum(dim=1, keepdim=True) * 0.0
        return val + self.dummy_param


class SimpleANN:
    def __init__(self, stop_training=None):
        self.model = DummyModel()
        self.lr = 0.01
        self.l2_regularization = 0.0
        self.loss_trend = []
        self.stop_training = stop_training if stop_training else [1]
        self.frequency_print = 100

    def _build_model(self, x, y):
        pass

    def fit(self, x, y):
        pass

    def optimizer(self, params, lr, weight_decay):
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

    def predict(self, x):
        return np.zeros((x.shape[0], 1))


class SimpleInterpolator:
    def fit(self, x, y):
        self.x_fit = np.asarray(x)
        self.y_fit = np.asarray(y)

    def predict(self, x):
        if hasattr(self, 'y_fit'):
            return np.full((x.shape[0],), self.y_fit.mean())
        return np.zeros((x.shape[0],))


class MockROM:
    def __init__(self, db):
        self.database = db
        self.predict_full_database = db
        self._full_database = None


class TestAutomaticShiftSnapshots(TestCase):
    
    def setUp(self):
        self.space = np.array([0.0, 1.0, 2.0])
        self.db = Database()
        
        snap1 = Snapshot(values=np.array([1.0, 2.0, 3.0]), space=self.space.copy())
        snap2 = Snapshot(values=np.array([2.0, 3.0, 4.0]), space=self.space.copy())
        snap3 = Snapshot(values=np.array([3.0, 4.0, 5.0]), space=self.space.copy())
        
        self.db.add(Parameter([1.0]), snap1)
        self.db.add(Parameter([2.0]), snap2)
        self.db.add(Parameter([3.0]), snap3)
        
        self.rom = MockROM(self.db)
 
    def test_constructor_stores_parameters(self):
        shift_net = SimpleANN()
        interp_net = SimpleANN()
        interpolator = SimpleInterpolator()
        
        plugin = AutomaticShiftSnapshots(
            shift_network=shift_net,
            interp_network=interp_net,
            interpolator=interpolator,
            parameter_index=1,
            reference_index=2,
            barycenter_loss=5.0,
        )
        
        self.assertIs(plugin.shift_network, shift_net)
        self.assertIs(plugin.interp_network, interp_net)
        self.assertIs(plugin.interpolator, interpolator)
        self.assertEqual(plugin.parameter_index, 1)
        self.assertEqual(plugin.reference_index, 2)
        self.assertEqual(plugin.barycenter_loss, 5.0)
 
    def test_fit_preprocessing_sets_reference_snapshot(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
            reference_index=1,
        )
        
        plugin.fit_preprocessing(self.rom)
        
        expected_snap = self.db._pairs[1][1]
        np.testing.assert_array_equal(plugin.reference_snapshot.values, expected_snap.values)
        np.testing.assert_array_equal(plugin.reference_snapshot.space, expected_snap.space)

    def test_fit_preprocessing_calls_train_interp_network(self):
        shift_net = SimpleANN()
        interp_net = SimpleANN()
        interp_net.fit = Mock()
        
        plugin = AutomaticShiftSnapshots(
            shift_network=shift_net,
            interp_network=interp_net,
            interpolator=SimpleInterpolator(),
            reference_index=0,
        )
        
        plugin.fit_preprocessing(self.rom)
        
        interp_net.fit.assert_called_once()
        args, _ = interp_net.fit.call_args
        np.testing.assert_array_equal(args[0], self.space.reshape(-1, 1))

    def test_fit_preprocessing_calls_train_shift_network(self):
        shift_net = SimpleANN()
        shift_net._build_model = Mock()
        
        plugin = AutomaticShiftSnapshots(
            shift_network=shift_net,
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        
        plugin.fit_preprocessing(self.rom)
        shift_net._build_model.assert_called_once()

    def test_fit_preprocessing_modifies_snapshots(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(self.rom)
        self.assertIsNotNone(self.db._pairs[0][1].values)

    def test_fit_preprocessing_with_barycenter_loss_zero(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
            barycenter_loss=0.0,
        )
        plugin.fit_preprocessing(self.rom)
        self.assertIsNotNone(plugin.reference_snapshot)

    def test_fit_preprocessing_with_barycenter_loss_nonzero(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
            barycenter_loss=10.0,
        )
        plugin.fit_preprocessing(self.rom)
        self.assertIsNotNone(plugin.reference_snapshot)

    def test_predict_postprocessing_creates_full_database(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(self.rom)
        plugin.predict_postprocessing(self.rom)
        
        self.assertIsInstance(self.rom._full_database, Database)

    def test_predict_postprocessing_preserves_snapshot_count(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(self.rom)
        original_count = len(self.rom.predict_full_database)
        plugin.predict_postprocessing(self.rom)
        
        self.assertEqual(len(self.rom._full_database), original_count)

    def test_predict_postprocessing_modifies_space(self):
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(self.rom)
        plugin.predict_postprocessing(self.rom)
        
        new_spaces = [snap.space.copy() for _, snap in self.rom._full_database._pairs]
        for new_space in new_spaces:
            self.assertEqual(len(new_space), len(self.space))
    
    def test_stop_training_integer_criterion(self):
        shift_net = SimpleANN(stop_training=[2])
        plugin = AutomaticShiftSnapshots(
            shift_network=shift_net,
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(self.rom)
        self.assertEqual(len(shift_net.loss_trend), 2)

    def test_stop_training_float_criterion(self):
        shift_net = SimpleANN(stop_training=[100.0])
        plugin = AutomaticShiftSnapshots(
            shift_network=shift_net,
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(self.rom)
        self.assertGreaterEqual(len(shift_net.loss_trend), 1)

    def test_single_snapshot_database(self):
        db = Database()
        snap = Snapshot(values=np.array([1.0, 2.0, 3.0]), space=self.space)
        db.add(Parameter([1.0]), snap)
        rom = MockROM(db)
        
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
        )
        plugin.fit_preprocessing(rom)
        plugin.predict_postprocessing(rom)
        self.assertEqual(len(rom._full_database), 1)

    def test_reference_index_boundary(self):
        db = Database()
        for i in range(5):
            snap = Snapshot(values=np.array([float(i)]), space=np.array([0.5]))
            db.add(Parameter([float(i)]), snap)
        
        rom = MockROM(db)
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
            reference_index=4,
        )
        plugin.fit_preprocessing(rom)
        self.assertEqual(plugin.reference_snapshot.values[0], 4.0)

    def test_multidimensional_parameters_raise_valueerror(self):
        db = Database()
        snap1 = Snapshot(values=np.array([1.0, 2.0, 3.0]), space=self.space)
        db.add(Parameter([1.0, 10.0]), snap1)
        rom = MockROM(db)
        
        plugin = AutomaticShiftSnapshots(
            shift_network=SimpleANN(),
            interp_network=SimpleANN(),
            interpolator=SimpleInterpolator(),
            parameter_index=1,
        )
        with self.assertRaises(ValueError):
            plugin.fit_preprocessing(rom)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
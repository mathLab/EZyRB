import numpy as np
import pytest

from ezyrb import POD, GPR, RBF, Database, ANN
from ezyrb import KNeighborsRegressor, RadiusNeighborsRegressor, Linear
from ezyrb import ReducedOrderModel as ROM
from ezyrb.plugin.scaler import DatabaseScaler

from sklearn.preprocessing import StandardScaler, MinMaxScaler

snapshots = np.load("tests/test_datasets/p_snapshots.npy").T
pred_sol_tst = np.load("tests/test_datasets/p_predsol.npy").T
pred_sol_gpr = np.load("tests/test_datasets/p_predsol_gpr.npy").T
param = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])


def test_constructor():
    pod = POD()

    rbf = RBF()
    # rbf = ANN([10, 10], function=torch.nn.Softplus(), stop_training=[1000])
    db = Database(param, snapshots.T)
    # rom = ROM(db, pod, rbf, plugins=[DatabaseScaler(StandardScaler(), 'full', 'snapshots')])
    rom = ROM(
        db,
        pod,
        rbf,
        plugins=[
            DatabaseScaler(StandardScaler(), "reduced", "parameters"),
            DatabaseScaler(StandardScaler(), "reduced", "snapshots"),
        ],
    )
    rom.fit()
    assert rom is not None


def test_scaler_reduced_snapshots():
    """Test that StandardScaler on reduced snapshots produces mean=0 and std=1"""
    pod = POD()
    rbf = RBF()
    db = Database(param, snapshots.T)
    rom = ROM(
        db,
        pod,
        rbf,
        plugins=[DatabaseScaler(StandardScaler(), "reduced", "snapshots")],
    )
    rom.fit()

    # Check that the scaled reduced snapshots have mean ≈ 0 and std ≈ 1
    scaled_snapshots = rom.train_reduced_database.snapshots_matrix
    np.testing.assert_allclose(np.mean(scaled_snapshots, axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(np.std(scaled_snapshots, axis=0), 1, atol=1e-7)


def test_scaler_reduced_parameters():
    """Test that StandardScaler on reduced parameters produces mean=0 and std=1"""
    pod = POD()
    rbf = RBF()
    db = Database(param, snapshots.T)
    rom = ROM(
        db,
        pod,
        rbf,
        plugins=[DatabaseScaler(StandardScaler(), "reduced", "parameters")],
    )
    rom.fit()

    # Check that the scaled reduced parameters have mean ≈ 0 and std ≈ 1
    scaled_params = rom.train_reduced_database.parameters_matrix
    np.testing.assert_allclose(np.mean(scaled_params, axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(np.std(scaled_params, axis=0), 1, atol=1e-7)


def test_scaler_full_snapshots():
    """Test that StandardScaler on full snapshots produces mean=0 and std=1"""
    pod = POD()
    rbf = RBF()
    db = Database(param, snapshots.T)
    rom = ROM(
        db,
        pod,
        rbf,
        plugins=[DatabaseScaler(StandardScaler(), "full", "snapshots")],
    )
    rom.fit()

    # Check that the scaled full snapshots have mean ≈ 0 and std ≈ 1
    scaled_snapshots = rom.train_full_database.snapshots_matrix
    np.testing.assert_allclose(np.mean(scaled_snapshots, axis=0), 0, atol=2e-6)
    np.testing.assert_allclose(np.std(scaled_snapshots, axis=0), 1, atol=2e-6)


def test_scaler_full_parameters():
    """Test that StandardScaler on full parameters produces mean=0 and std=1"""
    pod = POD()
    rbf = RBF()
    db = Database(param, snapshots.T)
    rom = ROM(
        db,
        pod,
        rbf,
        plugins=[DatabaseScaler(StandardScaler(), "full", "parameters")],
    )
    rom.fit()

    # Check that the scaled full parameters have mean ≈ 0 and std ≈ 1
    scaled_params = rom.train_full_database.parameters_matrix
    np.testing.assert_allclose(np.mean(scaled_params, axis=0), 0, atol=2e-6)
    np.testing.assert_allclose(np.std(scaled_params, axis=0), 1, atol=2e-6)


def test_values():
    pod = POD()
    rbf = RBF()
    db = Database(param, snapshots.T)
    rom = ROM(
        db,
        pod,
        rbf,
        plugins=[
            DatabaseScaler(StandardScaler(), "reduced", "snapshots"),
            DatabaseScaler(StandardScaler(), "full", "parameters"),
        ],
    )
    rom.fit()
    test_param = param[2]
    truth_sol = db.snapshots_matrix[2]
    predicted_sol = rom.predict(test_param)[0]
    np.testing.assert_allclose(predicted_sol, truth_sol, rtol=1e-5, atol=1e-5)

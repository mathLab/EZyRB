"""
Test SklearnApproximation and SklearnReduction wrappers
"""

import numpy as np
from unittest import TestCase
from ezyrb import SklearnApproximation, SklearnReduction


class TestSklearnWrappers(TestCase):
    def test_sklearn_approximation_import(self):
        """Test that SklearnApproximation can be imported"""
        from ezyrb import SklearnApproximation
        assert SklearnApproximation is not None

    def test_sklearn_reduction_import(self):
        """Test that SklearnReduction can be imported"""
        from ezyrb import SklearnReduction
        assert SklearnReduction is not None

    def test_sklearn_approximation_random_forest(self):
        """Test SklearnApproximation with RandomForestRegressor"""
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        approximation = SklearnApproximation(model)
        
        # Create dummy data
        X = np.random.rand(20, 3)
        y = np.random.rand(20, 2)
        
        approximation.fit(X, y)
        predictions = approximation.predict(X[:5])
        
        assert predictions.shape == (5, 2)

    def test_sklearn_approximation_svr(self):
        """Test SklearnApproximation with SVR"""
        from sklearn.svm import SVR
        from sklearn.multioutput import MultiOutputRegressor
        
        base_model = SVR(kernel='rbf')
        model = MultiOutputRegressor(base_model)
        approximation = SklearnApproximation(model)
        
        # Create dummy data
        X = np.random.rand(30, 2)
        y = np.random.rand(30, 3)
        
        approximation.fit(X, y)
        predictions = approximation.predict(X[:10])
        
        assert predictions.shape == (10, 3)

    def test_sklearn_reduction_pca(self):
        """Test SklearnReduction with PCA"""
        from sklearn.decomposition import PCA
        
        model = PCA(n_components=5)
        reduction = SklearnReduction(model)
        
        # Create dummy snapshots (10 snapshots, 50 features each)
        snapshots = np.random.rand(10, 50)
        
        reduction.fit(snapshots)
        reduced = reduction.transform(snapshots)
        
        assert reduced.shape == (5, 50)
        
        # Test inverse transform
        reconstructed = reduction.inverse_transform(reduced)
        assert reconstructed.shape == snapshots.shape

    def test_sklearn_reduction_kernel_pca(self):
        """Test SklearnReduction with KernelPCA"""
        from sklearn.decomposition import KernelPCA
        
        model = KernelPCA(n_components=3, kernel='rbf')
        reduction = SklearnReduction(model)
        
        # Create dummy snapshots
        snapshots = np.random.rand(8, 40)
        
        reduction.fit(snapshots)
        reduced = reduction.transform(snapshots)
        
        assert reduced.shape == (3, 40)

    def test_sklearn_approximation_invalid_model(self):
        """Test that invalid model raises ValueError"""
        class DummyModel:
            pass
        
        with self.assertRaises(ValueError):
            SklearnApproximation(DummyModel())

    def test_sklearn_reduction_invalid_model(self):
        """Test that invalid model raises ValueError"""
        class DummyModel:
            pass
        
        with self.assertRaises(ValueError):
            SklearnReduction(DummyModel())

    def test_sklearn_approximation_predict_before_fit(self):
        """Test that predict before fit raises RuntimeError"""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        approximation = SklearnApproximation(model)
        
        X = np.random.rand(10, 2)
        
        with self.assertRaises(RuntimeError):
            approximation.predict(X)

    def test_sklearn_reduction_transform_before_fit(self):
        """Test that transform before fit raises RuntimeError"""
        from sklearn.decomposition import PCA
        
        model = PCA(n_components=2)
        reduction = SklearnReduction(model)
        
        snapshots = np.random.rand(5, 20)
        
        with self.assertRaises(RuntimeError):
            reduction.transform(snapshots)

    def test_sklearn_reduction_reduce_expand_aliases(self):
        """Test that reduce/expand aliases work"""
        from sklearn.decomposition import PCA
        
        model = PCA(n_components=3)
        reduction = SklearnReduction(model)
        
        snapshots = np.random.rand(7, 25)
        
        reduction.fit(snapshots)
        
        # Test reduce (alias for transform)
        reduced = reduction.reduce(snapshots)
        assert reduced.shape == (3, 25)
        
        # Test expand (alias for inverse_transform)
        expanded = reduction.expand(reduced)
        assert expanded.shape == snapshots.shape

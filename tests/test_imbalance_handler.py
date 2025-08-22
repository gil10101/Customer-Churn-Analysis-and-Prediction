"""
Unit tests for ImbalanceHandler class.

Tests all imbalance handling techniques including class weighting,
SMOTE variants, ensemble methods, and evaluation functionality.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

from utils.imbalance_handler import ImbalanceHandler, ImbalanceStrategy, ImbalanceResults


class TestImbalanceHandler:
    """Test suite for ImbalanceHandler class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset for testing."""
        # Create imbalanced dataset (80-20 split)
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=1,
            weights=[0.8, 0.2],  # Imbalanced classes
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def handler(self):
        """Create ImbalanceHandler instance for testing."""
        return ImbalanceHandler(random_state=42)
    
    def test_initialization(self, handler):
        """Test ImbalanceHandler initialization."""
        assert handler.strategy == "adaptive"
        assert handler.random_state == 42
        assert len(handler.strategies) > 0
        assert 'smote_regular' in handler.strategies
        assert 'balanced_rf' in handler.strategies
        assert 'class_weight_balanced' in handler.strategies
    
    def test_strategy_initialization(self, handler):
        """Test that all strategies are properly initialized."""
        expected_strategies = [
            'class_weight_balanced', 'class_weight_custom',
            'smote_regular', 'smote_borderline', 'adasyn', 'smote_svm',
            'balanced_rf', 'easy_ensemble',
            'smote_tomek', 'smote_enn'
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in handler.strategies
            strategy = handler.strategies[strategy_name]
            assert isinstance(strategy, ImbalanceStrategy)
            assert strategy.name == strategy_name
            assert strategy.method in ['class_weight', 'oversample', 'ensemble', 'hybrid']
    
    def test_class_weighting_logistic_regression(self, handler, sample_data):
        """Test class weighting with LogisticRegression."""
        X, y = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Apply class weighting
        weighted_model = handler.apply_class_weighting(model, X, y)
        
        # Check that class_weight parameter is set
        assert weighted_model.class_weight is not None
        assert isinstance(weighted_model.class_weight, dict)
        assert len(weighted_model.class_weight) == 2  # Binary classification
    
    def test_class_weighting_random_forest(self, handler, sample_data):
        """Test class weighting with RandomForestClassifier."""
        X, y = sample_data
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        
        # Apply class weighting
        weighted_model = handler.apply_class_weighting(model, X, y)
        
        # Check that class_weight parameter is set
        assert weighted_model.class_weight is not None
        assert isinstance(weighted_model.class_weight, dict)
    
    def test_class_weighting_xgboost(self, handler, sample_data):
        """Test class weighting with XGBoost."""
        X, y = sample_data
        model = xgb.XGBClassifier(random_state=42, n_estimators=10)
        
        # Apply class weighting
        weighted_model = handler.apply_class_weighting(model, X, y)
        
        # Check that scale_pos_weight parameter is set
        assert weighted_model.scale_pos_weight is not None
        assert weighted_model.scale_pos_weight > 0
    
    def test_class_weighting_lightgbm(self, handler, sample_data):
        """Test class weighting with LightGBM."""
        X, y = sample_data
        model = lgb.LGBMClassifier(random_state=42, n_estimators=10, verbose=-1)
        
        # Apply class weighting
        weighted_model = handler.apply_class_weighting(model, X, y)
        
        # Check that class_weight parameter is set
        assert weighted_model.class_weight is not None
    
    def test_smote_regular(self, handler, sample_data):
        """Test regular SMOTE oversampling."""
        X, y = sample_data
        
        # Get original distribution
        original_counts = np.bincount(y)
        
        # Apply SMOTE
        X_resampled, y_resampled = handler.apply_smote_variants(X, y, variant="smote")
        
        # Check that minority class is oversampled
        new_counts = np.bincount(y_resampled)
        assert len(X_resampled) > len(X)
        assert len(y_resampled) > len(y)
        assert new_counts[1] > original_counts[1]  # Minority class increased
    
    def test_smote_borderline(self, handler, sample_data):
        """Test Borderline SMOTE oversampling."""
        X, y = sample_data
        
        # Apply Borderline SMOTE
        X_resampled, y_resampled = handler.apply_smote_variants(X, y, variant="borderline")
        
        # Check that data is resampled
        assert len(X_resampled) >= len(X)
        assert len(y_resampled) >= len(y)
        assert X_resampled.shape[1] == X.shape[1]  # Same number of features
    
    def test_adasyn(self, handler, sample_data):
        """Test ADASYN oversampling."""
        X, y = sample_data
        
        # Apply ADASYN
        X_resampled, y_resampled = handler.apply_smote_variants(X, y, variant="adasyn")
        
        # Check that data is resampled
        assert len(X_resampled) >= len(X)
        assert len(y_resampled) >= len(y)
        assert X_resampled.shape[1] == X.shape[1]
    
    def test_smote_svm(self, handler, sample_data):
        """Test SVM SMOTE oversampling."""
        X, y = sample_data
        
        # Apply SVM SMOTE
        X_resampled, y_resampled = handler.apply_smote_variants(X, y, variant="svm")
        
        # Check that data is resampled
        assert len(X_resampled) >= len(X)
        assert len(y_resampled) >= len(y)
        assert X_resampled.shape[1] == X.shape[1]
    
    def test_invalid_smote_variant(self, handler, sample_data):
        """Test error handling for invalid SMOTE variant."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown SMOTE variant"):
            handler.apply_smote_variants(X, y, variant="invalid_variant")
    
    def test_balanced_ensemble_default(self, handler):
        """Test creation of default balanced ensemble methods."""
        ensemble_methods = handler.create_balanced_ensemble()
        
        assert isinstance(ensemble_methods, list)
        assert len(ensemble_methods) == 2  # BalancedRF and EasyEnsemble
        
        # Check types
        from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
        assert any(isinstance(model, BalancedRandomForestClassifier) for model in ensemble_methods)
        assert any(isinstance(model, EasyEnsembleClassifier) for model in ensemble_methods)
    
    def test_balanced_ensemble_custom(self, handler):
        """Test creation of custom balanced ensemble."""
        base_models = [
            LogisticRegression(random_state=42),
            DecisionTreeClassifier(random_state=42)
        ]
        
        ensemble_methods = handler.create_balanced_ensemble(base_models)
        
        assert isinstance(ensemble_methods, list)
        assert len(ensemble_methods) == len(base_models)
        
        # Check that all are BaggingClassifier instances
        from sklearn.ensemble import BaggingClassifier
        for model in ensemble_methods:
            assert isinstance(model, BaggingClassifier)
    
    def test_evaluate_imbalance_strategies(self, handler, sample_data):
        """Test evaluation of imbalance handling strategies."""
        X, y = sample_data
        
        # Use small subset for faster testing
        X_small, _, y_small, _ = train_test_split(X, y, test_size=0.8, random_state=42)
        
        # Test with default models
        results_df = handler.evaluate_imbalance_strategies(X_small, y_small, cv_folds=3)
        
        # Check results structure
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        
        # Check required columns
        required_columns = ['strategy', 'model', 'method', 'roc_auc_mean', 'f1_mean']
        for col in required_columns:
            assert col in results_df.columns
        
        # Check that all strategies are represented
        strategies_in_results = set(results_df['strategy'].unique())
        assert len(strategies_in_results) > 0
    
    def test_evaluate_with_custom_models(self, handler, sample_data):
        """Test evaluation with custom models."""
        X, y = sample_data
        
        # Use small subset for faster testing
        X_small, _, y_small, _ = train_test_split(X, y, test_size=0.9, random_state=42)
        
        custom_models = [LogisticRegression(random_state=42, max_iter=1000)]
        
        results_df = handler.evaluate_imbalance_strategies(
            X_small, y_small, models=custom_models, cv_folds=2
        )
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert all(results_df['model'] == 'LogisticRegression')
    
    def test_get_optimal_strategy(self, handler, sample_data):
        """Test finding optimal strategy."""
        X, y = sample_data
        
        # Use small subset for faster testing
        X_small, _, y_small, _ = train_test_split(X, y, test_size=0.9, random_state=42)
        
        optimal_strategy = handler.get_optimal_strategy(X_small, y_small, cv_folds=2)
        
        assert isinstance(optimal_strategy, str)
        assert optimal_strategy in handler.strategies
    
    def test_get_optimal_strategy_f1(self, handler, sample_data):
        """Test finding optimal strategy using F1 score."""
        X, y = sample_data
        
        # Use small subset for faster testing
        X_small, _, y_small, _ = train_test_split(X, y, test_size=0.9, random_state=42)
        
        optimal_strategy = handler.get_optimal_strategy(X_small, y_small, metric='f1', cv_folds=2)
        
        assert isinstance(optimal_strategy, str)
        assert optimal_strategy in handler.strategies
    
    def test_apply_strategy_class_weight(self, handler, sample_data):
        """Test applying class weighting strategy."""
        X, y = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        result = handler.apply_strategy('class_weight_balanced', X, y, model=model)
        
        assert hasattr(result, 'class_weight')
        assert result.class_weight is not None
    
    def test_apply_strategy_oversample(self, handler, sample_data):
        """Test applying oversampling strategy."""
        X, y = sample_data
        
        X_resampled, y_resampled = handler.apply_strategy('smote_regular', X, y)
        
        assert len(X_resampled) >= len(X)
        assert len(y_resampled) >= len(y)
        assert X_resampled.shape[1] == X.shape[1]
    
    def test_apply_strategy_ensemble(self, handler, sample_data):
        """Test applying ensemble strategy."""
        X, y = sample_data
        
        ensemble_model = handler.apply_strategy('balanced_rf', X, y)
        
        from imblearn.ensemble import BalancedRandomForestClassifier
        assert isinstance(ensemble_model, BalancedRandomForestClassifier)
    
    def test_apply_strategy_hybrid(self, handler, sample_data):
        """Test applying hybrid strategy."""
        X, y = sample_data
        
        X_resampled, y_resampled = handler.apply_strategy('smote_tomek', X, y)
        
        assert len(X_resampled) != len(X)  # Should be different due to resampling
        assert len(y_resampled) != len(y)
        assert X_resampled.shape[1] == X.shape[1]
    
    def test_apply_strategy_invalid(self, handler, sample_data):
        """Test error handling for invalid strategy."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            handler.apply_strategy('invalid_strategy', X, y)
    
    def test_apply_strategy_class_weight_no_model(self, handler, sample_data):
        """Test error when applying class weight strategy without model."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Model required"):
            handler.apply_strategy('class_weight_balanced', X, y)
    
    def test_get_strategy_summary(self, handler):
        """Test getting strategy summary."""
        summary_df = handler.get_strategy_summary()
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == len(handler.strategies)
        
        required_columns = ['strategy_name', 'method', 'description']
        for col in required_columns:
            assert col in summary_df.columns
        
        # Check that all strategies are included
        strategy_names = set(summary_df['strategy_name'])
        expected_names = set(handler.strategies.keys())
        assert strategy_names == expected_names
    
    def test_get_class_distribution(self, handler, sample_data):
        """Test getting class distribution statistics."""
        X, y = sample_data
        
        distribution = handler.get_class_distribution(y)
        
        assert isinstance(distribution, dict)
        assert 'class_0' in distribution
        assert 'class_1' in distribution
        assert 'imbalance_ratio' in distribution
        
        # Check structure of class information
        for class_key in ['class_0', 'class_1']:
            assert 'count' in distribution[class_key]
            assert 'percentage' in distribution[class_key]
            assert isinstance(distribution[class_key]['count'], int)
            assert isinstance(distribution[class_key]['percentage'], float)
        
        # Check imbalance ratio
        assert isinstance(distribution['imbalance_ratio'], float)
        assert distribution['imbalance_ratio'] > 1.0  # Should be imbalanced
    
    def test_get_class_distribution_multiclass(self, handler):
        """Test class distribution with multiclass data."""
        # Create multiclass dataset
        y_multi = np.array([0, 0, 1, 1, 2, 2, 2])
        
        distribution = handler.get_class_distribution(y_multi)
        
        assert isinstance(distribution, dict)
        assert 'class_0' in distribution
        assert 'class_1' in distribution
        assert 'class_2' in distribution
        
        # Should not have imbalance_ratio for multiclass
        assert 'imbalance_ratio' not in distribution
    
    def test_clone_model(self, handler):
        """Test model cloning functionality."""
        original_model = LogisticRegression(random_state=42, C=0.5)
        cloned_model = handler._clone_model(original_model)
        
        # Should be different objects
        assert cloned_model is not original_model
        
        # Should have same parameters
        assert cloned_model.random_state == original_model.random_state
        assert cloned_model.C == original_model.C
    
    def test_strategy_methods_coverage(self, handler):
        """Test that all strategy methods are covered."""
        expected_methods = ['class_weight', 'oversample', 'ensemble', 'hybrid']
        
        actual_methods = set()
        for strategy in handler.strategies.values():
            actual_methods.add(strategy.method)
        
        for method in expected_methods:
            assert method in actual_methods, f"Method {method} not found in strategies"
    
    def test_reproducibility(self, handler, sample_data):
        """Test that results are reproducible with same random state."""
        X, y = sample_data
        
        # Apply SMOTE twice with same handler
        X_resampled1, y_resampled1 = handler.apply_smote_variants(X, y, variant="smote")
        
        # Create new handler with same random state
        handler2 = ImbalanceHandler(random_state=42)
        X_resampled2, y_resampled2 = handler2.apply_smote_variants(X, y, variant="smote")
        
        # Results should be identical
        np.testing.assert_array_equal(X_resampled1, X_resampled2)
        np.testing.assert_array_equal(y_resampled1, y_resampled2)


if __name__ == "__main__":
    pytest.main([__file__])
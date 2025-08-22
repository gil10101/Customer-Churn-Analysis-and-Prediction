"""
Tests for Model Training Pipeline with Imbalance Handling.

This module provides comprehensive tests for the integrated model training pipeline
that combines imbalance handling strategies with hyperparameter optimization and
cross-validation for imbalanced churn prediction datasets.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

# Import the modules to test
from utils.model_training_pipeline import (
    ModelTrainingPipeline, TrainingConfig, TrainingResult,
    train_imbalanced_model
)
from utils.imbalance_handler import ImbalanceHandler
from utils.model_evaluation import ModelEvaluator, BusinessMetrics


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.cv_folds == 5
        assert config.random_state == 42
        assert config.n_jobs == -1
        assert config.optimization_trials == 100
        assert config.optimization_metric == 'roc_auc'
        assert config.compare_strategies is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            cv_folds=3,
            optimization_trials=50,
            optimization_metric='f1'
        )
        
        assert config.cv_folds == 3
        assert config.optimization_trials == 50
        assert config.optimization_metric == 'f1'


class TestTrainingResult:
    """Test TrainingResult dataclass."""
    
    def test_training_result_creation(self):
        """Test TrainingResult creation."""
        model = RandomForestClassifier()
        result = TrainingResult(
            model_name='test_model',
            strategy_name='test_strategy',
            best_model=model,
            best_params={'n_estimators': 100},
            cv_scores={'roc_auc_mean': 0.85},
            training_time=10.5
        )
        
        assert result.model_name == 'test_model'
        assert result.strategy_name == 'test_strategy'
        assert result.best_model == model
        assert result.best_params == {'n_estimators': 100}
        assert result.cv_scores == {'roc_auc_mean': 0.85}
        assert result.training_time == 10.5


class TestModelTrainingPipeline:
    """Test ModelTrainingPipeline class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset."""
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
    def pipeline(self):
        """Create ModelTrainingPipeline instance."""
        config = TrainingConfig(
            cv_folds=3,  # Faster for testing
            optimization_trials=10,  # Fewer trials for testing
            optimization_timeout=60  # Shorter timeout for testing
        )
        return ModelTrainingPipeline(config=config)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config.cv_folds == 3
        assert pipeline.config.optimization_trials == 10
        assert isinstance(pipeline.imbalance_handler, ImbalanceHandler)
        assert isinstance(pipeline.evaluator, ModelEvaluator)
        assert len(pipeline.base_models) > 0
        assert 'logistic_regression' in pipeline.base_models
        assert 'random_forest' in pipeline.base_models
    
    def test_base_models_initialization(self, pipeline):
        """Test base models are properly initialized."""
        models = pipeline.base_models
        
        assert 'logistic_regression' in models
        assert 'random_forest' in models
        assert 'gradient_boosting' in models
        assert 'svm' in models
        
        # Check that models have correct random state
        assert models['logistic_regression'].random_state == pipeline.config.random_state
        assert models['random_forest'].random_state == pipeline.config.random_state
    
    def test_compare_imbalance_strategies(self, pipeline, sample_data):
        """Test imbalance strategy comparison."""
        X, y = sample_data
        
        # Test with limited strategies and models for speed
        strategies = ['class_weight_balanced', 'smote_regular']
        models = ['logistic_regression', 'random_forest']
        
        results = pipeline.compare_imbalance_strategies(
            X, y, models=models, strategies=strategies
        )
        
        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        assert 'strategy' in results.columns
        assert 'model' in results.columns
        assert 'roc_auc_mean' in results.columns
        assert 'f1_mean' in results.columns
        
        # Check that results are sorted by ROC-AUC
        roc_scores = results['roc_auc_mean'].values
        assert all(roc_scores[i] >= roc_scores[i+1] for i in range(len(roc_scores)-1))
    
    def test_get_optimal_strategy(self, pipeline, sample_data):
        """Test optimal strategy selection."""
        X, y = sample_data
        
        optimal_strategy, optimal_model = pipeline.get_optimal_strategy(X, y)
        
        assert isinstance(optimal_strategy, str)
        assert isinstance(optimal_model, str)
        assert optimal_strategy in pipeline.imbalance_handler.strategies
        assert optimal_model in pipeline.base_models
    
    def test_suggest_hyperparameters(self, pipeline):
        """Test hyperparameter suggestion for different models."""
        # Mock trial object
        trial = Mock()
        trial.suggest_float.return_value = 1.0
        trial.suggest_int.return_value = 100
        trial.suggest_categorical.return_value = 'l2'
        trial.params = {}
        
        # Test different model types
        models = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
        
        for model_name in models:
            params = pipeline._suggest_hyperparameters(trial, model_name)
            assert isinstance(params, dict)
            if model_name != 'unknown_model':
                assert len(params) > 0
    
    @pytest.mark.slow
    def test_optimize_hyperparameters_optuna(self, pipeline, sample_data):
        """Test hyperparameter optimization with Optuna."""
        X, y = sample_data
        
        # Use a small subset for faster testing
        X_small = X[:200]
        y_small = y[:200]
        
        # Test with limited trials
        pipeline.config.optimization_trials = 5
        pipeline.config.optimization_timeout = 30
        
        best_model, best_params, study = pipeline.optimize_hyperparameters(
            X_small, y_small, 'logistic_regression', 'class_weight_balanced', 'optuna'
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert study is not None
        assert hasattr(study, 'best_value')
    
    def test_optimize_hyperparameters_grid(self, pipeline, sample_data):
        """Test hyperparameter optimization with GridSearchCV."""
        X, y = sample_data
        
        # Use a small subset for faster testing
        X_small = X[:200]
        y_small = y[:200]
        
        best_model, best_params, grid_search = pipeline.optimize_hyperparameters(
            X_small, y_small, 'logistic_regression', 'class_weight_balanced', 'grid'
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert grid_search is not None
        assert hasattr(grid_search, 'best_score_')
    
    def test_train_with_cross_validation(self, pipeline, sample_data):
        """Test model training with cross-validation."""
        X, y = sample_data
        
        # Use a small subset for faster testing
        X_small = X[:300]
        y_small = y[:300]
        
        # Test without hyperparameter optimization for speed
        result = pipeline.train_with_cross_validation(
            X_small, y_small,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        assert isinstance(result, TrainingResult)
        assert result.model_name == 'logistic_regression'
        assert result.strategy_name == 'class_weight_balanced'
        assert result.best_model is not None
        assert 'roc_auc_mean' in result.cv_scores
        assert 'f1_mean' in result.cv_scores
        assert result.training_time > 0
    
    def test_train_with_auto_selection(self, pipeline, sample_data):
        """Test model training with automatic strategy/model selection."""
        X, y = sample_data
        
        # Use a small subset for faster testing
        X_small = X[:200]
        y_small = y[:200]
        
        result = pipeline.train_with_cross_validation(
            X_small, y_small,
            optimize_hyperparameters=False
        )
        
        assert isinstance(result, TrainingResult)
        assert result.model_name is not None
        assert result.strategy_name is not None
        assert result.strategy_comparison_results is not None
        assert not result.strategy_comparison_results.empty
    
    def test_evaluate_trained_model(self, pipeline, sample_data):
        """Test evaluation of trained model."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train a model
        result = pipeline.train_with_cross_validation(
            X_train, y_train,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        # Evaluate the model
        performance = pipeline.evaluate_trained_model(result, X_test, y_test)
        
        assert performance is not None
        assert performance.auc_roc > 0
        assert performance.f1_score > 0
        assert result.performance_metrics == performance
    
    def test_get_training_summary(self, pipeline, sample_data):
        """Test training summary generation."""
        X, y = sample_data
        X_small = X[:200]
        y_small = y[:200]
        
        # Train multiple models
        result1 = pipeline.train_with_cross_validation(
            X_small, y_small,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        result2 = pipeline.train_with_cross_validation(
            X_small, y_small,
            model_name='random_forest',
            strategy_name='smote_regular',
            optimize_hyperparameters=False
        )
        
        summary = pipeline.get_training_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'model_name' in summary.columns
        assert 'strategy_name' in summary.columns
        assert 'roc_auc_mean' in summary.columns
        
        # Check that summary is sorted by ROC-AUC
        roc_scores = summary['roc_auc_mean'].values
        assert roc_scores[0] >= roc_scores[1]
    
    def test_save_training_results(self, pipeline, sample_data):
        """Test saving training results to disk."""
        X, y = sample_data
        X_small = X[:200]
        y_small = y[:200]
        
        # Train a model
        result = pipeline.train_with_cross_validation(
            X_small, y_small,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        # Save results
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            pipeline.save_training_results(output_path)
            
            # Check that files were created
            assert (output_path / 'training_summary.csv').exists()
            assert (output_path / 'training_result_0.json').exists()
    
    def test_invalid_model_name(self, pipeline, sample_data):
        """Test handling of invalid model name."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown model"):
            pipeline.optimize_hyperparameters(
                X, y, 'invalid_model', 'class_weight_balanced'
            )
    
    def test_invalid_strategy_name(self, pipeline, sample_data):
        """Test handling of invalid strategy name."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            pipeline.optimize_hyperparameters(
                X, y, 'logistic_regression', 'invalid_strategy'
            )
    
    def test_invalid_optimization_method(self, pipeline, sample_data):
        """Test handling of invalid optimization method."""
        X, y = sample_data
        
        with pytest.raises(ValueError, match="Unknown optimization method"):
            pipeline.optimize_hyperparameters(
                X, y, 'logistic_regression', 'class_weight_balanced', 'invalid_method'
            )


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample imbalanced dataset."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            weights=[0.7, 0.3],
            random_state=42
        )
        return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    def test_train_imbalanced_model_basic(self, sample_data):
        """Test basic usage of train_imbalanced_model function."""
        X_train, X_test, y_train, y_test = sample_data
        
        training_result, performance = train_imbalanced_model(
            X_train, y_train, X_test, y_test,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        assert isinstance(training_result, TrainingResult)
        assert training_result.model_name == 'logistic_regression'
        assert training_result.strategy_name == 'class_weight_balanced'
        assert performance is not None
        assert performance.auc_roc > 0
    
    def test_train_imbalanced_model_auto_selection(self, sample_data):
        """Test train_imbalanced_model with auto-selection."""
        X_train, X_test, y_train, y_test = sample_data
        
        training_result, performance = train_imbalanced_model(
            X_train, y_train, X_test, y_test,
            optimize_hyperparameters=False
        )
        
        assert isinstance(training_result, TrainingResult)
        assert training_result.model_name is not None
        assert training_result.strategy_name is not None
        assert performance is not None
    
    def test_train_imbalanced_model_no_test_data(self, sample_data):
        """Test train_imbalanced_model without test data."""
        X_train, _, y_train, _ = sample_data
        
        training_result, performance = train_imbalanced_model(
            X_train, y_train,
            model_name='random_forest',
            strategy_name='smote_regular',
            optimize_hyperparameters=False
        )
        
        assert isinstance(training_result, TrainingResult)
        assert performance is None  # No test data provided
    
    def test_train_imbalanced_model_with_business_metrics(self, sample_data):
        """Test train_imbalanced_model with business metrics."""
        X_train, X_test, y_train, y_test = sample_data
        
        business_metrics = BusinessMetrics(
            customer_acquisition_cost=150.0,
            customer_retention_cost=30.0,
            average_customer_value=600.0
        )
        
        training_result, performance = train_imbalanced_model(
            X_train, y_train, X_test, y_test,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False,
            business_metrics=business_metrics
        )
        
        assert isinstance(training_result, TrainingResult)
        assert performance is not None
        assert performance.business_value != 0  # Business metrics calculated
        assert performance.roi_percentage != 0


class TestIntegrationWithExistingComponents:
    """Test integration with existing ImbalanceHandler and ModelEvaluator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            weights=[0.8, 0.2],
            random_state=42
        )
        return X, y
    
    def test_integration_with_imbalance_handler(self, sample_data):
        """Test integration with ImbalanceHandler."""
        X, y = sample_data
        
        # Create custom imbalance handler
        imbalance_handler = ImbalanceHandler(strategy='adaptive', random_state=42)
        
        # Create pipeline with custom handler
        pipeline = ModelTrainingPipeline(imbalance_handler=imbalance_handler)
        
        # Test that the handler is used
        assert pipeline.imbalance_handler == imbalance_handler
        
        # Test strategy comparison
        results = pipeline.compare_imbalance_strategies(
            X, y, models=['logistic_regression'], strategies=['class_weight_balanced']
        )
        
        assert not results.empty
    
    def test_integration_with_model_evaluator(self, sample_data):
        """Test integration with ModelEvaluator."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create custom evaluator
        business_metrics = BusinessMetrics(average_customer_value=500.0)
        evaluator = ModelEvaluator(business_metrics=business_metrics)
        
        # Create pipeline with custom evaluator
        pipeline = ModelTrainingPipeline(evaluator=evaluator)
        
        # Train and evaluate
        result = pipeline.train_with_cross_validation(
            X_train, y_train,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        performance = pipeline.evaluate_trained_model(result, X_test, y_test)
        
        assert performance is not None
        assert performance.business_value != 0  # Business metrics calculated
    
    def test_pandas_dataframe_input(self, sample_data):
        """Test handling of pandas DataFrame input."""
        X, y = sample_data
        
        # Convert to DataFrame and Series
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
        y_series = pd.Series(y, name='target')
        
        pipeline = ModelTrainingPipeline()
        
        result = pipeline.train_with_cross_validation(
            X_df, y_series,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        assert isinstance(result, TrainingResult)
        assert result.cv_scores['roc_auc_mean'] > 0


# Performance and edge case tests
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        
        pipeline = ModelTrainingPipeline()
        
        with pytest.raises((ValueError, IndexError)):
            pipeline.compare_imbalance_strategies(X, y)
    
    def test_single_class_dataset(self):
        """Test handling of single-class dataset."""
        X = np.random.randn(100, 5)
        y = np.ones(100)  # All same class
        
        pipeline = ModelTrainingPipeline()
        
        # This should handle the error gracefully
        results = pipeline.compare_imbalance_strategies(X, y)
        # Results might be empty due to stratification issues
        assert isinstance(results, pd.DataFrame)
    
    def test_very_small_dataset(self):
        """Test handling of very small dataset."""
        X = np.random.randn(10, 3)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        config = TrainingConfig(cv_folds=2)  # Reduce folds for small dataset
        pipeline = ModelTrainingPipeline(config=config)
        
        result = pipeline.train_with_cross_validation(
            X, y,
            model_name='logistic_regression',
            strategy_name='class_weight_balanced',
            optimize_hyperparameters=False
        )
        
        assert isinstance(result, TrainingResult)


if __name__ == '__main__':
    pytest.main([__file__])
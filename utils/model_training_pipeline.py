"""
Model Training Pipeline with Imbalance Handling.

This module provides a comprehensive model training pipeline that integrates
imbalance handling strategies with hyperparameter optimization and cross-validation
specifically designed for imbalanced churn prediction datasets.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import time
import json
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    classification_report, make_scorer
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import classification_report_imbalanced
import optuna
from optuna.samplers import TPESampler
import warnings

from .imbalance_handler import ImbalanceHandler, ImbalanceStrategy
from .model_evaluation import ModelEvaluator, ModelPerformance, BusinessMetrics
from .config import get_model_config, get_statistical_config
from .logging_setup import get_notebook_logger

logger = get_notebook_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training pipeline."""
    cv_folds: int = 5
    random_state: int = 42
    n_jobs: int = -1
    
    # Hyperparameter optimization
    optimization_trials: int = 100
    optimization_timeout: int = 3600  # 1 hour
    optimization_metric: str = 'roc_auc'
    
    # Strategy comparison
    compare_strategies: bool = True
    strategy_comparison_cv: int = 3  # Faster CV for strategy comparison
    
    # Model selection
    include_ensemble_methods: bool = True
    max_models_per_strategy: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    min_improvement: float = 0.001


@dataclass
class TrainingResult:
    """Results from model training pipeline."""
    model_name: str
    strategy_name: str
    best_model: BaseEstimator
    best_params: Dict[str, Any]
    cv_scores: Dict[str, float]
    training_time: float
    strategy_comparison_results: Optional[pd.DataFrame] = None
    hyperparameter_study: Optional[Any] = None  # Optuna study object
    performance_metrics: Optional[ModelPerformance] = None


class ModelTrainingPipeline:
    """
    Comprehensive model training pipeline with integrated imbalance handling.
    
    This pipeline implements:
    - Integration of ImbalanceHandler with model training workflow
    - Strategy comparison and automatic selection methods
    - Hyperparameter optimization for imbalanced datasets
    - Cross-validation with stratified sampling for imbalanced data
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        imbalance_handler: Optional[ImbalanceHandler] = None,
        evaluator: Optional[ModelEvaluator] = None
    ):
        """
        Initialize the model training pipeline.
        
        Args:
            config: Training configuration
            imbalance_handler: Imbalance handling system
            evaluator: Model evaluation system
        """
        self.config = config or TrainingConfig()
        self.imbalance_handler = imbalance_handler or ImbalanceHandler(random_state=self.config.random_state)
        self.evaluator = evaluator or ModelEvaluator(random_state=self.config.random_state)
        
        # Initialize base models
        self.base_models = self._initialize_base_models()
        
        # Store training results
        self.training_results: List[TrainingResult] = []
        
        logger.info(f"ModelTrainingPipeline initialized with {len(self.base_models)} base models")
    
    def _initialize_base_models(self) -> Dict[str, BaseEstimator]:
        """Initialize base models for training."""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                n_jobs=self.config.n_jobs
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=self.config.random_state
            )
        }
        
        # Add SVM if not too many samples (SVM can be slow on large datasets)
        models['svm'] = SVC(
            random_state=self.config.random_state,
            probability=True  # Enable probability predictions
        )
        
        return models
    
    def compare_imbalance_strategies(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare different imbalance handling strategies across multiple models.
        
        Args:
            X: Feature matrix
            y: Target vector
            models: List of model names to evaluate (None for all)
            strategies: List of strategy names to evaluate (None for all)
            
        Returns:
            DataFrame with strategy comparison results
        """
        logger.info("Starting imbalance strategy comparison")
        
        if models is None:
            models = list(self.base_models.keys())
        
        if strategies is None:
            strategies = list(self.imbalance_handler.strategies.keys())
        
        results = []
        cv = StratifiedKFold(
            n_splits=self.config.strategy_comparison_cv,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Evaluate each strategy-model combination
        for strategy_name in strategies:
            strategy = self.imbalance_handler.strategies[strategy_name]
            logger.info(f"Evaluating strategy: {strategy_name}")
            
            for model_name in models:
                if model_name not in self.base_models:
                    logger.warning(f"Model {model_name} not found, skipping")
                    continue
                
                try:
                    start_time = time.time()
                    model = clone(self.base_models[model_name])
                    
                    if strategy.method == 'class_weight':
                        # Apply class weighting to model
                        weighted_model = self.imbalance_handler.apply_class_weighting(model, X, y)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(
                            weighted_model, X, y, cv=cv,
                            scoring='roc_auc', n_jobs=self.config.n_jobs
                        )
                        f1_scores = cross_val_score(
                            weighted_model, X, y, cv=cv,
                            scoring='f1', n_jobs=self.config.n_jobs
                        )
                        
                    elif strategy.method in ['oversample', 'hybrid']:
                        # Create pipeline with resampling
                        pipeline = ImbPipeline([
                            ('sampler', strategy.parameters['sampler']),
                            ('classifier', model)
                        ])
                        
                        # Cross-validation
                        cv_scores = cross_val_score(
                            pipeline, X, y, cv=cv,
                            scoring='roc_auc', n_jobs=self.config.n_jobs
                        )
                        f1_scores = cross_val_score(
                            pipeline, X, y, cv=cv,
                            scoring='f1', n_jobs=self.config.n_jobs
                        )
                        
                    elif strategy.method == 'ensemble':
                        # Use ensemble method directly
                        ensemble_model = clone(strategy.parameters['classifier'])
                        
                        # Cross-validation
                        cv_scores = cross_val_score(
                            ensemble_model, X, y, cv=cv,
                            scoring='roc_auc', n_jobs=self.config.n_jobs
                        )
                        f1_scores = cross_val_score(
                            ensemble_model, X, y, cv=cv,
                            scoring='f1', n_jobs=self.config.n_jobs
                        )
                    
                    training_time = time.time() - start_time
                    
                    # Store results
                    results.append({
                        'strategy': strategy_name,
                        'model': model_name,
                        'method': strategy.method,
                        'roc_auc_mean': cv_scores.mean(),
                        'roc_auc_std': cv_scores.std(),
                        'f1_mean': f1_scores.mean(),
                        'f1_std': f1_scores.std(),
                        'training_time': training_time,
                        'description': strategy.description
                    })
                    
                    logger.info(
                        f"Strategy {strategy_name} + {model_name}: "
                        f"ROC-AUC = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error evaluating {strategy_name} + {model_name}: {str(e)}")
                    continue
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Sort by primary metric
            results_df = results_df.sort_values('roc_auc_mean', ascending=False)
            logger.info(
                f"Strategy comparison completed. Best combination: "
                f"{results_df.iloc[0]['strategy']} + {results_df.iloc[0]['model']} "
                f"(ROC-AUC: {results_df.iloc[0]['roc_auc_mean']:.3f})"
            )
        
        return results_df
    
    def get_optimal_strategy(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metric: str = 'roc_auc'
    ) -> Tuple[str, str]:
        """
        Get the optimal imbalance handling strategy and model combination.
        
        Args:
            X: Feature matrix
            y: Target vector
            metric: Evaluation metric for optimization
            
        Returns:
            Tuple of (optimal_strategy, optimal_model)
        """
        logger.info(f"Finding optimal strategy based on {metric}")
        
        # Compare all strategies
        comparison_results = self.compare_imbalance_strategies(X, y)
        
        if comparison_results.empty:
            logger.warning("No valid strategy comparison results, using defaults")
            return 'class_weight_balanced', 'random_forest'
        
        # Find best combination
        metric_col = f'{metric}_mean'
        if metric_col not in comparison_results.columns:
            logger.warning(f"Metric {metric} not found, using roc_auc")
            metric_col = 'roc_auc_mean'
        
        best_idx = comparison_results[metric_col].idxmax()
        best_strategy = comparison_results.loc[best_idx, 'strategy']
        best_model = comparison_results.loc[best_idx, 'model']
        best_score = comparison_results.loc[best_idx, metric_col]
        
        logger.info(
            f"Optimal combination: {best_strategy} + {best_model} "
            f"with {metric} = {best_score:.4f}"
        )
        
        return best_strategy, best_model
    
    def optimize_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: str,
        strategy_name: str,
        optimization_method: str = 'optuna'
    ) -> Tuple[BaseEstimator, Dict[str, Any], Any]:
        """
        Optimize hyperparameters for a specific model-strategy combination.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of the model to optimize
            strategy_name: Name of the imbalance strategy to use
            optimization_method: Method for optimization ('optuna', 'grid', 'random')
            
        Returns:
            Tuple of (best_model, best_params, study_object)
        """
        logger.info(f"Optimizing hyperparameters for {model_name} with {strategy_name}")
        
        if model_name not in self.base_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        if strategy_name not in self.imbalance_handler.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        base_model = clone(self.base_models[model_name])
        strategy = self.imbalance_handler.strategies[strategy_name]
        
        if optimization_method == 'optuna':
            return self._optimize_with_optuna(X, y, base_model, strategy, model_name)
        elif optimization_method == 'grid':
            return self._optimize_with_grid_search(X, y, base_model, strategy, model_name)
        elif optimization_method == 'random':
            return self._optimize_with_random_search(X, y, base_model, strategy, model_name)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def _optimize_with_optuna(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: BaseEstimator,
        strategy: ImbalanceStrategy,
        model_name: str
    ) -> Tuple[BaseEstimator, Dict[str, Any], Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters based on model type
            params = self._suggest_hyperparameters(trial, model_name)
            
            # Create model with suggested parameters
            model = clone(base_model)
            model.set_params(**params)
            
            # Apply imbalance handling strategy
            if strategy.method == 'class_weight':
                model = self.imbalance_handler.apply_class_weighting(model, X, y)
                pipeline = model
            elif strategy.method in ['oversample', 'hybrid']:
                pipeline = ImbPipeline([
                    ('sampler', strategy.parameters['sampler']),
                    ('classifier', model)
                ])
            elif strategy.method == 'ensemble':
                # For ensemble methods, optimize the ensemble parameters
                pipeline = clone(strategy.parameters['classifier'])
                ensemble_params = {k.replace('classifier__', ''): v for k, v in params.items() 
                                 if k.startswith('classifier__')}
                if ensemble_params:
                    pipeline.set_params(**ensemble_params)
            
            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            scores = cross_val_score(
                pipeline, X, y, cv=cv,
                scoring=self.config.optimization_metric,
                n_jobs=self.config.n_jobs
            )
            
            return scores.mean()
        
        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.random_state)
        )
        
        study.optimize(
            objective,
            n_trials=self.config.optimization_trials,
            timeout=self.config.optimization_timeout,
            show_progress_bar=True
        )
        
        # Get best parameters and create best model
        best_params = study.best_params
        best_model = clone(base_model)
        best_model.set_params(**best_params)
        
        # Apply best strategy
        if strategy.method == 'class_weight':
            best_model = self.imbalance_handler.apply_class_weighting(best_model, X, y)
        elif strategy.method in ['oversample', 'hybrid']:
            best_model = ImbPipeline([
                ('sampler', strategy.parameters['sampler']),
                ('classifier', best_model)
            ])
        elif strategy.method == 'ensemble':
            best_model = clone(strategy.parameters['classifier'])
            ensemble_params = {k.replace('classifier__', ''): v for k, v in best_params.items() 
                             if k.startswith('classifier__')}
            if ensemble_params:
                best_model.set_params(**ensemble_params)
        
        logger.info(
            f"Optuna optimization completed. Best score: {study.best_value:.4f}, "
            f"Best params: {best_params}"
        )
        
        return best_model, best_params, study
    
    def _suggest_hyperparameters(self, trial, model_name: str) -> Dict[str, Any]:
        """Suggest hyperparameters for different model types."""
        
        if model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'l1_ratio': trial.suggest_float('l1_ratio', 0, 1) if trial.params.get('penalty') == 'elasticnet' else None
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
        
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
            }
        
        else:
            return {}
    
    def _optimize_with_grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: BaseEstimator,
        strategy: ImbalanceStrategy,
        model_name: str
    ) -> Tuple[BaseEstimator, Dict[str, Any], Any]:
        """Optimize hyperparameters using GridSearchCV."""
        
        # Define parameter grids for different models
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        param_grid = param_grids.get(model_name, {})
        
        # Create pipeline based on strategy
        if strategy.method == 'class_weight':
            model = self.imbalance_handler.apply_class_weighting(base_model, X, y)
            pipeline = model
        elif strategy.method in ['oversample', 'hybrid']:
            pipeline = ImbPipeline([
                ('sampler', strategy.parameters['sampler']),
                ('classifier', base_model)
            ])
            # Prefix parameters for pipeline
            param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
        elif strategy.method == 'ensemble':
            pipeline = clone(strategy.parameters['classifier'])
            # Ensemble methods have their own parameters
            param_grid = {}
        
        # Perform grid search
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=self.config.optimization_metric,
            n_jobs=self.config.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(
            f"Grid search completed. Best score: {grid_search.best_score_:.4f}, "
            f"Best params: {grid_search.best_params_}"
        )
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search
    
    def _optimize_with_random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: BaseEstimator,
        strategy: ImbalanceStrategy,
        model_name: str
    ) -> Tuple[BaseEstimator, Dict[str, Any], Any]:
        """Optimize hyperparameters using RandomizedSearchCV."""
        
        from scipy.stats import uniform, randint
        
        # Define parameter distributions for different models
        param_distributions = {
            'logistic_regression': {
                'C': uniform(0.01, 100),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            },
            'gradient_boosting': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20)
            },
            'svm': {
                'C': uniform(0.01, 100),
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        }
        
        param_dist = param_distributions.get(model_name, {})
        
        # Create pipeline based on strategy
        if strategy.method == 'class_weight':
            model = self.imbalance_handler.apply_class_weighting(base_model, X, y)
            pipeline = model
        elif strategy.method in ['oversample', 'hybrid']:
            pipeline = ImbPipeline([
                ('sampler', strategy.parameters['sampler']),
                ('classifier', base_model)
            ])
            # Prefix parameters for pipeline
            param_dist = {f'classifier__{k}': v for k, v in param_dist.items()}
        elif strategy.method == 'ensemble':
            pipeline = clone(strategy.parameters['classifier'])
            param_dist = {}
        
        # Perform random search
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        random_search = RandomizedSearchCV(
            pipeline,
            param_dist,
            n_iter=min(self.config.optimization_trials, 50),  # Limit iterations for random search
            cv=cv,
            scoring=self.config.optimization_metric,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        logger.info(
            f"Random search completed. Best score: {random_search.best_score_:.4f}, "
            f"Best params: {random_search.best_params_}"
        )
        
        return random_search.best_estimator_, random_search.best_params_, random_search
    
    def train_with_cross_validation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model_name: Optional[str] = None,
        strategy_name: Optional[str] = None,
        optimize_hyperparameters: bool = True
    ) -> TrainingResult:
        """
        Train a model with cross-validation and imbalance handling.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of model to train (None for auto-selection)
            strategy_name: Name of strategy to use (None for auto-selection)
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            TrainingResult with comprehensive training information
        """
        logger.info("Starting model training with cross-validation")
        start_time = time.time()
        
        # Convert to numpy arrays for consistency
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Auto-select optimal strategy and model if not provided
        strategy_comparison_results = None
        if model_name is None or strategy_name is None:
            logger.info("Auto-selecting optimal strategy and model")
            strategy_comparison_results = self.compare_imbalance_strategies(X, y)
            
            if not strategy_comparison_results.empty:
                best_row = strategy_comparison_results.iloc[0]
                if strategy_name is None:
                    strategy_name = best_row['strategy']
                if model_name is None:
                    model_name = best_row['model']
            else:
                # Fallback defaults
                strategy_name = strategy_name or 'class_weight_balanced'
                model_name = model_name or 'random_forest'
        
        logger.info(f"Training {model_name} with {strategy_name} strategy")
        
        # Optimize hyperparameters if requested
        best_model = None
        best_params = {}
        study = None
        
        if optimize_hyperparameters:
            best_model, best_params, study = self.optimize_hyperparameters(
                X, y, model_name, strategy_name
            )
        else:
            # Use default model with strategy
            base_model = clone(self.base_models[model_name])
            strategy = self.imbalance_handler.strategies[strategy_name]
            
            if strategy.method == 'class_weight':
                best_model = self.imbalance_handler.apply_class_weighting(base_model, X, y)
            elif strategy.method in ['oversample', 'hybrid']:
                best_model = ImbPipeline([
                    ('sampler', strategy.parameters['sampler']),
                    ('classifier', base_model)
                ])
            elif strategy.method == 'ensemble':
                best_model = clone(strategy.parameters['classifier'])
        
        # Perform final cross-validation with best model
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Calculate multiple metrics
        metrics = ['roc_auc', 'f1', 'precision', 'recall']
        cv_scores = {}
        
        for metric in metrics:
            scores = cross_val_score(
                best_model, X, y, cv=cv,
                scoring=metric, n_jobs=self.config.n_jobs
            )
            cv_scores[f'{metric}_mean'] = scores.mean()
            cv_scores[f'{metric}_std'] = scores.std()
        
        # Fit the best model on full training data for later evaluation
        best_model.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Create training result
        result = TrainingResult(
            model_name=model_name,
            strategy_name=strategy_name,
            best_model=best_model,
            best_params=best_params,
            cv_scores=cv_scores,
            training_time=training_time,
            strategy_comparison_results=strategy_comparison_results,
            hyperparameter_study=study
        )
        
        # Store result
        self.training_results.append(result)
        
        logger.info(
            f"Training completed in {training_time:.2f}s. "
            f"CV ROC-AUC: {cv_scores['roc_auc_mean']:.3f} ± {cv_scores['roc_auc_std']:.3f}"
        )
        
        return result
    
    def evaluate_trained_model(
        self,
        training_result: TrainingResult,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        business_metrics: Optional[BusinessMetrics] = None
    ) -> ModelPerformance:
        """
        Evaluate a trained model on test data.
        
        Args:
            training_result: Result from model training
            X_test: Test feature matrix
            y_test: Test target vector
            business_metrics: Business parameters for ROI calculations
            
        Returns:
            ModelPerformance with comprehensive evaluation results
        """
        logger.info(f"Evaluating trained model: {training_result.model_name}")
        
        # Convert to numpy arrays if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Evaluate model
        performance = self.evaluator.evaluate_model(
            model=training_result.best_model,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{training_result.model_name}_{training_result.strategy_name}",
            calculate_business_metrics=business_metrics is not None
        )
        
        # Update training result with performance metrics
        training_result.performance_metrics = performance
        
        return performance
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all training results.
        
        Returns:
            DataFrame with training summary
        """
        if not self.training_results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.training_results:
            row = {
                'model_name': result.model_name,
                'strategy_name': result.strategy_name,
                'roc_auc_mean': result.cv_scores.get('roc_auc_mean', 0),
                'roc_auc_std': result.cv_scores.get('roc_auc_std', 0),
                'f1_mean': result.cv_scores.get('f1_mean', 0),
                'f1_std': result.cv_scores.get('f1_std', 0),
                'training_time': result.training_time,
                'n_params': len(result.best_params)
            }
            
            # Add performance metrics if available
            if result.performance_metrics:
                row.update({
                    'test_roc_auc': result.performance_metrics.auc_roc,
                    'test_f1': result.performance_metrics.f1_score,
                    'business_value': result.performance_metrics.business_value,
                    'roi_percentage': result.performance_metrics.roi_percentage
                })
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data).sort_values('roc_auc_mean', ascending=False)
    
    def save_training_results(self, output_path: Path) -> None:
        """
        Save training results to disk.
        
        Args:
            output_path: Path to save results
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_df = self.get_training_summary()
        summary_df.to_csv(output_path / 'training_summary.csv', index=False)
        
        # Save detailed results
        for i, result in enumerate(self.training_results):
            result_data = {
                'model_name': result.model_name,
                'strategy_name': result.strategy_name,
                'best_params': result.best_params,
                'cv_scores': result.cv_scores,
                'training_time': result.training_time
            }
            
            with open(output_path / f'training_result_{i}.json', 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            # Save strategy comparison if available
            if result.strategy_comparison_results is not None:
                result.strategy_comparison_results.to_csv(
                    output_path / f'strategy_comparison_{i}.csv', index=False
                )
        
        logger.info(f"Training results saved to {output_path}")


# Convenience functions for quick training
def train_imbalanced_model(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: Optional[str] = None,
    strategy_name: Optional[str] = None,
    optimize_hyperparameters: bool = True,
    business_metrics: Optional[BusinessMetrics] = None
) -> Tuple[TrainingResult, Optional[ModelPerformance]]:
    """
    Convenience function for training an imbalanced model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features (optional)
        y_test: Test target (optional)
        model_name: Model to use (None for auto-selection)
        strategy_name: Strategy to use (None for auto-selection)
        optimize_hyperparameters: Whether to optimize hyperparameters
        business_metrics: Business parameters for evaluation
        
    Returns:
        Tuple of (training_result, performance_metrics)
    """
    pipeline = ModelTrainingPipeline()
    
    # Train model
    training_result = pipeline.train_with_cross_validation(
        X_train, y_train,
        model_name=model_name,
        strategy_name=strategy_name,
        optimize_hyperparameters=optimize_hyperparameters
    )
    
    # Evaluate on test set if provided
    performance = None
    if X_test is not None and y_test is not None:
        performance = pipeline.evaluate_trained_model(
            training_result, X_test, y_test, business_metrics
        )
    
    return training_result, performance
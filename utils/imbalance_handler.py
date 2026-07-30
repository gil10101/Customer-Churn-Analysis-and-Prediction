"""
Class Imbalance Handling System for Customer Churn Prediction.

This module provides comprehensive techniques for handling class imbalance
in churn prediction datasets, including class weighting, oversampling,
undersampling, and ensemble methods.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

from .logging_setup import get_notebook_logger

logger = get_notebook_logger(__name__)


@dataclass
class ImbalanceStrategy:
    """Configuration for imbalance handling strategy."""
    name: str
    method: str  # 'class_weight', 'oversample', 'undersample', 'ensemble', 'hybrid'
    parameters: Dict[str, Any]
    description: str


@dataclass
class ImbalanceResults:
    """Results from imbalance handling evaluation."""
    strategy_name: str
    original_distribution: Dict[str, int]
    resampled_distribution: Dict[str, int]
    cv_scores: Dict[str, float]
    performance_metrics: Dict[str, float]
    execution_time: float


class ImbalanceHandler:
    """
    Comprehensive class imbalance handling system with multiple strategies.
    
    Supports class weighting, SMOTE variants, ensemble methods, and hybrid approaches
    for handling imbalanced churn prediction datasets.
    """
    
    def __init__(self, strategy: str = "adaptive", random_state: int = 42):
        """
        Initialize ImbalanceHandler.
        
        Args:
            strategy: Default strategy to use ('adaptive', 'class_weight', 'smote', 'ensemble')
            random_state: Random state for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.logger = get_notebook_logger(self.__class__.__name__)
        
        # Initialize available strategies
        self._initialize_strategies()
        
        # Store evaluation results
        self.evaluation_results: List[ImbalanceResults] = []
        
    def _initialize_strategies(self):
        """Initialize available imbalance handling strategies."""
        self.strategies = {
            'class_weight_balanced': ImbalanceStrategy(
                name='class_weight_balanced',
                method='class_weight',
                parameters={'class_weight': 'balanced'},
                description='Automatic class weighting based on frequency'
            ),
            'class_weight_custom': ImbalanceStrategy(
                name='class_weight_custom',
                method='class_weight',
                parameters={'class_weight': None},  # Will be computed
                description='Custom class weighting based on inverse frequency'
            ),
            'smote_regular': ImbalanceStrategy(
                name='smote_regular',
                method='oversample',
                parameters={'sampler': SMOTE(random_state=self.random_state)},
                description='Standard SMOTE oversampling'
            ),
            'smote_borderline': ImbalanceStrategy(
                name='smote_borderline',
                method='oversample',
                parameters={'sampler': BorderlineSMOTE(random_state=self.random_state)},
                description='Borderline SMOTE focusing on borderline cases'
            ),
            'adasyn': ImbalanceStrategy(
                name='adasyn',
                method='oversample',
                parameters={'sampler': ADASYN(random_state=self.random_state)},
                description='ADASYN adaptive synthetic sampling'
            ),
            'smote_svm': ImbalanceStrategy(
                name='smote_svm',
                method='oversample',
                parameters={'sampler': SVMSMOTE(random_state=self.random_state)},
                description='SVM-based SMOTE variant'
            ),
            'balanced_rf': ImbalanceStrategy(
                name='balanced_rf',
                method='ensemble',
                parameters={'classifier': BalancedRandomForestClassifier(random_state=self.random_state)},
                description='Balanced Random Forest with built-in sampling'
            ),
            'easy_ensemble': ImbalanceStrategy(
                name='easy_ensemble',
                method='ensemble',
                parameters={'classifier': EasyEnsembleClassifier(random_state=self.random_state)},
                description='Easy Ensemble with multiple balanced classifiers'
            ),
            'smote_tomek': ImbalanceStrategy(
                name='smote_tomek',
                method='hybrid',
                parameters={'sampler': SMOTETomek(random_state=self.random_state)},
                description='SMOTE oversampling + Tomek undersampling'
            ),
            'smote_enn': ImbalanceStrategy(
                name='smote_enn',
                method='hybrid',
                parameters={'sampler': SMOTEENN(random_state=self.random_state)},
                description='SMOTE oversampling + Edited Nearest Neighbours'
            )
        }
    
    def apply_class_weighting(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """
        Apply class weighting to a model.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Model with class weighting applied
        """
        self.logger.info("Applying class weighting to model")
        
        # Calculate class weights
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        self.logger.info(f"Computed class weights: {class_weight_dict}")
        
        # Apply class weights based on model type
        if hasattr(model, 'class_weight'):
            model.set_params(class_weight=class_weight_dict)
        elif isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            # For XGBoost and LightGBM, use scale_pos_weight
            pos_weight = class_weights[1] / class_weights[0] if len(class_weights) == 2 else 1.0
            if isinstance(model, xgb.XGBClassifier):
                model.set_params(scale_pos_weight=pos_weight)
            else:  # LightGBM
                model.set_params(class_weight=class_weight_dict)
        else:
            self.logger.warning(f"Class weighting not supported for {type(model).__name__}")
            
        return model
    
    def apply_smote_variants(self, X: np.ndarray, y: np.ndarray, 
                           variant: str = "borderline") -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE variants for oversampling.
        
        Args:
            X: Feature matrix
            y: Target vector
            variant: SMOTE variant ('smote', 'borderline', 'adasyn', 'svm')
            
        Returns:
            Resampled X and y arrays
        """
        self.logger.info(f"Applying {variant} oversampling")
        
        # Get original distribution
        original_dist = dict(zip(*np.unique(y, return_counts=True)))
        self.logger.info(f"Original distribution: {original_dist}")
        
        # Select sampler based on variant
        samplers = {
            'smote': SMOTE(random_state=self.random_state),
            'borderline': BorderlineSMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'svm': SVMSMOTE(random_state=self.random_state)
        }
        
        if variant not in samplers:
            raise ValueError(f"Unknown SMOTE variant: {variant}")
            
        sampler = samplers[variant]
        
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Log new distribution
            new_dist = dict(zip(*np.unique(y_resampled, return_counts=True)))
            self.logger.info(f"Resampled distribution: {new_dist}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error in {variant} resampling: {str(e)}")
            raise
    
    def create_balanced_ensemble(self, base_models: Optional[List[BaseEstimator]] = None) -> BaseEstimator:
        """
        Create balanced ensemble methods.
        
        Args:
            base_models: List of base models for ensemble (optional)
            
        Returns:
            Balanced ensemble classifier
        """
        self.logger.info("Creating balanced ensemble classifier")
        
        if base_models is None:
            # Use default ensemble methods
            ensemble_methods = [
                BalancedRandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                EasyEnsembleClassifier(
                    n_estimators=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            ]
            return ensemble_methods
        else:
            # Create custom balanced ensemble
            balanced_ensemble = []
            for model in base_models:
                # Wrap each model with balanced bagging
                balanced_model = BaggingClassifier(
                    estimator=model,
                    n_estimators=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                balanced_ensemble.append(balanced_model)
            
            return balanced_ensemble
    
    def evaluate_imbalance_strategies(self, X: np.ndarray, y: np.ndarray, 
                                    models: Optional[List[BaseEstimator]] = None,
                                    cv_folds: int = 5) -> pd.DataFrame:
        """
        Evaluate multiple imbalance handling strategies.
        
        Args:
            X: Feature matrix
            y: Target vector
            models: List of models to evaluate (optional)
            cv_folds: Number of cross-validation folds
            
        Returns:
            DataFrame with evaluation results
        """
        self.logger.info("Evaluating imbalance handling strategies")

        custom_models = models is not None
        if models is None:
            models = [
                LogisticRegression(random_state=self.random_state, max_iter=1000),
                RandomForestClassifier(random_state=self.random_state, n_estimators=100),
                DecisionTreeClassifier(random_state=self.random_state)
            ]
        
        results = []
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Evaluate each strategy
        for strategy_name, strategy in self.strategies.items():
            self.logger.info(f"Evaluating strategy: {strategy_name}")
            
            try:
                if strategy.method == 'class_weight':
                    # Class weighting approach
                    for model in models:
                        model_copy = self._clone_model(model)
                        weighted_model = self.apply_class_weighting(model_copy, X, y)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(weighted_model, X, y, cv=cv, 
                                                  scoring='roc_auc', n_jobs=-1)
                        f1_scores = cross_val_score(weighted_model, X, y, cv=cv, 
                                                  scoring='f1', n_jobs=-1)
                        
                        results.append({
                            'strategy': strategy_name,
                            'model': type(model).__name__,
                            'method': strategy.method,
                            'roc_auc_mean': cv_scores.mean(),
                            'roc_auc_std': cv_scores.std(),
                            'f1_mean': f1_scores.mean(),
                            'f1_std': f1_scores.std(),
                            'description': strategy.description
                        })
                
                elif strategy.method in ['oversample', 'hybrid']:
                    # Resampling approach
                    sampler = strategy.parameters['sampler']
                    
                    for model in models:
                        # Create pipeline with resampling
                        pipeline = ImbPipeline([
                            ('sampler', sampler),
                            ('classifier', self._clone_model(model))
                        ])
                        
                        # Cross-validation
                        cv_scores = cross_val_score(pipeline, X, y, cv=cv, 
                                                  scoring='roc_auc', n_jobs=-1)
                        f1_scores = cross_val_score(pipeline, X, y, cv=cv, 
                                                  scoring='f1', n_jobs=-1)
                        
                        results.append({
                            'strategy': strategy_name,
                            'model': type(model).__name__,
                            'method': strategy.method,
                            'roc_auc_mean': cv_scores.mean(),
                            'roc_auc_std': cv_scores.std(),
                            'f1_mean': f1_scores.mean(),
                            'f1_std': f1_scores.std(),
                            'description': strategy.description
                        })
                
                elif strategy.method == 'ensemble':
                    # Ensemble strategies substitute their own classifier; when
                    # the caller supplied specific models to evaluate, skip them
                    if custom_models:
                        continue
                    # Ensemble approach
                    ensemble_model = strategy.parameters['classifier']
                    
                    # Cross-validation
                    cv_scores = cross_val_score(ensemble_model, X, y, cv=cv, 
                                              scoring='roc_auc', n_jobs=-1)
                    f1_scores = cross_val_score(ensemble_model, X, y, cv=cv, 
                                              scoring='f1', n_jobs=-1)
                    
                    results.append({
                        'strategy': strategy_name,
                        'model': type(ensemble_model).__name__,
                        'method': strategy.method,
                        'roc_auc_mean': cv_scores.mean(),
                        'roc_auc_std': cv_scores.std(),
                        'f1_mean': f1_scores.mean(),
                        'f1_std': f1_scores.std(),
                        'description': strategy.description
                    })
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {strategy_name}: {str(e)}")
                continue
        
        results_df = pd.DataFrame(results)
        self.logger.info(f"Completed evaluation of {len(results)} strategy-model combinations")
        
        return results_df
    
    def get_optimal_strategy(self, X: np.ndarray, y: np.ndarray, 
                           metric: str = 'roc_auc', cv_folds: int = 5) -> str:
        """
        Get the optimal imbalance handling strategy based on cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            metric: Evaluation metric ('roc_auc' or 'f1')
            cv_folds: Number of cross-validation folds
            
        Returns:
            Name of optimal strategy
        """
        self.logger.info(f"Finding optimal strategy based on {metric}")
        
        # Evaluate all strategies
        results_df = self.evaluate_imbalance_strategies(X, y, cv_folds=cv_folds)
        
        # Find best strategy
        metric_col = f'{metric}_mean'
        if metric_col not in results_df.columns:
            raise ValueError(f"Metric {metric} not available in results")
        
        best_idx = results_df[metric_col].idxmax()
        best_strategy = results_df.loc[best_idx, 'strategy']
        best_score = results_df.loc[best_idx, metric_col]
        
        self.logger.info(f"Optimal strategy: {best_strategy} with {metric} = {best_score:.4f}")
        
        return best_strategy
    
    def apply_strategy(self, strategy_name: str, X: np.ndarray, y: np.ndarray,
                      model: Optional[BaseEstimator] = None) -> Union[Tuple[np.ndarray, np.ndarray], BaseEstimator]:
        """
        Apply a specific imbalance handling strategy.
        
        Args:
            strategy_name: Name of strategy to apply
            X: Feature matrix
            y: Target vector
            model: Model to apply strategy to (for class weighting)
            
        Returns:
            Either resampled (X, y) or modified model depending on strategy
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        self.logger.info(f"Applying strategy: {strategy_name}")
        
        if strategy.method == 'class_weight':
            if model is None:
                raise ValueError("Model required for class weighting strategy")
            return self.apply_class_weighting(model, X, y)
        
        elif strategy.method in ['oversample', 'hybrid']:
            sampler = strategy.parameters['sampler']
            return sampler.fit_resample(X, y)
        
        elif strategy.method == 'ensemble':
            return strategy.parameters['classifier']
        
        else:
            raise ValueError(f"Unknown strategy method: {strategy.method}")
    
    def _clone_model(self, model: BaseEstimator) -> BaseEstimator:
        """Clone a model with same parameters."""
        from sklearn.base import clone
        return clone(model)
    
    def get_strategy_summary(self) -> pd.DataFrame:
        """Get summary of available strategies."""
        summary_data = []
        for name, strategy in self.strategies.items():
            summary_data.append({
                'strategy_name': name,
                'method': strategy.method,
                'description': strategy.description
            })
        
        return pd.DataFrame(summary_data)
    
    def get_class_distribution(self, y: np.ndarray) -> Dict[str, Any]:
        """Get class distribution statistics."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        distribution = {}
        for cls, count in zip(unique, counts):
            distribution[f'class_{cls}'] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        # Calculate imbalance ratio
        if len(unique) == 2:
            minority_count = min(counts)
            majority_count = max(counts)
            imbalance_ratio = majority_count / minority_count
            distribution['imbalance_ratio'] = float(imbalance_ratio)
        
        return distribution
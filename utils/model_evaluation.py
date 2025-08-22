"""
Model Evaluation Module for Customer Churn Analysis.

This module provides comprehensive model evaluation capabilities including
standardized performance metrics calculation, bootstrap confidence intervals,
and business metrics integration with ROI and financial impact calculations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    log_loss, brier_score_loss
)
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
import warnings
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """
    Comprehensive model performance metrics dataclass.
    
    Stores all performance metrics including statistical measures,
    business metrics, and confidence intervals.
    """
    # Core classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0
    
    # Probabilistic metrics
    auc_roc: float = 0.0
    auc_pr: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    
    # Business metrics
    business_value: float = 0.0
    roi_percentage: float = 0.0
    cost_savings: float = 0.0
    revenue_impact: float = 0.0
    
    # Confidence intervals (95% by default)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Additional metrics
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None
    
    # Model metadata
    model_name: str = ""
    evaluation_timestamp: str = ""
    sample_size: int = 0
    
    def get_summary_dict(self) -> Dict[str, Any]:
        """Get summary dictionary of key metrics."""
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'business_value': self.business_value,
            'roi_percentage': self.roi_percentage,
            'sample_size': self.sample_size
        }
    
    def get_business_summary(self) -> Dict[str, Any]:
        """Get business-focused summary."""
        return {
            'model_name': self.model_name,
            'business_value': self.business_value,
            'roi_percentage': self.roi_percentage,
            'cost_savings': self.cost_savings,
            'revenue_impact': self.revenue_impact,
            'precision': self.precision,
            'recall': self.recall
        }


@dataclass
class BusinessMetrics:
    """Business impact calculation parameters."""
    customer_acquisition_cost: float = 100.0
    customer_retention_cost: float = 25.0
    average_customer_value: float = 500.0
    churn_cost_multiplier: float = 5.0
    intervention_success_rate: float = 0.7
    discount_rate: float = 0.1
    time_horizon_months: int = 12


class ModelEvaluator:
    """
    Comprehensive model evaluation engine.
    
    Provides standardized evaluation across multiple performance metrics
    with bootstrap confidence intervals and business impact calculations.
    """
    
    def __init__(
        self,
        business_metrics: Optional[BusinessMetrics] = None,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize model evaluator.
        
        Args:
            business_metrics: Business parameters for ROI calculations
            confidence_level: Confidence level for intervals (default 0.95)
            bootstrap_iterations: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.business_metrics = business_metrics or BusinessMetrics()
        self.confidence_level = confidence_level
        self.bootstrap_iterations = bootstrap_iterations
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        logger.info(f"Initialized ModelEvaluator with {bootstrap_iterations} bootstrap iterations")
    
    def evaluate_model(
        self,
        model: Any,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
        y_pred_proba: Optional[Union[pd.Series, np.ndarray]] = None,
        model_name: str = "Unknown Model",
        calculate_business_metrics: bool = True
    ) -> ModelPerformance:
        """
        Perform comprehensive model evaluation.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: True test labels
            y_pred: Predicted labels (optional, will be generated if not provided)
            y_pred_proba: Predicted probabilities (optional, will be generated if not provided)
            model_name: Name of the model for identification
            calculate_business_metrics: Whether to calculate business impact metrics
            
        Returns:
            ModelPerformance object with comprehensive evaluation results
        """
        logger.info(f"Starting evaluation for model: {model_name}")
        
        # Generate predictions if not provided
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        if y_pred_proba is None and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif y_pred_proba is None and hasattr(model, 'decision_function'):
            # Convert decision function to probabilities
            decision_scores = model.decision_function(X_test)
            y_pred_proba = 1 / (1 + np.exp(-decision_scores))  # Sigmoid transformation
        
        # Initialize performance object
        performance = ModelPerformance(
            model_name=model_name,
            evaluation_timestamp=pd.Timestamp.now().isoformat(),
            sample_size=len(y_test)
        )
        
        # Calculate core metrics
        performance = self._calculate_core_metrics(y_test, y_pred, y_pred_proba, performance)
        
        # Calculate confidence intervals using bootstrap
        performance.confidence_intervals = self._calculate_confidence_intervals(
            model, X_test, y_test, y_pred, y_pred_proba
        )
        
        # Calculate business metrics if requested
        if calculate_business_metrics and y_pred_proba is not None:
            performance = self._calculate_business_metrics(y_test, y_pred, y_pred_proba, performance)
        
        logger.info(f"Evaluation completed for {model_name}. AUC-ROC: {performance.auc_roc:.3f}")
        
        return performance
    
    def _calculate_core_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        performance: ModelPerformance
    ) -> ModelPerformance:
        """Calculate core classification metrics."""
        
        # Basic classification metrics
        performance.accuracy = accuracy_score(y_true, y_pred)
        performance.precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        performance.recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        performance.f1_score = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix and specificity
        cm = confusion_matrix(y_true, y_pred)
        performance.confusion_matrix = cm
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            performance.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Classification report
        try:
            performance.classification_report = classification_report(y_true, y_pred, output_dict=True)
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            performance.classification_report = {}
        
        # Probabilistic metrics (if probabilities available)
        if y_pred_proba is not None:
            try:
                performance.auc_roc = roc_auc_score(y_true, y_pred_proba)
                performance.auc_pr = average_precision_score(y_true, y_pred_proba)
                performance.log_loss = log_loss(y_true, y_pred_proba)
                performance.brier_score = brier_score_loss(y_true, y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probabilistic metrics: {e}")
        
        return performance
    
    def _calculate_confidence_intervals(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals for key metrics."""
        
        logger.debug(f"Calculating confidence intervals with {self.bootstrap_iterations} iterations")
        
        # Store bootstrap results
        bootstrap_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_roc': [],
            'auc_pr': []
        }
        
        n_samples = len(y_test)
        
        for i in range(self.bootstrap_iterations):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            y_true_boot = y_test[bootstrap_indices]
            y_pred_boot = y_pred[bootstrap_indices]
            
            # Calculate metrics for bootstrap sample
            try:
                bootstrap_results['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
                bootstrap_results['precision'].append(precision_score(y_true_boot, y_pred_boot, average='binary', zero_division=0))
                bootstrap_results['recall'].append(recall_score(y_true_boot, y_pred_boot, average='binary', zero_division=0))
                bootstrap_results['f1_score'].append(f1_score(y_true_boot, y_pred_boot, average='binary', zero_division=0))
                
                if y_pred_proba is not None:
                    y_pred_proba_boot = y_pred_proba[bootstrap_indices]
                    bootstrap_results['auc_roc'].append(roc_auc_score(y_true_boot, y_pred_proba_boot))
                    bootstrap_results['auc_pr'].append(average_precision_score(y_true_boot, y_pred_proba_boot))
                
            except Exception as e:
                logger.debug(f"Bootstrap iteration {i} failed: {e}")
                continue
        
        # Calculate confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        for metric, values in bootstrap_results.items():
            if values:  # Only calculate if we have values
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
                confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals
    
    def _calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        performance: ModelPerformance
    ) -> ModelPerformance:
        """Calculate business impact metrics and ROI."""
        
        bm = self.business_metrics
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            logger.warning("Cannot calculate business metrics: confusion matrix is not 2x2")
            return performance
        
        # Business impact calculations
        
        # True Positives: Correctly identified churners who can be retained
        retention_interventions = tp
        successful_retentions = retention_interventions * bm.intervention_success_rate
        retention_value = successful_retentions * bm.average_customer_value
        retention_cost = retention_interventions * bm.customer_retention_cost
        
        # False Positives: Unnecessary interventions
        unnecessary_interventions = fp
        unnecessary_cost = unnecessary_interventions * bm.customer_retention_cost
        
        # False Negatives: Missed churners (lost revenue)
        missed_churners = fn
        lost_revenue = missed_churners * bm.average_customer_value
        
        # True Negatives: Correctly identified non-churners (no action needed)
        # No direct cost or benefit
        
        # Total financial impact
        total_benefits = retention_value
        total_costs = retention_cost + unnecessary_cost
        net_benefit = total_benefits - total_costs
        
        # ROI calculation
        if total_costs > 0:
            roi_percentage = (net_benefit / total_costs) * 100
        else:
            roi_percentage = 0.0
        
        # Cost savings (compared to no intervention)
        baseline_churn_cost = (tp + fn) * bm.average_customer_value  # All actual churners
        intervention_churn_cost = fn * bm.average_customer_value + total_costs  # Remaining churners + intervention costs
        cost_savings = baseline_churn_cost - intervention_churn_cost
        
        # Revenue impact (positive retention value minus costs)
        revenue_impact = retention_value - total_costs
        
        # Business value score (normalized)
        total_customers = len(y_true)
        business_value = net_benefit / (total_customers * bm.average_customer_value) if total_customers > 0 else 0.0
        
        # Update performance object
        performance.business_value = business_value
        performance.roi_percentage = roi_percentage
        performance.cost_savings = cost_savings
        performance.revenue_impact = revenue_impact
        
        logger.debug(f"Business metrics calculated: ROI={roi_percentage:.1f}%, Revenue Impact=${revenue_impact:.0f}")
        
        return performance
    
    def compare_models(
        self,
        performances: List[ModelPerformance],
        primary_metric: str = 'auc_roc',
        include_business_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple model performances.
        
        Args:
            performances: List of ModelPerformance objects
            primary_metric: Primary metric for ranking
            include_business_metrics: Whether to include business metrics in comparison
            
        Returns:
            DataFrame with model comparison results
        """
        
        comparison_data = []
        
        for perf in performances:
            row = {
                'model_name': perf.model_name,
                'accuracy': perf.accuracy,
                'precision': perf.precision,
                'recall': perf.recall,
                'f1_score': perf.f1_score,
                'auc_roc': perf.auc_roc,
                'auc_pr': perf.auc_pr,
                'sample_size': perf.sample_size
            }
            
            if include_business_metrics:
                row.update({
                    'business_value': perf.business_value,
                    'roi_percentage': perf.roi_percentage,
                    'cost_savings': perf.cost_savings,
                    'revenue_impact': perf.revenue_impact
                })
            
            # Add confidence intervals if available
            for metric, ci in perf.confidence_intervals.items():
                row[f'{metric}_ci_lower'] = ci[0]
                row[f'{metric}_ci_upper'] = ci[1]
                row[f'{metric}_ci_width'] = ci[1] - ci[0]
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric (descending)
        if primary_metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(primary_metric, ascending=False)
        
        return comparison_df
    
    def calculate_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        optimization_metric: str = 'business_value'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate optimal probability threshold for business optimization.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            optimization_metric: Metric to optimize ('business_value', 'f1_score', 'precision', 'recall')
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics at this threshold
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred_thresh),
                'precision': precision_score(y_true, y_pred_thresh, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh, average='binary', zero_division=0),
                'f1_score': f1_score(y_true, y_pred_thresh, average='binary', zero_division=0)
            }
            
            # Calculate business value at this threshold
            if optimization_metric == 'business_value':
                cm = confusion_matrix(y_true, y_pred_thresh)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    
                    # Simplified business value calculation
                    bm = self.business_metrics
                    retention_value = tp * bm.intervention_success_rate * bm.average_customer_value
                    intervention_cost = (tp + fp) * bm.customer_retention_cost
                    net_benefit = retention_value - intervention_cost
                    
                    metrics['business_value'] = net_benefit / len(y_true) if len(y_true) > 0 else 0
                else:
                    metrics['business_value'] = 0
            
            threshold_results.append(metrics)
        
        # Find optimal threshold
        threshold_df = pd.DataFrame(threshold_results)
        
        if optimization_metric in threshold_df.columns:
            optimal_idx = threshold_df[optimization_metric].idxmax()
            optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
            optimal_metrics = threshold_df.loc[optimal_idx].to_dict()
        else:
            # Default to F1-score if metric not found
            optimal_idx = threshold_df['f1_score'].idxmax()
            optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
            optimal_metrics = threshold_df.loc[optimal_idx].to_dict()
        
        logger.info(f"Optimal threshold for {optimization_metric}: {optimal_threshold:.3f}")
        
        return optimal_threshold, optimal_metrics
    
    def cross_validate_model(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_folds: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target variable
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        
        logger.info(f"Performing {cv_folds}-fold cross-validation with {scoring} scoring")
        
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            
            results = {
                'cv_scores': cv_scores,
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'min_score': cv_scores.min(),
                'max_score': cv_scores.max(),
                'scoring_metric': scoring,
                'cv_folds': cv_folds
            }
            
            # Calculate confidence interval for mean score
            alpha = 1 - self.confidence_level
            margin_of_error = 1.96 * (cv_scores.std() / np.sqrt(cv_folds))  # Approximate 95% CI
            results['mean_score_ci'] = (
                results['mean_score'] - margin_of_error,
                results['mean_score'] + margin_of_error
            )
            
            logger.info(f"Cross-validation completed. Mean {scoring}: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def generate_evaluation_report(
        self,
        performance: ModelPerformance,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate comprehensive evaluation report."""
        
        report_lines = [
            "=" * 80,
            f"MODEL EVALUATION REPORT: {performance.model_name}",
            "=" * 80,
            f"Evaluation Date: {performance.evaluation_timestamp}",
            f"Sample Size: {performance.sample_size:,}",
            "",
            "PERFORMANCE METRICS:",
            "-" * 20,
            f"Accuracy:     {performance.accuracy:.4f}",
            f"Precision:    {performance.precision:.4f}",
            f"Recall:       {performance.recall:.4f}",
            f"F1-Score:     {performance.f1_score:.4f}",
            f"Specificity:  {performance.specificity:.4f}",
            "",
            "PROBABILISTIC METRICS:",
            "-" * 22,
            f"AUC-ROC:      {performance.auc_roc:.4f}",
            f"AUC-PR:       {performance.auc_pr:.4f}",
            f"Log Loss:     {performance.log_loss:.4f}",
            f"Brier Score:  {performance.brier_score:.4f}",
            "",
            "BUSINESS IMPACT:",
            "-" * 16,
            f"Business Value:   {performance.business_value:.4f}",
            f"ROI Percentage:   {performance.roi_percentage:.2f}%",
            f"Cost Savings:     ${performance.cost_savings:,.2f}",
            f"Revenue Impact:   ${performance.revenue_impact:,.2f}",
            ""
        ]
        
        # Add confidence intervals if available
        if performance.confidence_intervals:
            report_lines.extend([
                f"CONFIDENCE INTERVALS ({self.confidence_level:.0%}):",
                "-" * 35
            ])
            
            for metric, (lower, upper) in performance.confidence_intervals.items():
                report_lines.append(f"{metric:12}: [{lower:.4f}, {upper:.4f}]")
            
            report_lines.append("")
        
        # Add confusion matrix if available
        if performance.confusion_matrix is not None:
            cm = performance.confusion_matrix
            report_lines.extend([
                "CONFUSION MATRIX:",
                "-" * 17,
                f"                Predicted",
                f"              0      1",
                f"Actual   0   {cm[0,0]:4d}   {cm[0,1]:4d}",
                f"         1   {cm[1,0]:4d}   {cm[1,1]:4d}",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
    
    def save_performance_metrics(
        self,
        performance: ModelPerformance,
        output_path: Path
    ) -> None:
        """Save performance metrics to JSON file."""
        
        # Convert to serializable format
        metrics_dict = {
            'model_name': performance.model_name,
            'evaluation_timestamp': performance.evaluation_timestamp,
            'sample_size': performance.sample_size,
            'core_metrics': {
                'accuracy': performance.accuracy,
                'precision': performance.precision,
                'recall': performance.recall,
                'f1_score': performance.f1_score,
                'specificity': performance.specificity,
                'auc_roc': performance.auc_roc,
                'auc_pr': performance.auc_pr,
                'log_loss': performance.log_loss,
                'brier_score': performance.brier_score
            },
            'business_metrics': {
                'business_value': performance.business_value,
                'roi_percentage': performance.roi_percentage,
                'cost_savings': performance.cost_savings,
                'revenue_impact': performance.revenue_impact
            },
            'confidence_intervals': performance.confidence_intervals,
            'confusion_matrix': performance.confusion_matrix.tolist() if performance.confusion_matrix is not None else None,
            'classification_report': performance.classification_report
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Performance metrics saved to {output_path}")


# Convenience functions for quick evaluation
def evaluate_classification_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    model_name: str = "Model",
    business_metrics: Optional[BusinessMetrics] = None,
    generate_report: bool = False,
    report_path: Optional[Path] = None
) -> ModelPerformance:
    """
    Convenience function for quick model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name for the model
        business_metrics: Business parameters
        generate_report: Whether to generate text report
        report_path: Path to save report
        
    Returns:
        ModelPerformance object with evaluation results
    """
    evaluator = ModelEvaluator(business_metrics=business_metrics)
    performance = evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)
    
    if generate_report:
        report = evaluator.generate_evaluation_report(performance, report_path)
        if not report_path:
            print(report)
    
    return performance


def quick_model_comparison(
    models_and_names: List[Tuple[Any, str]],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    primary_metric: str = 'auc_roc'
) -> pd.DataFrame:
    """
    Quick comparison of multiple models.
    
    Args:
        models_and_names: List of (model, name) tuples
        X_test: Test features
        y_test: Test labels
        primary_metric: Primary metric for ranking
        
    Returns:
        DataFrame with model comparison
    """
    evaluator = ModelEvaluator()
    performances = []
    
    for model, name in models_and_names:
        performance = evaluator.evaluate_model(model, X_test, y_test, model_name=name)
        performances.append(performance)
    
    return evaluator.compare_models(performances, primary_metric=primary_metric)
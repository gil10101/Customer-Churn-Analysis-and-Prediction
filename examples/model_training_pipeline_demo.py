"""
Model Training Pipeline with Imbalance Handling Demo.

This script demonstrates how to use the integrated model training pipeline
that combines imbalance handling strategies with hyperparameter optimization
and cross-validation for imbalanced churn prediction datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_training_pipeline import (
    ModelTrainingPipeline, TrainingConfig, train_imbalanced_model
)
from utils.model_evaluation import BusinessMetrics


def create_sample_churn_data(n_samples=2000, imbalance_ratio=0.2):
    """Create a sample imbalanced churn dataset."""
    print(f"Creating sample dataset with {n_samples} samples and {imbalance_ratio:.1%} churn rate...")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[1-imbalance_ratio, imbalance_ratio],
        random_state=42
    )
    
    # Create feature names similar to churn dataset
    feature_names = [
        'tenure', 'monthly_charges', 'total_charges', 'usage_minutes',
        'data_usage_gb', 'support_tickets', 'contract_length', 'payment_method_risk',
        'service_count', 'billing_complexity', 'engagement_score', 'satisfaction_score',
        'late_payment_freq', 'usage_trend', 'value_ratio', 'cost_per_gb',
        'interaction_intensity', 'loyalty_score', 'churn_risk_score', 'customer_value'
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['churn'] = y
    
    print(f"Dataset created:")
    print(f"  - Total samples: {len(df):,}")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Churn rate: {y.mean():.1%}")
    print(f"  - Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return df


def demo_basic_training():
    """Demonstrate basic model training with imbalance handling."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Model Training with Imbalance Handling")
    print("="*60)
    
    # Create sample data
    df = create_sample_churn_data(n_samples=1000, imbalance_ratio=0.25)
    
    # Prepare data
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model using convenience function
    print("\nTraining logistic regression with class weighting...")
    training_result, performance = train_imbalanced_model(
        X_train, y_train, X_test, y_test,
        model_name='logistic_regression',
        strategy_name='class_weight_balanced',
        optimize_hyperparameters=False
    )
    
    # Display results
    print(f"\nTraining Results:")
    print(f"  - Model: {training_result.model_name}")
    print(f"  - Strategy: {training_result.strategy_name}")
    print(f"  - Training time: {training_result.training_time:.2f}s")
    print(f"  - CV ROC-AUC: {training_result.cv_scores['roc_auc_mean']:.3f} ± {training_result.cv_scores['roc_auc_std']:.3f}")
    print(f"  - CV F1-Score: {training_result.cv_scores['f1_mean']:.3f} ± {training_result.cv_scores['f1_std']:.3f}")
    
    print(f"\nTest Performance:")
    print(f"  - ROC-AUC: {performance.auc_roc:.3f}")
    print(f"  - F1-Score: {performance.f1_score:.3f}")
    print(f"  - Precision: {performance.precision:.3f}")
    print(f"  - Recall: {performance.recall:.3f}")


def demo_strategy_comparison():
    """Demonstrate strategy comparison and automatic selection."""
    print("\n" + "="*60)
    print("DEMO 2: Strategy Comparison and Auto-Selection")
    print("="*60)
    
    # Create sample data
    df = create_sample_churn_data(n_samples=800, imbalance_ratio=0.2)
    
    # Prepare data
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline with faster configuration for demo
    config = TrainingConfig(
        cv_folds=3,  # Faster cross-validation
        strategy_comparison_cv=3,
        optimization_trials=10
    )
    pipeline = ModelTrainingPipeline(config=config)
    
    # Compare strategies
    print("\nComparing imbalance handling strategies...")
    strategies = ['class_weight_balanced', 'smote_regular', 'balanced_rf']
    models = ['logistic_regression', 'random_forest']
    
    comparison_results = pipeline.compare_imbalance_strategies(
        X_train, y_train, models=models, strategies=strategies
    )
    
    print(f"\nStrategy Comparison Results:")
    print(comparison_results[['strategy', 'model', 'roc_auc_mean', 'f1_mean', 'training_time']].round(3))
    
    # Get optimal strategy
    optimal_strategy, optimal_model = pipeline.get_optimal_strategy(X_train, y_train)
    print(f"\nOptimal combination: {optimal_strategy} + {optimal_model}")
    
    # Train with optimal strategy
    print(f"\nTraining with optimal strategy...")
    training_result = pipeline.train_with_cross_validation(
        X_train, y_train,
        model_name=optimal_model,
        strategy_name=optimal_strategy,
        optimize_hyperparameters=False
    )
    
    # Evaluate
    performance = pipeline.evaluate_trained_model(training_result, X_test, y_test)
    
    print(f"\nOptimal Model Performance:")
    print(f"  - ROC-AUC: {performance.auc_roc:.3f}")
    print(f"  - F1-Score: {performance.f1_score:.3f}")
    print(f"  - Precision: {performance.precision:.3f}")
    print(f"  - Recall: {performance.recall:.3f}")


def demo_business_metrics():
    """Demonstrate business metrics integration."""
    print("\n" + "="*60)
    print("DEMO 3: Business Metrics Integration")
    print("="*60)
    
    # Create sample data
    df = create_sample_churn_data(n_samples=600, imbalance_ratio=0.3)
    
    # Prepare data
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define business parameters
    business_metrics = BusinessMetrics(
        customer_acquisition_cost=150.0,
        customer_retention_cost=30.0,
        average_customer_value=600.0,
        churn_cost_multiplier=5.0,
        intervention_success_rate=0.7
    )
    
    print(f"\nBusiness Parameters:")
    print(f"  - Customer acquisition cost: ${business_metrics.customer_acquisition_cost}")
    print(f"  - Customer retention cost: ${business_metrics.customer_retention_cost}")
    print(f"  - Average customer value: ${business_metrics.average_customer_value}")
    print(f"  - Intervention success rate: {business_metrics.intervention_success_rate:.1%}")
    
    # Train model with business metrics
    print(f"\nTraining model with business metrics...")
    training_result, performance = train_imbalanced_model(
        X_train, y_train, X_test, y_test,
        model_name='random_forest',
        strategy_name='smote_regular',
        optimize_hyperparameters=False,
        business_metrics=business_metrics
    )
    
    print(f"\nBusiness Impact Analysis:")
    print(f"  - Business Value Score: {performance.business_value:.4f}")
    print(f"  - ROI Percentage: {performance.roi_percentage:.1f}%")
    print(f"  - Cost Savings: ${performance.cost_savings:,.2f}")
    print(f"  - Revenue Impact: ${performance.revenue_impact:,.2f}")
    
    print(f"\nModel Performance:")
    print(f"  - ROC-AUC: {performance.auc_roc:.3f}")
    print(f"  - F1-Score: {performance.f1_score:.3f}")
    print(f"  - Precision: {performance.precision:.3f}")
    print(f"  - Recall: {performance.recall:.3f}")


def demo_multiple_models():
    """Demonstrate training and comparing multiple models."""
    print("\n" + "="*60)
    print("DEMO 4: Multiple Model Training and Comparison")
    print("="*60)
    
    # Create sample data
    df = create_sample_churn_data(n_samples=800, imbalance_ratio=0.25)
    
    # Prepare data
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline
    config = TrainingConfig(cv_folds=3, optimization_trials=5)
    pipeline = ModelTrainingPipeline(config=config)
    
    # Train multiple models
    models_to_train = [
        ('logistic_regression', 'class_weight_balanced'),
        ('random_forest', 'smote_regular'),
        ('gradient_boosting', 'class_weight_balanced'),
    ]
    
    print(f"\nTraining {len(models_to_train)} different model-strategy combinations...")
    
    for model_name, strategy_name in models_to_train:
        print(f"\nTraining {model_name} with {strategy_name}...")
        
        training_result = pipeline.train_with_cross_validation(
            X_train, y_train,
            model_name=model_name,
            strategy_name=strategy_name,
            optimize_hyperparameters=False
        )
        
        # Evaluate on test set
        performance = pipeline.evaluate_trained_model(training_result, X_test, y_test)
        
        print(f"  - CV ROC-AUC: {training_result.cv_scores['roc_auc_mean']:.3f}")
        print(f"  - Test ROC-AUC: {performance.auc_roc:.3f}")
        print(f"  - Test F1: {performance.f1_score:.3f}")
    
    # Get training summary
    print(f"\nTraining Summary:")
    summary = pipeline.get_training_summary()
    print(summary[['model_name', 'strategy_name', 'roc_auc_mean', 'test_roc_auc', 'test_f1']].round(3))
    
    # Find best model
    best_model_idx = summary['test_roc_auc'].idxmax()
    best_model_info = summary.iloc[best_model_idx]
    
    print(f"\nBest Model:")
    print(f"  - Model: {best_model_info['model_name']}")
    print(f"  - Strategy: {best_model_info['strategy_name']}")
    print(f"  - Test ROC-AUC: {best_model_info['test_roc_auc']:.3f}")
    print(f"  - Test F1: {best_model_info['test_f1']:.3f}")


def main():
    """Run all demos."""
    print("Model Training Pipeline with Imbalance Handling - Demo")
    print("=" * 60)
    
    try:
        # Run demos
        demo_basic_training()
        demo_strategy_comparison()
        demo_business_metrics()
        demo_multiple_models()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
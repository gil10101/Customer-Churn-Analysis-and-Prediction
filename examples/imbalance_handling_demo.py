"""
Demonstration of ImbalanceHandler capabilities for customer churn prediction.

This script shows how to use the ImbalanceHandler class to handle class imbalance
in churn prediction datasets using various techniques.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.imbalance_handler import ImbalanceHandler


def create_imbalanced_churn_data():
    """Create synthetic imbalanced churn dataset."""
    print("Creating synthetic imbalanced churn dataset...")
    
    # Create imbalanced dataset similar to churn data
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.75, 0.25],  # 75% no churn, 25% churn
        random_state=42
    )
    
    # Create feature names similar to churn data
    feature_names = [
        'tenure', 'monthly_charges', 'total_charges', 'contract_length',
        'payment_method_score', 'service_count', 'support_calls', 'complaints',
        'usage_minutes', 'data_usage', 'satisfaction_score', 'age_group',
        'family_size', 'income_level', 'region_code', 'promotion_usage',
        'billing_issues', 'service_changes', 'competitor_offers', 'loyalty_score'
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['churn'] = y
    
    print(f"Dataset created: {len(df)} samples, {len(feature_names)} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return df


def demonstrate_class_weighting():
    """Demonstrate class weighting techniques."""
    print("\n" + "="*60)
    print("DEMONSTRATING CLASS WEIGHTING TECHNIQUES")
    print("="*60)
    
    # Create data
    df = create_imbalanced_churn_data()
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize handler
    handler = ImbalanceHandler(random_state=42)
    
    # Test different models with class weighting
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
        ('Random Forest', RandomForestClassifier(random_state=42, n_estimators=100))
    ]
    
    for model_name, model in models:
        print(f"\n{model_name} with Class Weighting:")
        
        # Apply class weighting
        weighted_model = handler.apply_class_weighting(model, X_train, y_train)
        
        # Train and evaluate
        weighted_model.fit(X_train, y_train)
        y_pred = weighted_model.predict(X_test)
        y_pred_proba = weighted_model.predict_proba(X_test)[:, 1]
        
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))


def demonstrate_smote_variants():
    """Demonstrate SMOTE oversampling variants."""
    print("\n" + "="*60)
    print("DEMONSTRATING SMOTE OVERSAMPLING VARIANTS")
    print("="*60)
    
    # Create data
    df = create_imbalanced_churn_data()
    X = df.drop('churn', axis=1).values
    y = df['churn'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize handler
    handler = ImbalanceHandler(random_state=42)
    
    # Test different SMOTE variants
    smote_variants = ['smote', 'borderline', 'adasyn']
    
    for variant in smote_variants:
        print(f"\n{variant.upper()} Oversampling:")
        
        # Apply SMOTE variant
        X_resampled, y_resampled = handler.apply_smote_variants(
            X_train, y_train, variant=variant
        )
        
        print(f"Original training set: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"Resampled training set: {dict(zip(*np.unique(y_resampled, return_counts=True)))}")
        
        # Train model on resampled data
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_resampled, y_resampled)
        
        # Evaluate on original test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")


def demonstrate_ensemble_methods():
    """Demonstrate balanced ensemble methods."""
    print("\n" + "="*60)
    print("DEMONSTRATING BALANCED ENSEMBLE METHODS")
    print("="*60)
    
    # Create data
    df = create_imbalanced_churn_data()
    X = df.drop('churn', axis=1).values
    y = df['churn'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize handler
    handler = ImbalanceHandler(random_state=42)
    
    # Test balanced ensemble methods
    ensemble_methods = handler.create_balanced_ensemble()
    
    for i, ensemble_model in enumerate(ensemble_methods):
        model_name = type(ensemble_model).__name__
        print(f"\n{model_name}:")
        
        # Train and evaluate
        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)
        y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
        
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))


def demonstrate_strategy_evaluation():
    """Demonstrate comprehensive strategy evaluation."""
    print("\n" + "="*60)
    print("DEMONSTRATING STRATEGY EVALUATION")
    print("="*60)
    
    # Create smaller dataset for faster evaluation
    df = create_imbalanced_churn_data()
    X = df.drop('churn', axis=1).values[:500]  # Use subset for demo
    y = df['churn'].values[:500]
    
    # Initialize handler
    handler = ImbalanceHandler(random_state=42)
    
    # Get strategy summary
    print("\nAvailable Strategies:")
    strategy_summary = handler.get_strategy_summary()
    print(strategy_summary.to_string(index=False))
    
    # Evaluate strategies (using subset for speed)
    print("\nEvaluating imbalance handling strategies...")
    results_df = handler.evaluate_imbalance_strategies(X, y, cv_folds=3)
    
    # Show top strategies by ROC AUC
    print("\nTop 5 Strategies by ROC AUC:")
    top_strategies = results_df.nlargest(5, 'roc_auc_mean')[
        ['strategy', 'model', 'roc_auc_mean', 'f1_mean']
    ]
    print(top_strategies.to_string(index=False))
    
    # Find optimal strategy
    optimal_strategy = handler.get_optimal_strategy(X, y, cv_folds=3)
    print(f"\nOptimal strategy: {optimal_strategy}")


def demonstrate_class_distribution_analysis():
    """Demonstrate class distribution analysis."""
    print("\n" + "="*60)
    print("DEMONSTRATING CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Create data
    df = create_imbalanced_churn_data()
    y = df['churn'].values
    
    # Initialize handler
    handler = ImbalanceHandler(random_state=42)
    
    # Analyze class distribution
    distribution = handler.get_class_distribution(y)
    
    print("Class Distribution Analysis:")
    for key, value in distribution.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value:.2f}")


def main():
    """Run all demonstrations."""
    print("IMBALANCE HANDLER DEMONSTRATION")
    print("="*60)
    print("This demo shows various techniques for handling class imbalance")
    print("in customer churn prediction datasets.")
    
    # Run demonstrations
    demonstrate_class_distribution_analysis()
    demonstrate_class_weighting()
    demonstrate_smote_variants()
    demonstrate_ensemble_methods()
    demonstrate_strategy_evaluation()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("The ImbalanceHandler provides comprehensive tools for handling")
    print("class imbalance in churn prediction models. Choose the strategy")
    print("that works best for your specific dataset and business requirements.")


if __name__ == "__main__":
    main()
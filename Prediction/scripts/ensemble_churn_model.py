#!/usr/bin/env python3
# Advanced Ensemble Model for Customer Churn Prediction

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Import common utilities
from utils.data_preprocessing import load_telco_data, prepare_data_for_modeling

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../evaluation/images', exist_ok=True)
os.makedirs('../evaluation/docs', exist_ok=True)

def build_and_train_ensemble_model(X_train, y_train, cv=5):
    """
    Build and train an ensemble model for churn prediction
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    tuple
        (best_model, feature_importances)
    """
    print("Building and training ensemble model...")
    
    # Define base models
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Create ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('logistic', log_reg),
            ('random_forest', rf_clf),
            ('gradient_boosting', gb_clf)
        ],
        voting='soft'
    )
    
    # Train model
    ensemble.fit(X_train, y_train)
    
    # Get feature importances
    feature_importances = None
    if hasattr(ensemble.named_estimators_['random_forest'], 'feature_importances_'):
        feature_importances = ensemble.named_estimators_['random_forest'].feature_importances_
    
    # Save model
    joblib.dump(ensemble, '../models/ensemble_churn_model.joblib')
    
    return ensemble, feature_importances

def train_optimized_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train an optimized Gradient Boosting model with hyperparameter tuning
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test targets
    feature_names : list
        Names of features
        
    Returns:
    --------
    tuple
        (best_model, feature_importances)
    """
    print("Training optimized Gradient Boosting model with hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    # Create base model
    gb = GradientBoostingClassifier(random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=gb,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    
    # Get feature importances
    feature_importances = best_model.feature_importances_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    print("\nOptimized Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(best_model, '../models/optimized_gb_model.joblib')
    
    return best_model, feature_importances

def visualize_feature_importance(feature_importances, feature_names, model_name="Model"):
    """
    Create and save feature importance visualization
    
    Parameters:
    -----------
    feature_importances : numpy.ndarray
        Feature importance scores
    feature_names : list
        Names of features
    model_name : str
        Name of the model for plot title
    """
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Top 15 Features by Importance ({model_name})')
    plt.tight_layout()
    plt.savefig(f'../evaluation/images/feature_importance_{model_name.lower().replace(" ", "_")}.png')
    
    return importance_df

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance and generate visualizations
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Name of the model for plot titles
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'../evaluation/images/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'../evaluation/images/roc_curve_{model_name.lower().replace(" ", "_")}.png')
    
    # Save evaluation results
    with open(f'../evaluation/docs/{model_name.lower().replace(" ", "_")}_evaluation.md', 'w') as f:
        f.write(f"# {model_name} Evaluation\n\n")
        
        f.write("## Evaluation Metrics\n\n")
        f.write(f"- **Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Precision**: {precision:.4f}\n")
        f.write(f"- **Recall**: {recall:.4f}\n")
        f.write(f"- **F1 Score**: {f1:.4f}\n")
        f.write(f"- **ROC AUC**: {roc_auc:.4f}\n\n")
        
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n```\n\n")
        
        f.write("## Insights\n\n")
        f.write(f"- The {model_name} achieved good performance in predicting customer churn.\n")
        f.write("- The model's strength is in its ability to combine multiple algorithms, reducing overfitting and improving generalization.\n")
        f.write("- To further improve performance, consider feature engineering, collecting more data, or exploring deep learning approaches.\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def main():
    """Main function to run the ensemble model pipeline"""
    print("Advanced Ensemble Model for Customer Churn Prediction")
    print("="*50)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    
    # Use absolute path with os.path.join to ensure correct path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    data_path = os.path.join(project_root, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Ensure file exists before proceeding
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    df = load_telco_data(data_path)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_data_for_modeling(df)
    
    # Train ensemble model
    ensemble_model, ensemble_importances = build_and_train_ensemble_model(X_train, y_train)
    
    # Evaluate ensemble model
    ensemble_metrics = evaluate_model(ensemble_model, X_test, y_test, "Ensemble Model")
    
    # Visualize feature importance for ensemble model
    if ensemble_importances is not None:
        importance_df = visualize_feature_importance(ensemble_importances, feature_names, "Ensemble Model")
        print("\nTop 10 features by importance (Ensemble Model):")
        print(importance_df.head(10))
    
    # Train optimized Gradient Boosting model
    gb_model, gb_importances = train_optimized_model(X_train, y_train, X_test, y_test, feature_names)
    
    # Evaluate optimized model
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "Optimized GB Model")
    
    # Visualize feature importance for optimized model
    importance_df = visualize_feature_importance(gb_importances, feature_names, "Optimized GB Model")
    print("\nTop 10 features by importance (Optimized GB Model):")
    print(importance_df.head(10))
    
    # Compare models
    print("\nModel Comparison:")
    models = ["Ensemble Model", "Optimized GB Model"]
    metrics = [ensemble_metrics, gb_metrics]
    
    comparison_df = pd.DataFrame(metrics, index=models)
    print(comparison_df)
    
    # Save comparison to markdown
    with open('../evaluation/docs/model_comparison.md', 'w') as f:
        f.write("# Model Comparison\n\n")
        f.write(comparison_df.to_markdown())
        f.write("\n\n## Conclusion\n\n")
        
        best_model = models[0] if metrics[0]['f1'] > metrics[1]['f1'] else models[1]
        f.write(f"The {best_model} performs better overall based on F1 score, which balances precision and recall.\n")
        f.write("This model should be used for production deployment for churn prediction.\n")
    
    print("\nModel training and evaluation completed successfully.")
    print("Results saved to ../evaluation/docs/")
    print("Visualizations saved to ../evaluation/images/")

if __name__ == "__main__":
    main() 
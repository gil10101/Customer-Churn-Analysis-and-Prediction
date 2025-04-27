#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements a cost-sensitive churn prediction model that optimizes for 
minimizing business costs rather than just accuracy metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import xgboost as xgb
import joblib

# Add the parent directory to the path to import from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_preprocessing import load_and_preprocess_data

# Create directories for saving results and models if they don't exist
# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'Prediction', 'models', 'cost_sensitive')
EVAL_DIR = os.path.join(PROJECT_ROOT, 'Prediction', 'evaluation', 'cost_sensitive_model_evaluation')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

class CostSensitiveChurnModel:
    """
    A class to build and evaluate cost-sensitive churn prediction models.
    The models are optimized to minimize business costs rather than just accuracy.
    """
    
    def __init__(self, retention_cost=100, churn_cost=500):
        """
        Initialize the cost-sensitive churn model.
        
        Parameters:
        -----------
        retention_cost : float
            Cost of retention efforts (e.g., discounts or loyalty programs) per customer
        churn_cost : float
            Cost of losing a customer (lost revenue, acquisition cost of replacement)
        """
        self.retention_cost = retention_cost
        self.churn_cost = churn_cost
        self.models = {}
        self.best_model = None
        self.best_threshold = 0.5
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
    
    def prepare_data(self, df):
        """
        Prepare the data for modeling.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The preprocessed customer churn dataset
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : numpy.ndarray
            Split training and testing data
        """
        # Convert churn target to binary
        df['Churn_Binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Split features and target
        X = df.drop(['customerID', 'Churn', 'Churn_Binary'], axis=1)
        y = df['Churn_Binary']
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale numerical features
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        # Encode categorical variables
        X_train = pd.get_dummies(X_train, drop_first=True)
        X_test = pd.get_dummies(X_test, drop_first=True)
        
        # Ensure X_train and X_test have the same columns
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        # Ensure the order of columns is the same
        X_test = X_test[X_train.columns]
        
        # Update feature names to reflect encoded columns
        self.feature_names = X_train.columns.tolist()
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        """
        Train multiple models for churn prediction.
        """
        # Define models to train
        models = {
            'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(scale_pos_weight=sum(self.y_train == 0) / sum(self.y_train == 1), random_state=42)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            # Save the model
            joblib.dump(model, os.path.join(MODELS_DIR, f'{name}_model.pkl'))
        
        print("All models trained successfully.")
    
    def calculate_cost_matrix(self):
        """
        Calculate the cost matrix for cost-sensitive evaluation.
        
        Returns:
        --------
        cost_matrix : dict
            Dictionary containing the costs for each outcome (TP, FP, TN, FN)
        """
        # True Positive (TP): Predict churn correctly, apply retention efforts
        # Cost: Retention cost
        tp_cost = self.retention_cost
        
        # False Positive (FP): Predict churn incorrectly, apply unnecessary retention efforts
        # Cost: Retention cost
        fp_cost = self.retention_cost
        
        # True Negative (TN): Predict non-churn correctly, do nothing
        # Cost: 0
        tn_cost = 0
        
        # False Negative (FN): Predict non-churn incorrectly, customer churns without intervention
        # Cost: Churn cost
        fn_cost = self.churn_cost
        
        cost_matrix = {
            'TP': tp_cost,
            'FP': fp_cost,
            'TN': tn_cost,
            'FN': fn_cost
        }
        
        return cost_matrix
    
    def calculate_total_cost(self, y_true, y_pred):
        """
        Calculate the total cost of a model's predictions.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
            
        Returns:
        --------
        total_cost : float
            Total cost of the predictions
        """
        # Get the confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Get the cost matrix
        cost_matrix = self.calculate_cost_matrix()
        
        # Calculate total cost
        total_cost = (tp * cost_matrix['TP'] + 
                      fp * cost_matrix['FP'] + 
                      tn * cost_matrix['TN'] + 
                      fn * cost_matrix['FN'])
        
        return total_cost
    
    def find_optimal_threshold(self, model_name):
        """
        Find the optimal decision threshold that minimizes total costs.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to optimize
            
        Returns:
        --------
        optimal_threshold : float
            The threshold that minimizes total costs
        """
        model = self.models[model_name]
        
        # Get predicted probabilities
        if model_name == 'xgboost':
            y_probs = model.predict_proba(self.X_test)[:, 1]
        else:
            y_probs = model.predict_proba(self.X_test)[:, 1]
        
        # Initialize variables to track optimal threshold
        min_cost = float('inf')
        optimal_threshold = 0.5
        
        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            cost = self.calculate_total_cost(self.y_test, y_pred)
            costs.append(cost)
            
            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold
        
        # Plot costs vs thresholds
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, costs)
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                   label=f'Optimal threshold: {optimal_threshold:.2f}, Cost: ${min_cost:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Total Cost ($)')
        plt.title(f'Total Cost vs. Decision Threshold for {model_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f'{model_name}_threshold_optimization.png'))
        plt.close()
        
        return optimal_threshold
    
    def evaluate_models(self):
        """
        Evaluate all models based on cost metrics and select the best one.
        
        Returns:
        --------
        evaluation_results : dict
            Dictionary containing evaluation metrics for each model
        """
        evaluation_results = {}
        
        # Calculate the baseline cost (no model, assume no one churns)
        baseline_preds = np.zeros_like(self.y_test)
        baseline_cost = self.calculate_total_cost(self.y_test, baseline_preds)
        evaluation_results['baseline'] = {'cost': baseline_cost}
        
        # Evaluate each model
        for name, model in self.models.items():
            # Find optimal threshold
            optimal_threshold = self.find_optimal_threshold(name)
            
            # Get predictions using optimal threshold
            if name == 'xgboost':
                y_probs = model.predict_proba(self.X_test)[:, 1]
            else:
                y_probs = model.predict_proba(self.X_test)[:, 1]
            
            y_pred = (y_probs >= optimal_threshold).astype(int)
            
            # Calculate metrics
            total_cost = self.calculate_total_cost(self.y_test, y_pred)
            cost_savings = baseline_cost - total_cost
            
            # Store evaluation results
            evaluation_results[name] = {
                'optimal_threshold': optimal_threshold,
                'cost': total_cost,
                'cost_savings': cost_savings,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            # Print evaluation results
            print(f"\nEvaluation results for {name}:")
            print(f"Optimal threshold: {optimal_threshold:.4f}")
            print(f"Total cost: ${total_cost:.2f}")
            print(f"Cost savings over baseline: ${cost_savings:.2f}")
            print(f"Classification report:\n{classification_report(self.y_test, y_pred)}")
        
        # Find the best model based on cost savings
        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x].get('cost_savings', 0))
        
        if best_model_name != 'baseline':
            self.best_model = self.models[best_model_name]
            self.best_threshold = evaluation_results[best_model_name]['optimal_threshold']
            print(f"\nBest model: {best_model_name} with cost savings of ${evaluation_results[best_model_name]['cost_savings']:.2f}")
        
        # Plot confusion matrices for all models
        self.plot_confusion_matrices(evaluation_results)
        
        # Plot ROC curves for all models
        self.plot_roc_curves()
        
        # Plot feature importance for the best model
        if best_model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
            self.plot_feature_importance(best_model_name)
        
        return evaluation_results
    
    def plot_confusion_matrices(self, evaluation_results):
        """
        Plot confusion matrices for all models.
        
        Parameters:
        -----------
        evaluation_results : dict
            Dictionary containing evaluation metrics for each model
        """
        # Skip baseline in plotting
        models_to_plot = [name for name in evaluation_results.keys() if name != 'baseline']
        
        plt.figure(figsize=(16, 4 * len(models_to_plot)))
        
        for i, name in enumerate(models_to_plot):
            cm = evaluation_results[name]['confusion_matrix']
            
            plt.subplot(len(models_to_plot), 2, 2*i+1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            # Also plot normalized confusion matrix
            plt.subplot(len(models_to_plot), 2, 2*i+2)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
            plt.title(f'Normalized Confusion Matrix - {name}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrices.png'))
        plt.close()
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for all models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            if name == 'xgboost':
                y_probs = model.predict_proba(self.X_test)[:, 1]
            else:
                y_probs = model.predict_proba(self.X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(self.y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Add the cost-optimal threshold for the best model
        if self.best_model is not None:
            # Get name of best model
            best_model_name = next(name for name, model in self.models.items() if model == self.best_model)
            
            if best_model_name == 'xgboost':
                y_probs = self.best_model.predict_proba(self.X_test)[:, 1]
            else:
                y_probs = self.best_model.predict_proba(self.X_test)[:, 1]
            
            y_pred = (y_probs >= self.best_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
            
            # Calculate true and false positive rates for this threshold
            tpr_at_threshold = tp / (tp + fn)
            fpr_at_threshold = fp / (fp + tn)
            
            plt.plot(fpr_at_threshold, tpr_at_threshold, 'ro', markersize=10,
                    label=f'Cost-optimal threshold ({self.best_threshold:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, 'roc_curves.png'))
        plt.close()
    
    def plot_feature_importance(self, model_name):
        """
        Plot feature importances for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot feature importances for
        """
        model = self.models[model_name]
        
        # Get feature importances
        if model_name == 'xgboost':
            importances = model.feature_importances_
        else:
            importances = model.feature_importances_
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f'{model_name}_feature_importance.png'))
        plt.close()
    
    def generate_cost_analysis_report(self, evaluation_results):
        """
        Generate a comprehensive cost analysis report.
        
        Parameters:
        -----------
        evaluation_results : dict
            Dictionary containing evaluation metrics for each model
        """
        # Skip baseline in report generation
        models_to_report = [name for name in evaluation_results.keys() if name != 'baseline']
        
        # Calculate potential annual savings for each model
        baseline_cost = evaluation_results['baseline']['cost']
        customer_count = len(self.y_test)
        
        # Adjust to annual cost (assuming monthly costs in our initial calculation)
        annual_multiplier = 12
        annual_baseline_cost = baseline_cost * annual_multiplier
        
        # Display baseline cost at the beginning
        print(f"\nBaseline Cost if No Model: ${baseline_cost:.2f} per period (${annual_baseline_cost:.2f} annually)")
        
        report = [
            "# Cost-Sensitive Churn Model Evaluation Report\n\n",
            "## Overview\n",
            "This report presents the results of cost-sensitive churn prediction models. The models are evaluated based on their ability to minimize the total business cost, rather than just optimizing for accuracy metrics.\n\n",
            f"**Baseline Cost if No Model: ${baseline_cost:.2f} per period (${annual_baseline_cost:.2f} annually)**\n\n",
            "## Cost Parameters\n",
            f"- Retention Cost: ${self.retention_cost} per customer (cost of offering discounts, loyalty programs, etc.)\n",
            f"- Churn Cost: ${self.churn_cost} per customer (lost revenue, acquisition cost of replacement)\n\n",
            "## Model Performance\n\n",
            "### Cost Comparison\n",
            "| Model | Total Cost | Cost Savings | Annual Savings (Estimated) | Optimal Threshold |\n",
            "|-------|------------|--------------|----------------------------|------------------|\n"
        ]
        
        for name in models_to_report:
            results = evaluation_results[name]
            total_cost = results['cost']
            cost_savings = results['cost_savings']
            annual_savings = cost_savings * annual_multiplier
            optimal_threshold = results['optimal_threshold']
            
            report.append(f"| {name} | ${total_cost:.2f} | ${cost_savings:.2f} | ${annual_savings:.2f} | {optimal_threshold:.3f} |\n")
        
        report.append("\n### Classification Metrics for Cost-Optimal Models\n")
        report.append("| Model | Accuracy | Precision | Recall | F1 Score |\n")
        report.append("|-------|----------|-----------|--------|----------|\n")
        
        for name in models_to_report:
            metrics = evaluation_results[name]['classification_report']
            accuracy = metrics['accuracy']
            precision = metrics['1']['precision']
            recall = metrics['1']['recall']
            f1 = metrics['1']['f1-score']
            
            report.append(f"| {name} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {f1:.3f} |\n")
        
        report.append("\n## Interpretation\n\n")
        
        # Find the best model
        best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x].get('cost_savings', 0) if x != 'baseline' else 0)
        best_results = evaluation_results[best_model_name]
        
        report.append(f"### Best Model: {best_model_name}\n")
        report.append(f"The {best_model_name} model provides the highest cost savings at ${best_results['cost_savings']:.2f} per prediction period, with an estimated annual savings of ${best_results['cost_savings'] * annual_multiplier:.2f}.\n\n")
        
        report.append("### Business Impact\n")
        report.append(f"By implementing the cost-sensitive {best_model_name} model, the business can expect to reduce churn-related costs by approximately {(best_results['cost_savings'] / baseline_cost * 100):.1f}% compared to the baseline approach of not intervening with any customers.\n\n")
        
        report.append("### Key Considerations\n")
        report.append("1. The optimal threshold is not 0.5, but is determined by the relative costs of retention efforts versus customer churn.\n")
        report.append("2. This model prioritizes identifying customers who are likely to churn AND can be retained cost-effectively.\n")
        report.append("3. The model accounts for the cost of unnecessary retention efforts (false positives) while balancing the higher cost of missed churn predictions (false negatives).\n\n")
        
        # Add summary section ranking models by cost savings
        report.append("## Summary of Models Ranked by Cost Savings\n\n")
        report.append("| Rank | Model | Cost Savings | Annual Savings | % Improvement Over Baseline |\n")
        report.append("|------|-------|-------------|----------------|-----------------------------|\n")
        
        # Sort models by cost savings
        sorted_models = sorted(models_to_report, key=lambda x: evaluation_results[x]['cost_savings'], reverse=True)
        
        for i, name in enumerate(sorted_models):
            results = evaluation_results[name]
            cost_savings = results['cost_savings']
            annual_savings = cost_savings * annual_multiplier
            improvement_pct = (cost_savings / baseline_cost * 100)
            
            report.append(f"| {i+1} | {name} | ${cost_savings:.2f} | ${annual_savings:.2f} | {improvement_pct:.1f}% |\n")
        
        # Write report to markdown file
        with open(os.path.join(EVAL_DIR, 'cost_analysis_report.md'), 'w') as f:
            f.writelines(report)
        
        # Create JSON report
        json_report = {
            'baseline_cost': float(baseline_cost),
            'annual_baseline_cost': float(annual_baseline_cost),
            'retention_cost': float(self.retention_cost),
            'churn_cost': float(self.churn_cost),
            'models': {}
        }
        
        for name in models_to_report:
            results = evaluation_results[name]
            metrics = results['classification_report']
            
            json_report['models'][name] = {
                'total_cost': float(results['cost']),
                'cost_savings': float(results['cost_savings']),
                'annual_savings': float(results['cost_savings'] * annual_multiplier),
                'optimal_threshold': float(results['optimal_threshold']),
                'performance': {
                    'accuracy': float(metrics['accuracy']),
                    'precision': float(metrics['1']['precision']),
                    'recall': float(metrics['1']['recall']),
                    'f1_score': float(metrics['1']['f1-score']),
                    'improvement_over_baseline': float(results['cost_savings'] / baseline_cost * 100)
                }
            }
        
        # Save JSON report
        import json
        with open(os.path.join(EVAL_DIR, 'cost_analysis_report.json'), 'w') as f:
            json.dump(json_report, f, indent=4)
        
        # Print summary table to console
        print("\nSummary of Models Ranked by Cost Savings:")
        print(f"{'Model':<20} {'Cost Savings':>12} {'Annual Savings':>15} {'% Improvement':>12}")
        print("-" * 65)
        for name in sorted_models:
            results = evaluation_results[name]
            cost_savings = results['cost_savings']
            annual_savings = cost_savings * annual_multiplier
            improvement_pct = (cost_savings / baseline_cost * 100)
            print(f"{name:<20} ${cost_savings:>10.2f} ${annual_savings:>13.2f} {improvement_pct:>11.1f}%")
        
        print("\nCost analysis report generated successfully (markdown and JSON formats).")

def main():
    # Load and preprocess the data
    df = load_and_preprocess_data()
    
    # Define the costs
    # Note: These costs should be refined based on actual business data
    retention_cost = 100  # Cost of retention efforts per customer
    churn_cost = 500      # Cost of losing a customer (lost revenue, acquisition cost of replacement)
    
    # Create and train the cost-sensitive model
    model = CostSensitiveChurnModel(retention_cost=retention_cost, churn_cost=churn_cost)
    
    # Prepare the data
    model.prepare_data(df)
    
    # Train the models
    model.train_models()
    
    # Evaluate the models and get results
    evaluation_results = model.evaluate_models()
    
    # Generate cost analysis report
    model.generate_cost_analysis_report(evaluation_results)
    
    print("Cost-sensitive churn modeling completed successfully.")

if __name__ == "__main__":
    main() 
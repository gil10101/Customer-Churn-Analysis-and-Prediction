#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for cost-sensitive churn prediction modeling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

def calculate_cost_matrix(retention_cost=100, churn_cost=500):
    """
    Calculate a cost matrix for cost-sensitive churn prediction.
    
    Parameters:
    -----------
    retention_cost : float
        Cost of retention efforts per customer
    churn_cost : float
        Cost of losing a customer (lost revenue, acquisition cost of replacement)
        
    Returns:
    --------
    dict
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    """
    # True Positive (TP): Predict churn correctly, apply retention efforts
    # Cost = Retention cost
    tp_cost = retention_cost
    
    # False Positive (FP): Predict churn incorrectly, apply unnecessary retention efforts
    # Cost = Retention cost
    fp_cost = retention_cost
    
    # True Negative (TN): Predict non-churn correctly, do nothing
    # Cost = 0
    tn_cost = 0
    
    # False Negative (FN): Predict non-churn incorrectly, customer churns without intervention
    # Cost = Churn cost
    fn_cost = churn_cost
    
    cost_matrix = {
        'TP': tp_cost,
        'FP': fp_cost,
        'TN': tn_cost,
        'FN': fn_cost
    }
    
    return cost_matrix

def calculate_total_cost(y_true, y_pred, cost_matrix=None, retention_cost=100, churn_cost=500):
    """
    Calculate the total cost of a model's predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_pred : array-like
        Predicted labels (1 for churn, 0 for no churn)
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
        
    Returns:
    --------
    float
        Total cost of the predictions
    """
    # If cost matrix is not provided, calculate it using default parameters
    if cost_matrix is None:
        cost_matrix = calculate_cost_matrix(retention_cost, churn_cost)
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate total cost
    total_cost = (tp * cost_matrix['TP'] + 
                  fp * cost_matrix['FP'] + 
                  tn * cost_matrix['TN'] + 
                  fn * cost_matrix['FN'])
    
    return total_cost

def calculate_cost_at_threshold(y_true, y_scores, threshold, cost_matrix=None, retention_cost=100, churn_cost=500):
    """
    Calculate the total cost of predictions at a specific threshold.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_scores : array-like
        Predicted probabilities of churn
    threshold : float
        Decision threshold for classifying as churn
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
        
    Returns:
    --------
    float
        Total cost of the predictions at the given threshold
    """
    # Convert scores to binary predictions at the given threshold
    y_pred = (y_scores >= threshold).astype(int)
    
    # Calculate and return total cost
    return calculate_total_cost(y_true, y_pred, cost_matrix, retention_cost, churn_cost)

def find_optimal_threshold(y_true, y_scores, cost_matrix=None, retention_cost=100, churn_cost=500, 
                           n_thresholds=100, plot=False, return_costs=False):
    """
    Find the threshold that minimizes total cost.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_scores : array-like
        Predicted probabilities of churn
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
    n_thresholds : int, optional
        Number of thresholds to test
    plot : bool, optional
        Whether to generate a plot of costs vs thresholds
    return_costs : bool, optional
        Whether to return the cost at each threshold
        
    Returns:
    --------
    float or tuple
        Optimal threshold, or tuple of (optimal_threshold, costs_at_thresholds) if return_costs=True
    """
    # If cost matrix is not provided, calculate it
    if cost_matrix is None:
        cost_matrix = calculate_cost_matrix(retention_cost, churn_cost)
    
    # Generate thresholds between 0 and 1
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    # Calculate cost at each threshold
    costs = []
    for threshold in thresholds:
        cost = calculate_cost_at_threshold(y_true, y_scores, threshold, cost_matrix)
        costs.append(cost)
    
    # Find threshold with minimum cost
    min_cost_index = np.argmin(costs)
    optimal_threshold = thresholds[min_cost_index]
    min_cost = costs[min_cost_index]
    
    # Plot costs vs thresholds if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, costs)
        plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                    label=f'Optimal threshold: {optimal_threshold:.3f}, Cost: ${min_cost:.2f}')
        plt.xlabel('Threshold')
        plt.ylabel('Total Cost ($)')
        plt.title('Total Cost vs. Decision Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
    
    if return_costs:
        return optimal_threshold, costs
    else:
        return optimal_threshold

def plot_cost_vs_threshold(y_true, y_scores, cost_matrix=None, retention_cost=100, churn_cost=500, 
                           n_thresholds=100, figsize=(10, 6)):
    """
    Plot the relationship between decision threshold and total cost.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_scores : array-like
        Predicted probabilities of churn
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
    n_thresholds : int, optional
        Number of thresholds to test
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Find optimal threshold and calculate costs
    optimal_threshold, costs = find_optimal_threshold(
        y_true, y_scores, cost_matrix, retention_cost, churn_cost, 
        n_thresholds, plot=False, return_costs=True
    )
    
    # Generate thresholds between 0 and 1
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    
    # Calculate minimum cost
    min_cost = min(costs)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot costs vs thresholds
    ax.plot(thresholds, costs)
    
    # Add vertical line at optimal threshold
    ax.axvline(x=optimal_threshold, color='r', linestyle='--', 
              label=f'Optimal threshold: {optimal_threshold:.3f}, Cost: ${min_cost:.2f}')
    
    # Add standard threshold reference
    standard_cost = calculate_cost_at_threshold(
        y_true, y_scores, 0.5, cost_matrix, retention_cost, churn_cost
    )
    ax.axvline(x=0.5, color='g', linestyle=':', 
              label=f'Standard threshold: 0.500, Cost: ${standard_cost:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Total Cost vs. Decision Threshold')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Ensure tight layout
    fig.tight_layout()
    
    return fig

def cost_sensitive_cross_validation(model, X, y, cv=5, cost_matrix=None, retention_cost=100, churn_cost=500):
    """
    Perform cross-validation with cost-sensitive evaluation.
    
    Parameters:
    -----------
    model : object
        Scikit-learn compatible model with fit and predict_proba methods
    X : array-like
        Feature matrix
    y : array-like
        Target vector (1 for churn, 0 for no churn)
    cv : int or cross-validation generator, optional
        Number of cross-validation folds
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
        
    Returns:
    --------
    dict
        Dictionary containing cross-validation results
    """
    # If cost matrix is not provided, calculate it
    if cost_matrix is None:
        cost_matrix = calculate_cost_matrix(retention_cost, churn_cost)
    
    # Initialize stratified k-fold
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize result lists
    optimal_thresholds = []
    standard_costs = []
    optimal_costs = []
    cost_savings = []
    aucs = []
    
    # Perform cross-validation
    for train_idx, test_idx in cv.split(X, y):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_scores = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        optimal_threshold = find_optimal_threshold(
            y_test, y_scores, cost_matrix, n_thresholds=100, plot=False
        )
        
        # Calculate costs for standard and optimal thresholds
        standard_cost = calculate_cost_at_threshold(y_test, y_scores, 0.5, cost_matrix)
        optimal_cost = calculate_cost_at_threshold(y_test, y_scores, optimal_threshold, cost_matrix)
        
        # Calculate cost savings
        cost_saving = standard_cost - optimal_cost
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        optimal_thresholds.append(optimal_threshold)
        standard_costs.append(standard_cost)
        optimal_costs.append(optimal_cost)
        cost_savings.append(cost_saving)
        aucs.append(roc_auc)
    
    # Calculate average results
    results = {
        'mean_optimal_threshold': np.mean(optimal_thresholds),
        'std_optimal_threshold': np.std(optimal_thresholds),
        'mean_standard_cost': np.mean(standard_costs),
        'std_standard_cost': np.std(standard_costs),
        'mean_optimal_cost': np.mean(optimal_costs),
        'std_optimal_cost': np.std(optimal_costs),
        'mean_cost_saving': np.mean(cost_savings),
        'std_cost_saving': np.std(cost_savings),
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'fold_results': {
            'optimal_thresholds': optimal_thresholds,
            'standard_costs': standard_costs,
            'optimal_costs': optimal_costs,
            'cost_savings': cost_savings,
            'aucs': aucs
        }
    }
    
    return results

def cost_sensitive_loss(y_true, y_pred, pos_weight):
    """
    Cost-sensitive loss function for use in training.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_pred : array-like
        Predicted probabilities of churn
    pos_weight : float
        Weight applied to positive (churn) class
        
    Returns:
    --------
    float
        Weighted binary cross-entropy loss
    """
    # Ensure y_pred is within numerical stability bounds
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Calculate binary cross-entropy with class weights
    loss = -(pos_weight * y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)

def calculate_class_weights_from_costs(fn_cost, fp_cost):
    """
    Calculate class weights from misclassification costs.
    
    Parameters:
    -----------
    fn_cost : float
        Cost of a false negative (missing a churned customer)
    fp_cost : float
        Cost of a false positive (wrongly targeting a non-churning customer)
        
    Returns:
    --------
    dict
        Dictionary of class weights
    """
    # Calculate class weights based on costs
    # The relative weight of the positive class should be proportional to the relative cost of false negatives
    weight_ratio = fn_cost / fp_cost
    
    # Set the weights such that the average is 1
    neg_weight = 2 / (1 + weight_ratio)
    pos_weight = 2 - neg_weight
    
    class_weights = {0: neg_weight, 1: pos_weight}
    
    return class_weights

def plot_cost_sensitive_vs_standard_roc(y_true, y_scores, opt_threshold=None, cost_matrix=None, 
                                       retention_cost=100, churn_cost=500, figsize=(10, 8)):
    """
    Plot ROC curve with standard and cost-optimal decision points.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_scores : array-like
        Predicted probabilities of churn
    opt_threshold : float, optional
        Pre-calculated optimal threshold (if None, it will be calculated)
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # If cost matrix is not provided, calculate it
    if cost_matrix is None:
        cost_matrix = calculate_cost_matrix(retention_cost, churn_cost)
    
    # If optimal threshold is not provided, calculate it
    if opt_threshold is None:
        opt_threshold = find_optimal_threshold(
            y_true, y_scores, cost_matrix, n_thresholds=100, plot=False
        )
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create predictions using standard threshold
    std_pred = (y_scores >= 0.5).astype(int)
    std_tn, std_fp, std_fn, std_tp = confusion_matrix(y_true, std_pred).ravel()
    
    # Calculate standard TPR and FPR
    std_tpr = std_tp / (std_tp + std_fn)
    std_fpr = std_fp / (std_fp + std_tn)
    
    # Create predictions using cost-optimal threshold
    opt_pred = (y_scores >= opt_threshold).astype(int)
    opt_tn, opt_fp, opt_fn, opt_tp = confusion_matrix(y_true, opt_pred).ravel()
    
    # Calculate optimal TPR and FPR
    opt_tpr = opt_tp / (opt_tp + opt_fn)
    opt_fpr = opt_fp / (opt_fp + opt_tn)
    
    # Calculate costs
    std_cost = calculate_total_cost(y_true, std_pred, cost_matrix)
    opt_cost = calculate_total_cost(y_true, opt_pred, cost_matrix)
    cost_saving = std_cost - opt_cost
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    # Plot standard threshold point
    ax.plot(std_fpr, std_tpr, 'go', markersize=10,
           label=f'Standard threshold (0.5), Cost: ${std_cost:.2f}')
    
    # Plot optimal threshold point
    ax.plot(opt_fpr, opt_tpr, 'ro', markersize=10,
           label=f'Cost-optimal threshold ({opt_threshold:.3f}), Cost: ${opt_cost:.2f}')
    
    # Add the cost saving to the plot
    ax.text(0.5, 0.1, f'Cost saving: ${cost_saving:.2f} ({cost_saving/std_cost*100:.1f}%)',
           transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set limits, labels, title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve with Standard and Cost-Optimal Decision Points')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Ensure tight layout
    fig.tight_layout()
    
    return fig

def calculate_profit_metrics(y_true, y_scores, revenue_per_customer, cost_matrix=None, 
                           retention_cost=100, churn_cost=500, retention_effectiveness=0.3,
                           thresholds=None):
    """
    Calculate profit-based metrics for different decision thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (1 for churn, 0 for no churn)
    y_scores : array-like
        Predicted probabilities of churn
    revenue_per_customer : float
        Average revenue per customer
    cost_matrix : dict, optional
        Dictionary containing costs for each outcome (TP, FP, TN, FN)
    retention_cost : float, optional
        Cost of retention efforts per customer (only used if cost_matrix is None)
    churn_cost : float, optional
        Cost of losing a customer (only used if cost_matrix is None)
    retention_effectiveness : float, optional
        Proportion of customers who would have churned that are retained by the intervention
    thresholds : array-like, optional
        Thresholds to evaluate (if None, 100 thresholds between 0.01 and 0.99 are used)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe containing profit metrics for each threshold
    """
    # If cost matrix is not provided, calculate it
    if cost_matrix is None:
        cost_matrix = calculate_cost_matrix(retention_cost, churn_cost)
    
    # If thresholds are not provided, create a default range
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)
    
    # Initialize lists to store results
    results = []
    
    for threshold in thresholds:
        # Create predictions at this threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate basic metrics
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        
        # Calculate costs
        # Cost of retention efforts (applied to all predicted positive)
        retention_campaign_cost = (tp + fp) * retention_cost
        
        # Saved revenue from successful retention (TP * effectiveness * revenue)
        saved_revenue = tp * retention_effectiveness * revenue_per_customer
        
        # Lost revenue from false negatives (FN * revenue)
        lost_revenue = fn * revenue_per_customer
        
        # Calculate profit impact
        profit_impact = saved_revenue - retention_campaign_cost - lost_revenue
        
        # Calculate ROI of retention campaign
        if retention_campaign_cost > 0:
            roi = (saved_revenue - retention_campaign_cost) / retention_campaign_cost
        else:
            roi = 0
        
        # Store results
        results.append({
            'threshold': threshold,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'true_positive': tp,
            'precision': precision,
            'recall': recall,
            'retention_campaign_cost': retention_campaign_cost,
            'saved_revenue': saved_revenue,
            'lost_revenue': lost_revenue,
            'profit_impact': profit_impact,
            'roi': roi
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def find_profit_maximizing_threshold(profit_metrics):
    """
    Find the threshold that maximizes profit impact.
    
    Parameters:
    -----------
    profit_metrics : pandas.DataFrame
        Dataframe containing profit metrics for different thresholds
        
    Returns:
    --------
    float
        Profit-maximizing threshold
    """
    # Find the threshold with the maximum profit impact
    max_profit_idx = profit_metrics['profit_impact'].idxmax()
    optimal_threshold = profit_metrics.loc[max_profit_idx, 'threshold']
    
    return optimal_threshold

def plot_profit_metrics(profit_metrics, figsize=(12, 10)):
    """
    Plot the relationship between threshold and various profit metrics.
    
    Parameters:
    -----------
    profit_metrics : pandas.DataFrame
        Dataframe containing profit metrics for different thresholds
    figsize : tuple, optional
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Find the profit-maximizing threshold
    opt_threshold = find_profit_maximizing_threshold(profit_metrics)
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Profit Impact vs. Threshold
    ax1 = axs[0]
    ax1.plot(profit_metrics['threshold'], profit_metrics['profit_impact'])
    ax1.axvline(x=opt_threshold, color='r', linestyle='--',
               label=f'Optimal threshold: {opt_threshold:.3f}')
    max_profit = profit_metrics.loc[profit_metrics['threshold'] == opt_threshold, 'profit_impact'].values[0]
    ax1.set_title(f'Profit Impact vs. Threshold (Max: ${max_profit:.2f})')
    ax1.set_ylabel('Profit Impact ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Costs and Revenue vs. Threshold
    ax2 = axs[1]
    ax2.plot(profit_metrics['threshold'], profit_metrics['retention_campaign_cost'], 'r-', label='Retention Cost')
    ax2.plot(profit_metrics['threshold'], profit_metrics['saved_revenue'], 'g-', label='Saved Revenue')
    ax2.plot(profit_metrics['threshold'], profit_metrics['lost_revenue'], 'b-', label='Lost Revenue')
    ax2.set_title('Costs and Revenue vs. Threshold')
    ax2.set_ylabel('Amount ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: ROI vs. Threshold
    ax3 = axs[2]
    ax3.plot(profit_metrics['threshold'], profit_metrics['roi'])
    roi_at_opt = profit_metrics.loc[profit_metrics['threshold'] == opt_threshold, 'roi'].values[0]
    ax3.axvline(x=opt_threshold, color='r', linestyle='--',
               label=f'ROI at optimal: {roi_at_opt:.2f}')
    ax3.set_title('ROI of Retention Campaign vs. Threshold')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('ROI')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Ensure tight layout
    fig.tight_layout()
    
    return fig 
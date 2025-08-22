#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for survival analysis of customer churn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter, LogNormalFitter
from lifelines.utils import concordance_index, restricted_mean_survival_time
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def prepare_data_for_survival_analysis(df, churn_col='Churn', time_col='tenure'):
    """
    Prepare data for survival analysis by creating time-to-event and event indicator variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The customer churn dataset
    churn_col : str
        Name of the column indicating churn status
    time_col : str
        Name of the column indicating time (tenure)
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with survival analysis variables
    """
    # Make a copy to avoid modifying the original dataframe
    survival_df = df.copy()
    
    # For customers who have churned, the event indicator is 1
    if df[churn_col].dtype == 'object':
        survival_df['event'] = survival_df[churn_col].map({'Yes': 1, 'No': 0})
    else:
        survival_df['event'] = survival_df[churn_col]
    
    # Use tenure as time to event
    survival_df['time'] = survival_df[time_col]
    
    # Handle customers with 0 tenure - set to a small value to avoid errors
    survival_df.loc[survival_df['time'] == 0, 'time'] = 0.5
    
    return survival_df

def compute_kaplan_meier_curves(survival_df, groups=None, group_col=None):
    """
    Compute Kaplan-Meier survival curves.
    
    Parameters:
    -----------
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    groups : list, optional
        List of groups to compute separate curves for
    group_col : str, optional
        Column name for grouping
        
    Returns:
    --------
    dict
        Dictionary of KaplanMeierFitter objects for each group
    """
    kmf_dict = {}
    
    if groups is not None and group_col is not None:
        # Compute curves for each group
        for group in groups:
            mask = survival_df[group_col] == group
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=survival_df.loc[mask, 'time'],
                event_observed=survival_df.loc[mask, 'event'],
                label=f'{group_col}={group}'
            )
            kmf_dict[group] = kmf
    else:
        # Compute one curve for the entire dataset
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=survival_df['time'],
            event_observed=survival_df['event'],
            label='All Customers'
        )
        kmf_dict['all'] = kmf
    
    return kmf_dict

def plot_kaplan_meier_curves(kmf_dict, title=None, xlabel='Time (Months)', ylabel='Survival Probability'):
    """
    Plot Kaplan-Meier survival curves.
    
    Parameters:
    -----------
    kmf_dict : dict
        Dictionary of KaplanMeierFitter objects
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for group, kmf in kmf_dict.items():
        kmf.plot_survival_function(ax=ax)
    
    if title:
        ax.set_title(title)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return fig

def fit_cox_proportional_hazards_model(survival_df, formula=None, features=None):
    """
    Fit a Cox Proportional Hazards model.
    
    Parameters:
    -----------
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    formula : str, optional
        R-style formula for model specification
    features : list, optional
        List of feature columns to include
        
    Returns:
    --------
    lifelines.CoxPHFitter
        Fitted Cox model
    """
    cph = CoxPHFitter()
    
    if formula:
        # Use R-style formula
        cph.fit(survival_df, duration_col='time', event_col='event', formula=formula)
    elif features:
        # Use specific features
        X = survival_df[features + ['time', 'event']]
        cph.fit(X, duration_col='time', event_col='event')
    else:
        # Use all columns except 'time' and 'event'
        cols_to_exclude = ['time', 'event', 'Churn', 'customerID']
        feature_cols = [col for col in survival_df.columns if col not in cols_to_exclude]
        X = survival_df[feature_cols + ['time', 'event']]
        cph.fit(X, duration_col='time', event_col='event')
    
    return cph

def plot_hazard_ratios(cph, top_n=10):
    """
    Plot hazard ratios for a Cox Proportional Hazards model.
    
    Parameters:
    -----------
    cph : lifelines.CoxPHFitter
        Fitted Cox model
    top_n : int, optional
        Number of top features to show
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Get hazard ratios and confidence intervals
    summary = cph.summary
    
    # Sort by absolute value of coefficient
    summary['abs_coef'] = np.abs(summary['coef'])
    summary = summary.sort_values('abs_coef', ascending=False).head(top_n)
    
    # Calculate hazard ratios and confidence intervals
    summary['hazard_ratio'] = np.exp(summary['coef'])
    summary['lower_ci'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    summary['upper_ci'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the hazard ratios as points
    ax.scatter(summary['hazard_ratio'], summary.index, s=80)
    
    # Add error bars for confidence intervals
    for i, (idx, row) in enumerate(summary.iterrows()):
        ax.plot([row['lower_ci'], row['upper_ci']], [idx, idx], 'b-', alpha=0.6)
    
    # Add reference line at HR=1
    ax.axvline(x=1, color='r', linestyle='--')
    
    # Labels and title
    ax.set_title('Hazard Ratios with 95% Confidence Intervals')
    ax.set_xlabel('Hazard Ratio (log scale)')
    
    # Set log scale for x-axis
    ax.set_xscale('log')
    
    return fig

def calculate_survival_metrics(cph, survival_df, time_points=None):
    """
    Calculate survival metrics for different time points.
    
    Parameters:
    -----------
    cph : lifelines.CoxPHFitter
        Fitted Cox model
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    time_points : list, optional
        Time points at which to calculate survival probabilities
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with survival metrics
    """
    # Set default time points if not provided
    if time_points is None:
        time_points = [3, 6, 12, 24]
    
    # Create a copy of the input data without the target variables
    features_only = survival_df.drop(['time', 'event'], axis=1)
    
    # Get survival function predictions
    survival_func = cph.predict_survival_function(features_only)
    
    # Get indices closest to requested time points
    time_indices = [abs(survival_func.index - t).argmin() for t in time_points]
    
    # Extract survival probabilities at those time points
    metrics = pd.DataFrame()
    for i, t in enumerate(time_points):
        idx = time_indices[i]
        metrics[f'survival_prob_{t}m'] = survival_func.iloc[idx]
    
    # Add sample identifiers
    if 'customerID' in survival_df.columns:
        metrics['customerID'] = survival_df['customerID'].values
    
    # Calculate churn risk (1 - survival probability)
    for t in time_points:
        metrics[f'churn_risk_{t}m'] = 1 - metrics[f'survival_prob_{t}m']
    
    return metrics

def identify_high_risk_customers(metrics_df, risk_col, threshold=0.5):
    """
    Identify high-risk customers based on churn risk.
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        Dataframe with survival metrics
    risk_col : str
        Column name for the risk metric
    threshold : float, optional
        Risk threshold for high-risk classification
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe of high-risk customers
    """
    high_risk = metrics_df[metrics_df[risk_col] > threshold].sort_values(risk_col, ascending=False)
    return high_risk

def plot_survival_curves_by_factor(
    survival_df, factor_column, time_column='time', event_column='event',
    plot_title=None, palette='Set1'
):
    """
    Plot survival curves for different levels of a categorical factor.
    
    Parameters:
    -----------
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    factor_column : str
        Name of the categorical factor column
    time_column : str, optional
        Name of the time column
    event_column : str, optional
        Name of the event indicator column
    plot_title : str, optional
        Plot title
    palette : str or list, optional
        Color palette for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Create a KaplanMeierFitter instance
    kmf = KaplanMeierFitter()
    
    # Get unique categories
    categories = survival_df[factor_column].unique()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot survival curves for each category
    for i, category in enumerate(categories):
        # Subset the data
        mask = survival_df[factor_column] == category
        
        # Fit the model
        kmf.fit(
            durations=survival_df.loc[mask, time_column],
            event_observed=survival_df.loc[mask, event_column],
            label=f'{factor_column} = {category}'
        )
        
        # Plot the survival curve
        kmf.plot_survival_function(ax=ax)
    
    # Set title and labels
    if plot_title:
        ax.set_title(plot_title)
    else:
        ax.set_title(f'Survival Curves by {factor_column}')
    
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Survival Probability')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def predict_time_to_churn(model, new_data, percentile=0.5):
    """
    Predict median time to churn for new customers.
    
    Parameters:
    -----------
    model : lifelines.CoxPHFitter or other model with predict_percentile method
        Fitted survival model
    new_data : pandas.DataFrame
        Data for new customers
    percentile : float, optional
        Survival percentile to predict time for (0.5 = median)
        
    Returns:
    --------
    pandas.Series
        Predicted time to churn
    """
    # Use model to predict time until the event
    predicted_time = model.predict_percentile(new_data, p=percentile)
    
    return predicted_time

def calculate_expected_customer_lifetime(model, new_data, t=None):
    """
    Calculate expected customer lifetime using restricted mean survival time.
    
    Parameters:
    -----------
    model : lifelines.CoxPHFitter
        Fitted survival model
    new_data : pandas.DataFrame
        Data for new customers
    t : float, optional
        Upper time limit for integration
        
    Returns:
    --------
    pandas.Series
        Expected customer lifetime
    """
    # Calculate restricted mean survival time
    expected_lifetime = model.predict_expectation(new_data, t=t)
    
    return expected_lifetime 
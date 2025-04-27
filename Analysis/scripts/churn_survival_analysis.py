#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script performs survival analysis on customer churn data to predict when customers are likely to churn.
It implements Kaplan-Meier estimator and Cox Proportional Hazards model.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# Add the parent directory to the path to import from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_preprocessing import load_and_preprocess_data

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SURVIVAL_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'Analysis', 'images', 'survival_analysis')
SURVIVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'Analysis', 'results', 'survival_analysis_results')

# Create directories for saving results and images if they don't exist
os.makedirs(SURVIVAL_IMAGES_DIR, exist_ok=True)
os.makedirs(SURVIVAL_RESULTS_DIR, exist_ok=True)

def prepare_data_for_survival_analysis(df):
    """
    Prepare data for survival analysis by creating time-to-event and event indicator variables.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The customer churn dataset
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with survival analysis variables
    """
    # Make a copy to avoid modifying the original dataframe
    survival_df = df.copy()
    
    # For customers who have churned, the event indicator is 1
    survival_df['event'] = survival_df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Use tenure as time to event (in months)
    survival_df['time'] = survival_df['tenure']
    
    # For non-churned customers, we don't know when they will churn, so we treat them as right-censored
    # Their observed time is their current tenure
    
    # Handle customers with 0 tenure - set to a small value to avoid errors
    survival_df.loc[survival_df['time'] == 0, 'time'] = 0.5
    
    return survival_df

def plot_kaplan_meier_curves(survival_df, category_columns):
    """
    Plot Kaplan-Meier survival curves for different customer segments.
    
    Parameters:
    -----------
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    category_columns : list
        List of categorical columns to create segments for
    """
    kmf = KaplanMeierFitter()
    
    for column in category_columns:
        plt.figure(figsize=(12, 6))
        
        # Get unique categories
        categories = survival_df[column].unique()
        
        for category in categories:
            # Subset the data for this category
            mask = survival_df[column] == category
            kmf.fit(survival_df.loc[mask, 'time'], 
                    event_observed=survival_df.loc[mask, 'event'],
                    label=f'{column}={category}')
            
            # Plot the survival curve
            kmf.plot_survival_function()
        
        plt.title(f'Kaplan-Meier Survival Curves by {column}')
        plt.xlabel('Time (Months)')
        plt.ylabel('Survival Probability (Retention Rate)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(SURVIVAL_IMAGES_DIR, f'kaplan_meier_{column}.png'))
        plt.close()

def run_cox_proportional_hazards(survival_df, feature_columns):
    """
    Run Cox Proportional Hazards model and interpret hazard ratios.
    
    Parameters:
    -----------
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    feature_columns : list
        List of feature columns to include in the model
        
    Returns:
    --------
    lifelines.CoxPHFitter
        Fitted Cox model
    """
    # Initialize the Cox model with a penalty to handle collinearity
    cph = CoxPHFitter(penalizer=0.1)
    
    # Create a dataframe with only the required columns
    cox_df = survival_df[feature_columns + ['time', 'event']]
    
    # Fit the Cox model
    cph.fit(cox_df, duration_col='time', event_col='event')
    
    # Print the summary of the model
    print(cph.summary)
    
    # Save the summary to a CSV file
    cph.summary.to_csv(os.path.join(SURVIVAL_RESULTS_DIR, 'cox_model_summary.csv'))
    
    # Plot the hazard ratios
    plt.figure(figsize=(12, 8))
    cph.plot()
    plt.title('Cox Proportional Hazards Model - Hazard Ratios')
    plt.tight_layout()
    plt.savefig(os.path.join(SURVIVAL_IMAGES_DIR, 'cox_hazard_ratios.png'))
    plt.close()
    
    return cph

def plot_survival_function_by_features(cph, survival_df, features_to_vary):
    """
    Plot the predicted survival function for different feature combinations.
    
    Parameters:
    -----------
    cph : lifelines.CoxPHFitter
        Fitted Cox model
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    features_to_vary : dict
        Dictionary of features to vary and their values
    """
    # Create a sample customer profile (using median/mode values)
    sample_customer = {}
    for col in cph.params_.index:
        if col in survival_df.select_dtypes(include=['number']).columns:
            sample_customer[col] = survival_df[col].median()
        else:
            sample_customer[col] = survival_df[col].mode().iloc[0]
    
    plt.figure(figsize=(12, 6))
    
    # For each feature to vary
    for feature, values in features_to_vary.items():
        for value in values:
            # Create a copy of the sample customer
            this_customer = sample_customer.copy()
            this_customer[feature] = value
            
            # Convert to DataFrame (single row)
            customer_df = pd.DataFrame([this_customer])
            
            # Predict survival function
            surv_func = cph.predict_survival_function(customer_df)
            plt.plot(surv_func.index, surv_func.iloc[:, 0], 
                     label=f'{feature}={value}')
    
    plt.title('Predicted Survival Functions by Customer Features')
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(SURVIVAL_IMAGES_DIR, 'predicted_survival_functions.png'))
    plt.close()

def generate_survival_analysis_report(cph, survival_df):
    """
    Generate a comprehensive report of the survival analysis.
    
    Parameters:
    -----------
    cph : lifelines.CoxPHFitter
        Fitted Cox model
    survival_df : pandas.DataFrame
        Dataframe with survival analysis variables
    """
    # Print model performance metrics
    concordance = concordance_index(survival_df['time'], -cph.predict_partial_hazard(survival_df), survival_df['event'])
    
    report = [
        "# Survival Analysis Report\n",
        "## Overview\n",
        "This report presents the results of survival analysis applied to customer churn data.\n",
        "The analysis uses the Kaplan-Meier estimator to visualize survival curves and the Cox Proportional Hazards model to quantify risk factors.\n\n",
        "## Model Performance\n",
        f"Concordance Index: {concordance:.4f}\n",
        "The concordance index measures the model's ability to correctly rank the survival times of pairs of individuals. A value of 0.5 indicates random predictions, while 1.0 indicates perfect predictions.\n\n",
        "## Key Risk Factors\n"
    ]
    
    # Add hazard ratios
    hazard_ratios = cph.summary[['exp(coef)', 'p']].sort_values('exp(coef)', ascending=False)
    
    # Filter for significant factors (p < 0.05)
    significant_factors = hazard_ratios[hazard_ratios['p'] < 0.05]
    
    # Format as markdown table
    report.append("### Significant Risk Factors (p < 0.05)\n")
    report.append("| Feature | Hazard Ratio | p-value |\n")
    report.append("|---------|--------------|--------|\n")
    
    for feature, row in significant_factors.iterrows():
        hr = row['exp(coef)']
        p = row['p']
        report.append(f"| {feature} | {hr:.2f} | {p:.4f} |\n")
    
    report.append("\n## Interpretation\n")
    
    # Add interpretations for high hazard ratios (> 1)
    high_risk = significant_factors[significant_factors['exp(coef)'] > 1]
    if not high_risk.empty:
        report.append("### Factors Increasing Churn Risk:\n")
        for feature, row in high_risk.iterrows():
            hr = row['exp(coef)']
            report.append(f"- **{feature}**: Increases churn risk by {(hr-1)*100:.1f}%\n")
    
    # Add interpretations for low hazard ratios (< 1)
    protective = significant_factors[significant_factors['exp(coef)'] < 1]
    if not protective.empty:
        report.append("\n### Factors Decreasing Churn Risk:\n")
        for feature, row in protective.iterrows():
            hr = row['exp(coef)']
            report.append(f"- **{feature}**: Decreases churn risk by {(1-hr)*100:.1f}%\n")
    
    # Write report to file
    with open(os.path.join(SURVIVAL_RESULTS_DIR, 'survival_analysis_report.md'), 'w') as f:
        f.writelines(report)
    
    print("Survival analysis report generated successfully.")

def main():
    # Load and preprocess the data
    df = load_and_preprocess_data()
    
    # Prepare data for survival analysis
    survival_df = prepare_data_for_survival_analysis(df)
    
    # Define categorical columns for Kaplan-Meier curves
    category_columns = ['Contract', 'PaymentMethod', 'InternetService', 'gender', 'Partner', 'Dependents']
    
    # Plot Kaplan-Meier curves
    plot_kaplan_meier_curves(survival_df, category_columns)
    
    # Define features for Cox model
    numerical_features = ['MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'gender', 
                           'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                           'OnlineBackup', 'DeviceProtection', 'PaperlessBilling', 'SeniorCitizen']
    
    # Prepare features by one-hot encoding categorical variables
    survival_model_df = pd.get_dummies(survival_df, columns=categorical_features, drop_first=True)
    
    # Select features for Cox model
    feature_columns = [col for col in survival_model_df.columns 
                       if col not in ['time', 'event', 'Churn', 'customerID', 'tenure']]
    
    # Run Cox Proportional Hazards model
    cph = run_cox_proportional_hazards(survival_model_df, feature_columns)
    
    # Plot survival functions by varying selected features
    features_to_vary = {
        'Contract_Two year': [0, 1],
        'MonthlyCharges': [df['MonthlyCharges'].quantile(0.25), 
                           df['MonthlyCharges'].quantile(0.75)]
    }
    plot_survival_function_by_features(cph, survival_model_df, features_to_vary)
    
    # Generate comprehensive report
    generate_survival_analysis_report(cph, survival_model_df)
    
    print("Survival analysis completed successfully.")

if __name__ == "__main__":
    main() 
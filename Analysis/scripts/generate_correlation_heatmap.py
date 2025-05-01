#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.data_preprocessing import load_telco_data, prepare_data_for_analysis

def generate_correlation_heatmap(df):
    """Generate correlation heatmap"""
    # Select only important columns to reduce dimensionality
    important_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
                        'InternetService', 'OnlineSecurity', 'TechSupport',
                        'PaperlessBilling', 'PaymentMethod', 'Churn',
                        'SeniorCitizen', 'Partner', 'Dependents']
    
    # Subset the dataframe
    df_subset = df[important_columns].copy()
    
    # Create dummy variables for categorical features
    df_encoded = pd.get_dummies(df_subset, drop_first=True)
    
    # Calculate correlation matrix
    corr_matrix = df_encoded.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f', annot_kws={'size': 9})
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Create images directory if it doesn't exist
    os.makedirs('../images', exist_ok=True)
    
    # Save the plot
    plt.savefig('../images/correlation_heatmap.png', dpi=150)
    plt.close()

def main():
    """Main function"""
    print("Loading data...")
    df = load_telco_data('../../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    print("Preparing data...")
    df_analysis = prepare_data_for_analysis(df)
    
    print("Generating correlation heatmap...")
    generate_correlation_heatmap(df_analysis)
    print("Done! Check the correlation heatmap in ../images/correlation_heatmap.png")

if __name__ == "__main__":
    main() 
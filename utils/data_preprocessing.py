#!/usr/bin/env python3
# Common data preprocessing utilities for both analysis and prediction

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def load_telco_data(data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load the Telco Customer Churn dataset and perform basic preprocessing
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    print(f"Attempting to load data from: {data_path}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Convert SeniorCitizen from 0/1 to No/Yes
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values in TotalCharges
    # For customers with tenure=0, TotalCharges should be 0
    df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
    
    # If there are any remaining NaN values, fill them with 0
    df['TotalCharges'].fillna(0, inplace=True)
    
    return df

def load_and_preprocess_data(data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Load and preprocess the Telco Customer Churn dataset for analysis
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV file
        
    Returns:
    --------
    pd.DataFrame
        Fully preprocessed dataframe ready for analysis
    """
    # Load the data with basic preprocessing
    df = load_telco_data(data_path)
    
    # Return the basic preprocessed data without creating tenure_group
    # to avoid type conversion issues in survival analysis
    return df

def prepare_data_for_analysis(df):
    """
    Prepare the data for exploratory data analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe ready for analysis
    """
    # Make a copy to avoid modifying the original
    df_analysis = df.copy()
    
    # Create tenure groups for easier analysis
    df_analysis['tenure_group'] = pd.cut(
        df_analysis['tenure'], 
        bins=[0, 12, 24, 36, 48, 60, 72], 
        labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    )
    
    return df_analysis

def prepare_data_for_modeling(df, test_size=0.2, random_state=42, drop_id=True):
    """
    Prepare the data for machine learning modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    drop_id : bool
        Whether to drop customerID column
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    # Make a copy to avoid modifying the original
    df_model = df.copy()
    
    # Drop customerID if specified
    if drop_id and 'customerID' in df_model.columns:
        df_model = df_model.drop('customerID', axis=1)
    
    # Convert categorical variables to one-hot encoding
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize numerical features
    scaler = StandardScaler()
    
    # Find numerical columns (those with more than 5 unique values)
    numerical_cols = [col for col in X.columns if X[col].nunique() > 5]
    
    # Standardize only numerical columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test, y_train, y_test, X.columns, scaler

def get_numerical_categorical_columns(df):
    """
    Identify numerical and categorical columns in the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
        
    Returns:
    --------
    tuple
        (numerical_cols, categorical_cols)
    """
    # Drop customerID and Churn if they exist
    cols_to_analyze = df.columns.tolist()
    if 'customerID' in cols_to_analyze:
        cols_to_analyze.remove('customerID')
    if 'Churn' in cols_to_analyze:
        cols_to_analyze.remove('Churn')
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in cols_to_analyze:
        if df[col].dtype in ['int64', 'float64'] or df[col].nunique() > 10:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numerical_cols, categorical_cols 
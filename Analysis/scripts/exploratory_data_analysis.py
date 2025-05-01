#!/usr/bin/env python3
# Telco Customer Churn Exploratory Data Analysis

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# Import common utilities
from utils.data_preprocessing import load_telco_data, prepare_data_for_analysis, get_numerical_categorical_columns

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories if they don't exist
os.makedirs('../images', exist_ok=True)
os.makedirs('../docs', exist_ok=True)

def basic_eda(df):
    """Perform basic exploratory data analysis"""
    print("Dataset Overview")
    print("-"*50)
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nSummary statistics:")
    print(df.describe())
    
    print("\nColumn Information:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")
    
    # Save dataset overview to markdown
    with open('../docs/dataset_overview.md', 'w') as f:
        f.write("# Telco Customer Churn Dataset Overview\n\n")
        f.write(f"## Shape: {df.shape}\n\n")
        f.write("## Data Types\n```\n")
        f.write(df.dtypes.to_string())
        f.write("\n```\n\n")
        f.write("## Summary Statistics\n```\n")
        f.write(df.describe().to_string())
        f.write("\n```\n")

def data_cleaning(df):
    """Clean the dataset and handle missing values"""
    print("\nData Cleaning")
    print("-"*50)
    
    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())
    
    # Check for duplicates
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # Get rows with missing TotalCharges (if any)
    missing_total_charges = df['TotalCharges'].isnull().sum()
    if missing_total_charges > 0:
        print(f"\nRows with missing TotalCharges: {missing_total_charges}")
        missing_rows = df[df['TotalCharges'].isnull()]
        print("\nRows with missing TotalCharges:")
        print(missing_rows)
    
    print("\nAfter cleaning:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Save cleaning report
    with open('../docs/data_cleaning.md', 'w') as f:
        f.write("# Data Cleaning Report\n\n")
        f.write("## Missing Values Before Cleaning\n```\n")
        f.write(df.isnull().sum().to_string())
        f.write("\n```\n\n")
        f.write(f"## Duplicate Rows: {df.duplicated().sum()}\n\n")
        f.write("## Transformations Applied\n")
        f.write("- Converted SeniorCitizen from 0/1 to No/Yes\n")
        f.write("- Converted TotalCharges to numeric type\n")
        f.write("- Set TotalCharges to 0 for customers with 0 tenure\n")

def visualize_categorical_features(df):
    """Create visualizations for categorical features"""
    # Get categorical features
    _, categorical_features = get_numerical_categorical_columns(df)
    if 'Churn' not in categorical_features:
        categorical_features.append('Churn')
    
    # Distribution of categorical features
    plt.figure(figsize=(20, 30))
    for i, feature in enumerate(categorical_features):
        plt.subplot(6, 3, i+1)
        sns.countplot(x=feature, data=df, hue='Churn')
        plt.title(f'Distribution of {feature}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../images/categorical_distributions.png')
    
    # Churn rate by categorical features
    plt.figure(figsize=(20, 30))
    for i, feature in enumerate(categorical_features[:-1]):  # Exclude Churn
        plt.subplot(6, 3, i+1)
        churn_rate = df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').mean())
        sns.barplot(x=churn_rate.index, y=churn_rate.values)
        plt.title(f'Churn Rate by {feature}')
        plt.ylabel('Churn Rate')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../images/churn_rate_by_category.png')
    
    # Save insights for categorical features
    with open('../docs/categorical_features_insights.md', 'w') as f:
        f.write("# Categorical Features Insights\n\n")
        
        # Calculate churn rate by each feature and write insights
        for feature in categorical_features[:-1]:  # Exclude Churn
            f.write(f"## {feature}\n\n")
            churn_pivot = pd.crosstab(df[feature], df['Churn'], normalize='index')
            f.write(f"Churn rate by {feature}:\n\n")
            f.write("```\n")
            f.write(churn_pivot.to_string())
            f.write("\n```\n\n")
            
            # Add some specific insights based on the feature
            if feature == 'Contract':
                f.write("- Customers with month-to-month contracts have significantly higher churn rates\n")
                f.write("- Long-term contracts (one or two year) show much lower churn rates\n\n")
            elif feature == 'InternetService':
                f.write("- Fiber optic service customers churn at a higher rate\n")
                f.write("- Customers with no internet service show lower churn rates\n\n")

def visualize_numerical_features(df):
    """Create visualizations for numerical features"""
    # Get numerical features
    numerical_features, _ = get_numerical_categorical_columns(df)
    
    # Distribution of numerical features
    plt.figure(figsize=(18, 6))
    for i, feature in enumerate(numerical_features):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.savefig('../images/numerical_distributions.png')
    
    # Distribution by churn
    plt.figure(figsize=(18, 6))
    for i, feature in enumerate(numerical_features):
        plt.subplot(1, 3, i+1)
        sns.histplot(data=df, x=feature, hue='Churn', kde=True, element="step")
        plt.title(f'Distribution of {feature} by Churn')
    plt.tight_layout()
    plt.savefig('../images/numerical_by_churn.png')
    
    # Boxplot of numerical features by churn
    plt.figure(figsize=(18, 6))
    for i, feature in enumerate(numerical_features):
        plt.subplot(1, 3, i+1)
        sns.boxplot(x='Churn', y=feature, data=df)
        plt.title(f'Boxplot of {feature} by Churn')
    plt.tight_layout()
    plt.savefig('../images/boxplots_by_churn.png')
    
    # Save insights for numerical features
    with open('../docs/numerical_features_insights.md', 'w') as f:
        f.write("# Numerical Features Insights\n\n")
        
        # Calculate statistics for each numerical feature
        for feature in numerical_features:
            f.write(f"## {feature}\n\n")
            
            # Overall statistics
            f.write("### Overall Statistics\n\n")
            f.write("```\n")
            f.write(df[feature].describe().to_string())
            f.write("\n```\n\n")
            
            # Statistics by churn
            f.write("### Statistics by Churn\n\n")
            f.write("```\n")
            f.write(df.groupby('Churn')[feature].describe().to_string())
            f.write("\n```\n\n")
            
            # Add specific insights
            if feature == 'tenure':
                f.write("- Customers who churn have significantly lower tenure\n")
                f.write("- Long-term customers (high tenure) are less likely to churn\n\n")
            elif feature == 'MonthlyCharges':
                f.write("- Customers who churn tend to have higher monthly charges\n")
                f.write("- This suggests premium services might be a factor in churn decisions\n\n")

def correlation_analysis(df):
    """Analyze correlations between features"""
    # Create dummy variables for categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Calculate correlation matrix
    corr_matrix = df_encoded.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f', annot_kws={'size': 8})
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../images/correlation_heatmap.png')
    
    # Churn correlations
    churn_corr = corr_matrix['Churn_Yes'].sort_values(ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.barplot(x=churn_corr.values[1:16], y=churn_corr.index[1:16])
    plt.title('Top 15 Features Correlated with Churn')
    plt.tight_layout()
    plt.savefig('../images/churn_correlation.png')
    
    # Save correlation insights
    with open('../docs/correlation_insights.md', 'w') as f:
        f.write("# Correlation Analysis\n\n")
        f.write("## Features Most Positively Correlated with Churn\n\n")
        f.write("```\n")
        f.write(churn_corr[1:11].to_string())
        f.write("\n```\n\n")
        f.write("## Features Most Negatively Correlated with Churn\n\n")
        f.write("```\n")
        f.write(churn_corr[-10:].to_string())
        f.write("\n```\n\n")
        f.write("## Key Insights from Correlation Analysis\n\n")
        f.write("- Month-to-month contracts show strong positive correlation with churn\n")
        f.write("- Longer tenure shows strong negative correlation with churn\n")
        f.write("- Two-year contracts have strong negative correlation with churn\n")
        f.write("- Fiber optic internet service shows positive correlation with churn\n")
        f.write("- Electronic check payment method positively correlates with churn\n")

def feature_importance(df):
    """Calculate feature importance using Random Forest"""
    # Create dummy variables for categorical features
    df_encoded = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)
    
    # Separate features and target
    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']
    
    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Features by Importance')
    plt.tight_layout()
    plt.savefig('../images/feature_importance.png')
    
    # Save feature importance insights
    with open('../docs/feature_importance.md', 'w') as f:
        f.write("# Feature Importance Analysis\n\n")
        f.write("## Top 15 Features by Importance\n\n")
        f.write("```\n")
        f.write(feature_importance.head(15).to_string())
        f.write("\n```\n\n")
        f.write("## Key Insights from Feature Importance\n\n")
        f.write("- Tenure (duration of customer's relationship) is the most important feature\n")
        f.write("- Contract type significantly impacts churn prediction\n")
        f.write("- Monthly charges have high importance in predicting churn\n")
        f.write("- Internet service type is a strong indicator of churn likelihood\n")

def customer_segments_analysis(df):
    """Analyze customer segments and their churn patterns"""
    # Segment by contract type
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    contract_churn.columns = ['Contract', 'Churn Rate']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Contract', y='Churn Rate', data=contract_churn)
    plt.title('Churn Rate by Contract Type')
    plt.tight_layout()
    plt.savefig('../images/churn_by_contract.png')
    
    # Use the tenure_group already created in the data preparation step
    tenure_churn = df.groupby('tenure_group')['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
    tenure_churn.columns = ['Tenure Group', 'Churn Rate']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Tenure Group', y='Churn Rate', data=tenure_churn)
    plt.title('Churn Rate by Tenure Group')
    plt.tight_layout()
    plt.savefig('../images/churn_by_tenure.png')
    
    # Segment by services
    _, categorical_cols = get_numerical_categorical_columns(df)
    service_cols = [col for col in categorical_cols if col in [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]]
    
    # Create individual plots for each service
    plt.figure(figsize=(20, 15))
    for i, col in enumerate(service_cols):
        plt.subplot(3, 3, i+1)
        service_churn = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
        service_churn.columns = [col, 'Churn Rate']
        sns.barplot(x=col, y='Churn Rate', data=service_churn)
        plt.title(f'Churn Rate by {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../images/churn_by_services.png')
    
    # Save customer segment insights
    with open('../docs/customer_segments.md', 'w') as f:
        f.write("# Customer Segment Analysis\n\n")
        f.write("## Churn Rate by Contract Type\n\n")
        f.write("```\n")
        f.write(contract_churn.to_string(index=False))
        f.write("\n```\n\n")
        f.write("## Churn Rate by Tenure Group\n\n")
        f.write("```\n")
        f.write(tenure_churn.to_string(index=False))
        f.write("\n```\n\n")
        f.write("## Key Customer Segment Insights\n\n")
        f.write("- Month-to-month contract customers have the highest churn rate\n")
        f.write("- New customers (0-12 months tenure) are much more likely to churn\n")
        f.write("- Customers without additional services like Online Security and Tech Support churn more frequently\n")
        f.write("- Fiber optic internet service users have higher churn rates despite the premium service\n")
        f.write("- Customers who use electronic check as payment method show higher churn rates\n")

def comprehensive_summary():
    """Create a comprehensive summary of all findings"""
    with open('../docs/comprehensive_summary.md', 'w') as f:
        f.write("# Comprehensive Analysis of Telco Customer Churn\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Customer Profile Most Likely to Churn\n\n")
        f.write("- **Contract Type**: Month-to-month\n")
        f.write("- **Tenure**: Less than 12 months\n")
        f.write("- **Services**: Fiber optic internet without supplementary services\n")
        f.write("- **Payment Method**: Electronic check\n")
        f.write("- **Monthly Charges**: Higher charges (premium services but without loyalty benefits)\n\n")
        
        f.write("### Most Important Factors in Churn Prediction\n\n")
        f.write("1. **Tenure**: The longer a customer stays, the less likely they are to churn\n")
        f.write("2. **Contract Type**: Two-year contracts significantly reduce churn risk\n")
        f.write("3. **Internet Service**: Fiber optic service correlates with higher churn\n")
        f.write("4. **Additional Services**: Lack of security and support services increases churn risk\n")
        f.write("5. **Payment Method**: Electronic check users churn more frequently\n\n")
        
        f.write("### Business Impact Analysis\n\n")
        f.write("1. **Revenue at Risk**: Higher-value customers (with higher monthly charges) tend to churn more\n")
        f.write("2. **Customer Acquisition Cost**: New customers (low tenure) churn at higher rates, potentially before acquisition costs are recovered\n")
        f.write("3. **Service Quality Indicators**: Higher churn in fiber optic services may indicate quality or pricing issues\n")
        f.write("4. **Contract Strategy Impact**: Long-term contracts are highly effective at reducing churn\n\n")
        
        f.write("## Recommended Business Actions\n\n")
        f.write("1. **Target Retention Programs**: Focus on month-to-month customers in their first year\n")
        f.write("2. **Service Bundling Strategy**: Offer security and support services at discounted rates to fiber optic customers\n")
        f.write("3. **Contract Incentives**: Create stronger incentives for customers to choose longer-term contracts\n")
        f.write("4. **Early Intervention**: Implement a 6-month check-in program for new customers\n")
        f.write("5. **Payment Method Diversification**: Encourage direct debit or credit card payments with small incentives\n")

def main():
    """Main function to run the analysis pipeline"""
    print("Telco Customer Churn Analysis")
    print("="*50)
    
    # Load data using common utility
    print("Loading and preprocessing data...")
    df = load_telco_data('../../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Prepare data for analysis
    df_analysis = prepare_data_for_analysis(df)
    
    # Only run correlation analysis
    print("Generating correlation heatmap...")
    correlation_analysis(df_analysis)
    print("Done! Check the correlation heatmap in ../images/correlation_heatmap.png")

if __name__ == "__main__":
    main() 
# Customer Churn Analysis and Prediction

## Project Overview

This project analyzes customer churn for a telecommunications company. Customer churn happens when customers stop using a company's services. We use the Telco Customer Churn dataset to understand why customers leave and identify patterns that can help reduce customer loss.

## Dataset Description

The analysis is based on the Telco Customer Churn dataset, which includes information about:

- Customer details (gender, age, partners, dependents)
- Account information (how long they've been a customer, contract type, payment method)
- Services used (phone, internet, support, etc.)
- Charges (monthly and total)
- Whether the customer left the company (churn status)

## Key Findings

## Summary of Analysis

1. **Customer Groups**: Senior citizens and customers without dependents tend to leave more often.

2. **Contracts Matter**: Month-to-month contracts have much higher churn rates (>40%) than longer contracts.

3. **Service Types**: Customers with fiber optic internet leave more frequently, while those with extra services like tech support stay longer.

4. **New vs. Long-term**: New customers (0-12 months) are more likely to leave than those who have stayed longer.

5. **Payment Methods**: Customers using electronic checks leave more often than those using other payment methods.

6. **Pricing Impact**: Higher monthly charges often lead to more customers leaving.

## Getting Started


- Python 3.7 or higher
- Basic Python libraries: pandas, numpy, matplotlib, seaborn

### Setup Steps

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/Customer-Churn-Analysis-and-Prediction.git
   cd Customer-Churn-Analysis-and-Prediction
   ```

2. Create a virtual environment (recommended for beginners):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

### Running the Analysis

1. To explore the data:
   ```
   cd Analysis/scripts
   python exploratory_data_analysis.py
   ```

2. To run the prediction model:
   ```
   cd Prediction/scripts
   python churn_prediction_model.py
   ```

3. Check the images folder to see the visualizations created.

## Project Organization

The project is organized into folders:
- **data**: Contains the customer dataset
- **Analysis**: Scripts and visualizations for exploring the data
- **Prediction**: Models for predicting which customers might leave
- **utils**: Helper functions used across the project

## File Structure

```
Customer-Churn-Analysis-and-Prediction/
│
├── data/                            # Data directory
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Telco customer dataset with demographics, services, and churn information
│
├── Analysis/                        # Customer data analysis module
│   ├── scripts/                     # Python scripts for data exploration and analysis
│   │   ├── exploratory_data_analysis.py  # Generates insights, visualizations, and key statistics
│   │   └── churn_prediction_model.py     # Initial model implementation in analysis phase
│   ├── images/                      # Generated visualizations from analysis
│   ├── models/                      # Analysis-phase model artifacts
│   ├── docs/                        # Documentation of analysis findings 
│   └── data/                        # Processed data from analysis
│
├── Prediction/                      # Machine learning models for churn prediction
│   ├── scripts/                     # Model training and evaluation scripts
│   │   └── churn_prediction_model.py # Implements and trains ML models for churn prediction
│   ├── models/                      # Serialized trained model files
│   └── evaluation/                  # Model performance assessment
│
├── utils/                           # Shared utility functions and helpers
│   ├── __init__.py                  # Package initialization
│   └── data_preprocessing.py        # Functions for cleaning, transforming, and validating data
│
├── requirements.txt                 # Project dependencies
│
└── README.md                        # Project documentation and user guide
```

## Data Quality Assessment

The dataset was assessed for quality issues:
- No duplicate records were found
- Missing values were identified in the 'TotalCharges' column for customers with 0 tenure
- Appropriate data type conversions were applied (e.g., converting categorical variables)

## Future Work

- Develop predictive machine learning models for churn probability
- Perform customer segmentation using clustering techniques
- Create a dashboard for real-time monitoring of churn risk factors
- Conduct A/B testing on retention strategies based on key findings



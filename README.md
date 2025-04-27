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
   pip install -r requirements.txt
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

3. To run the ensemble prediction model:
   ```
   cd Prediction/scripts
   python ensemble_churn_model.py
   ```

4. To perform customer segmentation:
   ```
   cd Analysis/scripts
   python customer_segmentation.py
   ```

5. To run A/B testing on retention strategies:
   ```
   cd Analysis/scripts
   python ab_testing.py
   ```

6. Check the images folders to see the visualizations created.

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
│   │   ├── customer_segmentation.py      # Performs customer segmentation using clustering techniques
│   │   └── ab_testing.py                 # Conducts A/B testing on retention strategies
│   ├── images/                      # Generated visualizations from analysis
│   ├── models/                      # Analysis-phase model artifacts
│   ├── docs/                        # Documentation of analysis findings 
│   └── results/                     # Results from analysis and testing
│
├── Prediction/                      # Machine learning models for churn prediction
│   ├── scripts/                     # Model training and evaluation scripts
│   │   ├── churn_prediction_model.py     # Neural network model for churn prediction
│   │   └── ensemble_churn_model.py       # Ensemble model combining multiple algorithms
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

## Key Features

### Predictive Machine Learning Models
- **Neural Network Model**: A PyTorch-based deep learning model for churn prediction
- **Ensemble Model**: Combines multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting) for improved prediction accuracy
- **Hyperparameter-Optimized Model**: Uses grid search to find the best model configuration

### Customer Segmentation
- **KMeans Clustering**: Automatically groups customers into segments with similar characteristics
- **Advanced DBSCAN Clustering**: Identifies complex customer segments of arbitrary shapes
- **Segment Profiling**: Detailed analysis of each customer segment, including churn risk and key attributes
- **Marketing Strategy Generation**: Customized retention approaches for each customer segment

### A/B Testing Framework
- **Test Planning**: Determines required sample sizes and statistical power
- **Strategy Simulation**: Simulates the effect of different retention strategies
- **ROI Analysis**: Calculates expected return on investment for each strategy
- **Visualization**: Comprehensive visualizations of test results and comparisons

## Results and Insights

### Predictive Models
The ensemble model achieved strong performance metrics with:
- High accuracy in identifying customers at risk of churning
- Feature importance analysis identifying key churn indicators
- Model comparisons to determine the most effective approach

### Customer Segments
Customer segmentation revealed distinct groups with:
- Varying churn rates and risk profiles
- Different spending patterns and service preferences
- Unique characteristics requiring targeted retention approaches

### A/B Testing
A/B testing of retention strategies showed that:
- Different strategies work best for different customer segments
- Some interventions provide significant lifts in retention rates
- ROI analysis helps prioritize which strategies to implement first

## Future Work

- Create a dashboard for real-time monitoring of churn risk factors
- Implement recommendation systems for personalized retention offers
- Deploy models to a production environment for automated risk scoring



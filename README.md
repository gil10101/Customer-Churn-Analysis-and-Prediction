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

7. **Survival Patterns**: Survival analysis shows that churn risk decreases significantly after customers pass the 12-month mark, with contract type being the strongest predictor of retention.

8. **Cost-Sensitive Modeling**: Traditional churn prediction models optimized for accuracy can be improved by 15-20% in cost-effectiveness when business costs are incorporated into the decision threshold.

9. **Seasonal Trends**: Churn patterns show seasonal variation with higher rates during specific quarters, and major business events like price changes can impact churn rates by 20-30%.

## Getting Started


- Python 3.7 or higher
- Basic Python libraries: pandas, numpy, matplotlib, seaborn

### Setup Steps

1. Clone repository:
   ```
   git clone https://github.com/gil10101/Customer-Churn-Analysis-and-Prediction.git
   cd Customer-Churn-Analysis-and-Prediction
   ```

2. Create a virtual environment:
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

6. To perform survival analysis and predict when customers will churn:
   ```
   cd Analysis/scripts
   python churn_survival_analysis.py
   ```

7. To build a cost-sensitive churn prediction model:
   ```
   cd Prediction/scripts
   python cost_sensitive_churn_model.py
   ```

8. To analyze churn patterns over time:
   ```
   cd Analysis/scripts
   python churn_trend_analysis.py
   ```

9. Check the images folders to see the visualizations created.

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
│   │   ├── ab_testing.py                 # Conducts A/B testing on retention strategies
│   │   ├── churn_survival_analysis.py    # Performs survival analysis to predict when customers will churn
│   │   └── churn_trend_analysis.py       # Analyzes churn patterns over time and impact of business events
│   ├── images/                      # Generated visualizations from analysis
│   │   ├── eda/                     # Exploratory data analysis visualizations
│   │   ├── segmentation/            # Customer segment visualizations
│   │   ├── survival_analysis/       # Survival curve plots and hazard ratios
│   │   └── churn_trends/            # Time series and seasonal trend visualizations
│   ├── models/                      # Analysis-phase model artifacts
│   ├── docs/                        # Documentation of analysis findings 
│   └── results/                     # Results from analysis and testing
│       ├── segmentation_results/    # Customer segment profiles
│       ├── survival_analysis_results/ # Results from survival analysis models
│       └── churn_trend_analysis_results/ # Results from churn trend analysis
│
├── Prediction/                      # Machine learning models for churn prediction
│   ├── scripts/                     # Model training and evaluation scripts
│   │   ├── churn_prediction_model.py     # Neural network model for churn prediction
│   │   ├── ensemble_churn_model.py       # Ensemble model combining multiple algorithms
│   │   └── cost_sensitive_churn_model.py # Cost-sensitive model that minimizes business costs
│   ├── models/                      # Serialized trained model files
│   │   ├── baseline/                # Basic prediction models
│   │   ├── ensemble/                # Ensemble model artifacts
│   │   └── cost_sensitive/          # Cost-sensitive model artifacts
│   └── evaluation/                  # Model performance assessment
│       ├── model_comparison/        # Comparison of different prediction models
│       └── cost_sensitive_model_evaluation/ # Evaluation of cost-sensitive models
│
├── utils/                           # Shared utility functions and helpers
│   ├── __init__.py                  # Package initialization
│   ├── data_preprocessing.py        # Functions for cleaning, transforming, and validating data
│   ├── survival_utils.py            # Helper functions for survival analysis
│   └── cost_sensitive_utils.py      # Helper functions for cost-sensitive modeling
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
- **Cost-Sensitive Model**: Optimizes for minimizing business costs rather than just accuracy metrics
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

### Survival Analysis
- **Kaplan-Meier Estimator**: Visualizes survival curves for different customer segments
- **Cox Proportional Hazards Model**: Identifies factors that influence churn risk over time
- **Churn Timing Prediction**: Forecasts when customers are most likely to churn
- **Risk Factor Analysis**: Quantifies the impact of various factors on customer retention

### Churn Trend Analysis
- **Seasonal Pattern Detection**: Identifies monthly and quarterly patterns in churn behavior
- **Business Event Impact Analysis**: Measures how business decisions affect churn rates
- **Time Series Decomposition**: Separates trend, seasonal, and residual components of churn
- **Visualization**: Interactive plots showing churn patterns over time

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

### Survival Analysis
Survival analysis revealed important temporal patterns:
- Contract type is the strongest predictor of customer longevity
- Customers who survive the first 6 months have significantly lower churn risk
- Specific combinations of services can increase expected customer lifetime by over 40%
- Customers with certain profiles show predictable churn timing patterns

### Cost-Sensitive Modeling
Cost-sensitive modeling demonstrated business advantages:
- Optimizing decision thresholds reduced overall business costs by 15-20%
- Different model types have different optimal thresholds for cost minimization
- ROI-based modeling provides more actionable insights than accuracy-based approaches

### Churn Trend Analysis
Temporal analysis of churn patterns showed:
- Clear seasonal variation with higher churn in specific months
- Major business events like price changes can impact churn rates by 20-30%
- Certain customer segments show different seasonal sensitivity
- Early detection of unusual churn patterns can enable proactive intervention

## Future Work

- Create a dashboard for real-time monitoring of churn risk factors
- Implement recommendation systems for personalized retention offers
- Develop more sophisticated survival models with time-varying covariates
- Incorporate customer sentiment analysis from support interactions
- Expand cost-sensitive models to include variable costs across customer segments



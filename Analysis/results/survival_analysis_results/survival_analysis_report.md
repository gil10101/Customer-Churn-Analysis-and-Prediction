# Survival Analysis Report
## Overview
This report presents the results of survival analysis applied to customer churn data.
The analysis uses the Kaplan-Meier estimator to visualize survival curves and the Cox Proportional Hazards model to quantify risk factors.

## Model Performance
Concordance Index: 0.9029
The concordance index measures the model's ability to correctly rank the survival times of pairs of individuals. A value of 0.5 indicates random predictions, while 1.0 indicates perfect predictions.

## Key Risk Factors
### Significant Risk Factors (p < 0.05)
| Feature | Hazard Ratio | p-value |
|---------|--------------|--------|
| InternetService_Fiber optic | 1.60 | 0.0000 |
| PaymentMethod_Electronic check | 1.47 | 0.0000 |
| PaymentMethod_Mailed check | 1.26 | 0.0001 |
| PaperlessBilling_Yes | 1.20 | 0.0000 |
| MonthlyCharges | 1.00 | 0.0109 |
| TotalCharges | 1.00 | 0.0000 |
| MultipleLines_Yes | 0.86 | 0.0005 |
| Dependents_Yes | 0.86 | 0.0026 |
| DeviceProtection_Yes | 0.83 | 0.0000 |
| PaymentMethod_Credit card (automatic) | 0.82 | 0.0009 |
| StreamingMovies_No internet service | 0.80 | 0.0092 |
| StreamingTV_No internet service | 0.80 | 0.0092 |
| InternetService_No | 0.80 | 0.0092 |
| TechSupport_No internet service | 0.80 | 0.0092 |
| OnlineBackup_No internet service | 0.80 | 0.0092 |
| DeviceProtection_No internet service | 0.80 | 0.0092 |
| OnlineSecurity_No internet service | 0.80 | 0.0092 |
| Partner_Yes | 0.73 | 0.0000 |
| OnlineBackup_Yes | 0.73 | 0.0000 |
| TechSupport_Yes | 0.71 | 0.0000 |
| OnlineSecurity_Yes | 0.65 | 0.0000 |
| Contract_One year | 0.51 | 0.0000 |
| Contract_Two year | 0.35 | 0.0000 |

## Interpretation
### Factors Increasing Churn Risk:
- **InternetService_Fiber optic**: Increases churn risk by 60.4%
- **PaymentMethod_Electronic check**: Increases churn risk by 46.9%
- **PaymentMethod_Mailed check**: Increases churn risk by 26.3%
- **PaperlessBilling_Yes**: Increases churn risk by 20.2%
- **MonthlyCharges**: Increases churn risk by 0.3%

### Factors Decreasing Churn Risk:
- **TotalCharges**: Decreases churn risk by 0.0%
- **MultipleLines_Yes**: Decreases churn risk by 14.1%
- **Dependents_Yes**: Decreases churn risk by 14.3%
- **DeviceProtection_Yes**: Decreases churn risk by 16.7%
- **PaymentMethod_Credit card (automatic)**: Decreases churn risk by 17.6%
- **StreamingMovies_No internet service**: Decreases churn risk by 20.1%
- **StreamingTV_No internet service**: Decreases churn risk by 20.1%
- **InternetService_No**: Decreases churn risk by 20.1%
- **TechSupport_No internet service**: Decreases churn risk by 20.1%
- **OnlineBackup_No internet service**: Decreases churn risk by 20.1%
- **DeviceProtection_No internet service**: Decreases churn risk by 20.1%
- **OnlineSecurity_No internet service**: Decreases churn risk by 20.1%
- **Partner_Yes**: Decreases churn risk by 26.7%
- **OnlineBackup_Yes**: Decreases churn risk by 26.9%
- **TechSupport_Yes**: Decreases churn risk by 28.5%
- **OnlineSecurity_Yes**: Decreases churn risk by 34.6%
- **Contract_One year**: Decreases churn risk by 48.6%
- **Contract_Two year**: Decreases churn risk by 65.2%

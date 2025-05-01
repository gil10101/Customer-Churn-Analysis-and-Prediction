# Data Cleaning Report

## Missing Values Before Cleaning
```
customerID           0
gender               0
SeniorCitizen        0
Partner              0
Dependents           0
tenure               0
PhoneService         0
MultipleLines        0
InternetService      0
OnlineSecurity       0
OnlineBackup         0
DeviceProtection     0
TechSupport          0
StreamingTV          0
StreamingMovies      0
Contract             0
PaperlessBilling     0
PaymentMethod        0
MonthlyCharges       0
TotalCharges         0
Churn                0
tenure_group        11
```

## Duplicate Rows: 0

## Transformations Applied
- Converted SeniorCitizen from 0/1 to No/Yes
- Converted TotalCharges to numeric type
- Set TotalCharges to 0 for customers with 0 tenure

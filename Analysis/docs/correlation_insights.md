# Correlation Analysis

## Features Most Positively Correlated with Churn

```
InternetService_Fiber optic       0.308020
PaymentMethod_Electronic check    0.301919
MonthlyCharges                    0.193356
PaperlessBilling_Yes              0.191825
SeniorCitizen_Yes                 0.150889
StreamingTV_Yes                   0.063228
StreamingMovies_Yes               0.061382
MultipleLines_Yes                 0.040102
customerID_1114-CENIM             0.019827
customerID_6437-UKHMV             0.019827
```

## Features Most Negatively Correlated with Churn

```
TotalCharges                           -0.198324
DeviceProtection_No internet service   -0.227890
StreamingMovies_No internet service    -0.227890
StreamingTV_No internet service        -0.227890
TechSupport_No internet service        -0.227890
OnlineBackup_No internet service       -0.227890
OnlineSecurity_No internet service     -0.227890
InternetService_No                     -0.227890
Contract_Two year                      -0.302253
tenure                                 -0.352229
```

## Key Insights from Correlation Analysis

- Month-to-month contracts show strong positive correlation with churn
- Longer tenure shows strong negative correlation with churn
- Two-year contracts have strong negative correlation with churn
- Fiber optic internet service shows positive correlation with churn
- Electronic check payment method positively correlates with churn

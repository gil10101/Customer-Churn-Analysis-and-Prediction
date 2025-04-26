# PyTorch Model Evaluation

## Model Architecture

```
ChurnPredictionModel(
  (layers): Sequential(
    (0): Linear(in_features=30, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=64, out_features=32, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.2, inplace=False)
    (6): Linear(in_features=32, out_features=16, bias=True)
    (7): ReLU()
    (8): Linear(in_features=16, out_features=1, bias=True)
    (9): Sigmoid()
  )
)
```

## Evaluation Metrics

- **Accuracy**: 0.7913
- **Precision**: 0.6299
- **Recall**: 0.5187
- **F1 Score**: 0.5689
- **ROC AUC**: 0.8349

## Top 15 Features by Importance

```
                           Feature  Importance
25               Contract_Two year    0.250203
24               Contract_One year    0.231883
8   MultipleLines_No phone service    0.170057
29      PaymentMethod_Mailed check    0.161247
19                 TechSupport_Yes    0.158408
15                OnlineBackup_Yes    0.153453
9                MultipleLines_Yes    0.146996
4                SeniorCitizen_Yes    0.145353
0                           tenure    0.144882
28  PaymentMethod_Electronic check    0.144240
10     InternetService_Fiber optic    0.142732
21                 StreamingTV_Yes    0.141942
13              OnlineSecurity_Yes    0.141737
5                      Partner_Yes    0.139197
23             StreamingMovies_Yes    0.134895
```

## Insights

- The PyTorch model achieved good performance in predicting customer churn.
- The most important features align with our previous analysis, with tenure, contract type, and charges being significant predictors.
- The confusion matrix shows that the model is better at predicting customers who won't churn than those who will, which is common in imbalanced datasets.
- To improve model performance, we could explore techniques for handling imbalanced data, such as oversampling or using weighted loss functions.

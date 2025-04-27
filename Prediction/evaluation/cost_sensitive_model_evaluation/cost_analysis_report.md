# Cost-Sensitive Churn Model Evaluation Report

## Overview
This report presents the results of cost-sensitive churn prediction models. The models are evaluated based on their ability to minimize the total business cost, rather than just optimizing for accuracy metrics.

**Baseline Cost if No Model: $187000.00 per period ($2244000.00 annually)**

## Cost Parameters
- Retention Cost: $100 per customer (cost of offering discounts, loyalty programs, etc.)
- Churn Cost: $500 per customer (lost revenue, acquisition cost of replacement)

## Model Performance

### Cost Comparison
| Model | Total Cost | Cost Savings | Annual Savings (Estimated) | Optimal Threshold |
|-------|------------|--------------|----------------------------|------------------|
| logistic_regression | $93300.00 | $93700.00 | $1124400.00 | 0.340 |
| random_forest | $95400.00 | $91600.00 | $1099200.00 | 0.150 |
| gradient_boosting | $93400.00 | $93600.00 | $1123200.00 | 0.210 |
| xgboost | $96600.00 | $90400.00 | $1084800.00 | 0.230 |

### Classification Metrics for Cost-Optimal Models
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| logistic_regression | 0.674 | 0.444 | 0.912 | 0.597 |
| random_forest | 0.676 | 0.445 | 0.890 | 0.593 |
| gradient_boosting | 0.722 | 0.486 | 0.850 | 0.619 |
| xgboost | 0.697 | 0.462 | 0.853 | 0.599 |

## Interpretation

### Best Model: logistic_regression
The logistic_regression model provides the highest cost savings at $93700.00 per prediction period, with an estimated annual savings of $1124400.00.

### Business Impact
By implementing the cost-sensitive logistic_regression model, the business can expect to reduce churn-related costs by approximately 50.1% compared to the baseline approach of not intervening with any customers.

### Key Considerations
1. The optimal threshold is not 0.5, but is determined by the relative costs of retention efforts versus customer churn.
2. This model prioritizes identifying customers who are likely to churn AND can be retained cost-effectively.
3. The model accounts for the cost of unnecessary retention efforts (false positives) while balancing the higher cost of missed churn predictions (false negatives).

## Summary of Models Ranked by Cost Savings

| Rank | Model | Cost Savings | Annual Savings | % Improvement Over Baseline |
|------|-------|-------------|----------------|-----------------------------|
| 1 | logistic_regression | $93700.00 | $1124400.00 | 50.1% |
| 2 | gradient_boosting | $93600.00 | $1123200.00 | 50.1% |
| 3 | random_forest | $91600.00 | $1099200.00 | 49.0% |
| 4 | xgboost | $90400.00 | $1084800.00 | 48.3% |

# Model Comparison

|                    |   accuracy |   precision |   recall |       f1 |   roc_auc |
|:-------------------|-----------:|------------:|---------:|---------:|----------:|
| Ensemble Model     |   0.787793 |    0.59542  | 0.625668 | 0.610169 |  0.84407  |
| Optimized GB Model |   0.799858 |    0.657534 | 0.513369 | 0.576577 |  0.844395 |

## Conclusion

The Ensemble Model performs better overall based on F1 score, which balances precision and recall.
This model should be used for production deployment for churn prediction.

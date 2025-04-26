# Numerical Features Insights

## tenure

### Overall Statistics

```
count    7043.000000
mean       32.371149
std        24.559481
min         0.000000
25%         9.000000
50%        29.000000
75%        55.000000
max        72.000000
```

### Statistics by Churn

```
        count       mean        std  min   25%   50%   75%   max
Churn                                                           
No     5174.0  37.569965  24.113777  0.0  15.0  38.0  61.0  72.0
Yes    1869.0  17.979133  19.531123  1.0   2.0  10.0  29.0  72.0
```

- Customers who churn have significantly lower tenure
- Long-term customers (high tenure) are less likely to churn

## MonthlyCharges

### Overall Statistics

```
count    7043.000000
mean       64.761692
std        30.090047
min        18.250000
25%        35.500000
50%        70.350000
75%        89.850000
max       118.750000
```

### Statistics by Churn

```
        count       mean        std    min    25%     50%   75%     max
Churn                                                                  
No     5174.0  61.265124  31.092648  18.25  25.10  64.425  88.4  118.75
Yes    1869.0  74.441332  24.666053  18.85  56.15  79.650  94.2  118.35
```

- Customers who churn tend to have higher monthly charges
- This suggests premium services might be a factor in churn decisions

## TotalCharges

### Overall Statistics

```
count    7043.000000
mean     2279.734304
std      2266.794470
min         0.000000
25%       398.550000
50%      1394.550000
75%      3786.600000
max      8684.800000
```

### Statistics by Churn

```
        count         mean          std    min    25%       50%      75%      max
Churn                                                                            
No     5174.0  2549.911442  2329.954215   0.00  572.9  1679.525  4262.85  8672.45
Yes    1869.0  1531.796094  1890.822994  18.85  134.5   703.550  2331.30  8684.80
```


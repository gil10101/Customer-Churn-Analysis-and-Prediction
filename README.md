# Customer Churn Analysis and Prediction

End-to-end churn analysis of 7,043 telecom customer records (IBM Telco Customer Churn dataset): exploratory analysis, customer segmentation, survival modeling, a 14-classifier benchmark topped by a stacked ensemble (81.1% holdout accuracy, 0.849 ROC AUC, 2.8× top-decile lift), an A/B testing framework with projected retention impact, and a FastAPI prediction service.

## Dataset

7,043 customers, 21 attributes:

- Demographics — gender, senior-citizen status, partner, dependents
- Account — tenure (months), contract type, payment method, paperless billing
- Services — phone, multiple lines, internet type, security/backup/support add-ons, streaming
- Billing — monthly charges, total charges
- Target — churn within the observation window (26.5% of customers)

A cleaned copy is checked in at `Analysis/data/telco_churn_cleaned.csv`. Scripts that expect the raw file look for `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (available from the [IBM sample datasets](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).

## Key results

All figures below are computed from the checked-in dataset with fixed seeds (see `Analysis/scripts/` and `Prediction/scripts/`).

**Churn concentrates early and on flexible contracts.**

| Cohort | Customers | Churn rate |
|---|---|---|
| Month-to-month contract | 3,875 | 42.7% |
| One-year contract | 1,473 | 11.3% |
| Two-year contract | 1,695 | 2.8% |
| Tenure 0–12 months | 2,186 | 47.4% |
| Tenure 49–72 months | 2,239 | 9.5% |
| Fiber-optic internet | 3,096 | 41.9% |

- Median tenure at churn is 10 months, versus 38 months for retained customers.
- Strongest positive churn correlates: fiber-optic internet (+0.31), electronic-check payment (+0.30), monthly charges (+0.19). Strongest negative: tenure (−0.35), two-year contract (−0.30).
- Churned customers represent $139K of $456K (30.5%) in monthly recurring charges.

**Survival analysis (Kaplan-Meier, Cox proportional hazards).** Two-year contracts retain 93.6% of customers through month 72; month-to-month retains 12.9%. Cox hazard ratios: fiber-optic internet HR 2.27, electronic check HR 1.78, one-year contract HR 0.22, two-year contract HR 0.08 (all p < 0.01).

**Segmentation (K-Means, k=4 on tenure / monthly / total charges).**

| Segment | Customers | Churn rate | Avg tenure | Avg monthly | Monthly revenue churned |
|---|---|---|---|---|---|
| New · High spend | 2,276 | 48.2% | 15 mo | $81 | $91.8K |
| New · Basic service | 1,703 | 24.7% | 10 mo | $32 | $15.7K |
| Established · Premium | 1,904 | 15.4% | 60 mo | $93 | $28.9K |
| Loyal · Value plans | 1,160 | 5.0% | 54 mo | $35 | $2.7K |

The "New · High spend" segment accounts for 66% of churned monthly revenue and is the primary retention target.

**Classifier benchmark (80/20 stratified holdout, 5-fold CV on the training split, seed 42).** Fourteen models under identical preprocessing; LightGBM and XGBoost tuned with Optuna (60 TPE trials each, AUC objective).

| Model | CV accuracy | Test accuracy | ROC AUC | F1 |
|---|---|---|---|---|
| Stacked Ensemble v2 (LGBM+XGB+GB+RF+LR) † | 0.807 | **0.811** | **0.849** | 0.614 |
| Logistic Regression | 0.803 | 0.806 | 0.842 | 0.605 |
| XGBoost (Optuna) | 0.806 | 0.804 | **0.849** | 0.584 |
| AdaBoost | 0.807 | 0.803 | 0.843 | 0.581 |
| Extra Trees | 0.800 | 0.803 | 0.839 | 0.575 |
| CatBoost | 0.798 | 0.802 | 0.842 | 0.584 |
| Random Forest | 0.803 | 0.801 | 0.842 | 0.582 |
| LightGBM (Optuna) | 0.804 | 0.801 | 0.847 | 0.571 |
| Soft-Voting Ensemble | 0.807 | 0.800 | 0.845 | 0.580 |
| Gradient Boosting | 0.802 | 0.798 | 0.842 | 0.572 |
| Hist Gradient Boosting | 0.801 | 0.797 | 0.837 | 0.573 |
| SVM (RBF) | 0.804 | 0.793 | 0.796 | 0.534 |
| KNN | 0.790 | 0.778 | 0.815 | 0.578 |
| Naive Bayes | 0.666 | 0.656 | 0.809 | 0.572 |

† Scored at its CV-tuned decision threshold (0.46, chosen on out-of-fold training predictions); all other models at 0.50.

**Final model** (`Prediction/models/final/churn_model_stack_v2.joblib`): stacking ensemble with engineered features (service counts, contract × payment interactions, spend deltas). Holdout: 81.1% accuracy, 0.849 AUC, precision 0.67 / recall 0.57. Ranking quality is the operative metric for retention targeting: the top-scored decile is 75.0% churners (**2.8× lift** over the 26.5% base rate), the top 20% captures 50.8% of all churners, and the top 30% captures 66.8%. Because 73.5% accuracy is attainable by predicting "no churn" for everyone, lift and AUC are the headline metrics; raw accuracy on this dataset plateaus at ~81% across all model families.

Top features by Gradient Boosting importance: tenure (0.30), fiber-optic internet (0.19), electronic-check payment (0.12), two-year contract (0.07).

**A/B testing framework (simulated cohorts, seeded).** Five retention strategies evaluated with power analysis and significance testing. Highest simulated ROI: free premium tech support for fiber-optic users (+22.8% relative retention lift, ROI 160%) and discount offers for price-sensitive churners (+16.6% lift, ROI 137%). Results in `Analysis/results/ab_test_strategy_results.csv`.

**Churn trend analysis** operates on a synthetic time index derived from tenure (the source dataset has no calendar dimension); it demonstrates the seasonal-decomposition and event-impact methodology rather than observed calendar effects.

## Impact projections

Real cohort sizes and churned-customer revenue combined with the simulated A/B lift rates and the final model's targeting capture (contacting the top-30% risk scores reaches 66.8% of churners at roughly a third of blanket-outreach cost). Save rate per strategy = (treatment − baseline retention) / (1 − baseline retention); 12-month revenue horizon against one-time program cost.

| Strategy (model-targeted) | Pool | Contacted | Customers saved / yr | Net 12-mo impact | ROI |
|---|---|---|---|---|---|
| Free premium tech support (fiber-optic users) | 3,096 | 929 | 249 | $216K | 466% |
| Contract incentive (month-to-month) | 3,875 | 1,162 | 224 | $115K | 141% |
| Service upgrade (new high-spend) | 1,233 | 370 | 87 | $70K | 318% |
| Enhanced loyalty program (tenure ≥ 48 mo) | 2,303 | 691 | 35 | $22K | 126% |
| Discount offer (low-spend) | 2,019 | 606 | 52 | −$10K | −35% |

Running the four positive-ROI programs projects ~595 customers retained per year, $49K/month of revenue protected ($591K over 12 months), against $167K program cost — net $423K (253% ROI). Blanket (untargeted) outreach drops the contract-incentive program to $22K net and the discount program to −$64K, which is the quantitative case for model-based targeting. Projections inherit the simulated lift rates and are labeled as such; `dashboards/06_impact_planner.html` includes a conservative case at half lift.

## Repository layout

```
├── notebooks/            # Jupyter entry points (01–08) mirroring the analysis stages
├── Analysis/
│   ├── scripts/          # EDA, segmentation, survival, A/B testing, trend analysis
│   ├── data/             # Cleaned dataset, cluster profiles, marketing strategies
│   ├── docs/             # Generated findings (feature insights, cluster analysis, …)
│   ├── images/           # Generated figures (eda/, segmentation/, survival_analysis/, …)
│   └── results/          # Survival, segmentation, and A/B test outputs
├── Prediction/
│   ├── scripts/          # Neural network, ensemble, cost-sensitive models
│   ├── models/           # Serialized model artifacts and metrics
│   │   └── final/        # Stacked Ensemble v2 (production model) + metadata
│   └── evaluation/       # Model comparison outputs
├── dashboards/           # Static HTML dashboards built from the analysis outputs
├── portfolio-assets/     # Regenerated portfolio figures (PNG)
├── tools/portfolio/      # Scripts that regenerate results.json, figures, and dashboards
├── api/                  # FastAPI prediction service (Docker, monitoring, deployment)
├── utils/                # Shared preprocessing, evaluation, and plotting utilities
├── tests/                # Pytest suites for pipeline, API, and deployment
└── requirements.txt
```

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/gil10101/Customer-Churn-Analysis-and-Prediction.git
cd Customer-Churn-Analysis-and-Prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the analysis

Scripts are run from their own directory and write figures/results into `Analysis/` and `Prediction/`:

```bash
cd Analysis/scripts
python exploratory_data_analysis.py    # EDA figures + insight docs
python customer_segmentation.py        # K-Means / DBSCAN segments
python churn_survival_analysis.py      # Kaplan-Meier + Cox models
python ab_testing.py                   # Retention-strategy simulations
python churn_trend_analysis.py         # Seasonal decomposition (synthetic index)

cd ../../Prediction/scripts
python churn_prediction_model.py       # PyTorch neural network
python ensemble_churn_model.py         # Stacked / voting ensembles
python cost_sensitive_churn_model.py   # Cost-optimized thresholds
```

Notebooks `notebooks/01–08` provide the same stages in notebook form; `notebooks/00_master_analysis_overview.ipynb` documents execution order and environment checks.

## Dashboards

`dashboards/` contains five static HTML dashboards generated from the analysis outputs (no server required — open in a browser):

1. `01_executive_overview.html` — KPIs, churn drivers, revenue at risk
2. `02_customer_flow_sankey.html` — contract → tenure → outcome flows
3. `03_segments.html` — segment profiles and churn risk map
4. `04_retention_survival.html` — survival curves, hazard ratios, interventions
5. `05_model_performance.html` — benchmark, ROC, confusion matrix, feature importance
6. `06_impact_planner.html` — projected saves, cost, net impact, and ROI per intervention

## Prediction API

`api/` contains a FastAPI service exposing the trained model with request validation, Redis caching, Prometheus monitoring, and Docker deployment. See `api/README.md` and `api/DEPLOYMENT.md`.

```bash
cd api && python run_server.py    # development server on :8000
```

## Testing

```bash
pytest tests/
```

Covers feature engineering, imbalance handling, the training pipeline, the prediction service, API integration, and deployment validation.

# Analysis Notebooks

Eight Jupyter notebooks mirror the stages of the churn analysis pipeline. Each notebook sets up its stage's environment, imports, and data contracts; the underlying implementations live in `Analysis/scripts/`, `Prediction/scripts/`, and `utils/`, and can be run either from the notebooks or directly as scripts.

## Notebooks

| # | Notebook | Stage |
|---|---|---|
| 00 | `00_master_analysis_overview.ipynb` | Execution order, environment validation, runtime notes |
| 01 | `01_exploratory_data_analysis.ipynb` | Data quality assessment, distributions, churn correlates |
| 02 | `02_customer_segmentation.ipynb` | K-Means / DBSCAN / hierarchical clustering, cluster validation |
| 03 | `03_churn_prediction_modeling.ipynb` | Classifier training: logistic regression, tree ensembles, PyTorch network |
| 04 | `04_survival_analysis.ipynb` | Kaplan-Meier estimation, Cox proportional hazards (lifelines) |
| 05 | `05_ab_testing_framework.ipynb` | Power analysis, simulated retention experiments, significance testing |
| 06 | `06_cost_sensitive_modeling.ipynb` | Cost matrices, threshold optimization |
| 07 | `07_model_comparison_evaluation.ipynb` | Cross-model benchmark with standardized metrics |
| 08 | `08_business_insights_recommendations.ipynb` | Findings synthesis, ROI quantification |

## Related implementations

- Stage 01 → `Analysis/scripts/exploratory_data_analysis.py`, `generate_correlation_heatmap.py`
- Stage 02 → `Analysis/scripts/customer_segmentation.py`, `generate_segmentation_images.py`
- Stage 03 → `Prediction/scripts/churn_prediction_model.py`, `ensemble_churn_model.py`
- Stage 04 → `Analysis/scripts/churn_survival_analysis.py`, `utils/survival_utils.py`
- Stage 05 → `Analysis/scripts/ab_testing.py`
- Stage 06 → `Prediction/scripts/cost_sensitive_churn_model.py`, `utils/cost_sensitive_utils.py`
- Stage 07 → `utils/model_evaluation.py`, `utils/model_training_pipeline.py`

## Usage

```bash
pip install -r ../requirements.txt
jupyter notebook
```

Run notebooks in numerical order; `00_master_analysis_overview.ipynb` lists dependencies between stages. The dataset is expected at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (a cleaned copy is checked in at `Analysis/data/telco_churn_cleaned.csv`).

## Outputs

Figures are written to `Analysis/images/` and `Prediction/evaluation/images/`; tabular results to `Analysis/results/` and `Prediction/models/`; generated findings documents to `Analysis/docs/`.

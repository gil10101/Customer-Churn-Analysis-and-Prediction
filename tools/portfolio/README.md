# Portfolio asset generators

Regenerates every number, figure, and dashboard from the checked-in dataset (`Analysis/data/telco_churn_cleaned.csv`, seed 42).

```bash
python run_analysis.py       # EDA, base benchmark, survival, segmentation → results.json
python final_experiment.py   # Optuna-tuned LGBM/XGB, CatBoost, stacked ensemble v2,
                             # impact projections; saves the final model artifact and
                             # regenerates baseline/segmentation/ensemble CSVs
python gen_figures.py        # → portfolio-assets/ (4 PNGs, 2240×1440)
python gen_dashboards.py     # → dashboards/ (6 static HTML)
```

Requires: `pip install -r ../../requirements.txt` (pandas, scikit-learn, lifelines, xgboost, lightgbm, catboost, optuna, matplotlib, pillow).

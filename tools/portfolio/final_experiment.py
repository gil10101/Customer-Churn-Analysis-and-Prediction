#!/usr/bin/env python3
"""Final modeling round with XGBoost / LightGBM / CatBoost / Optuna.
Same discipline: 80/20 stratified holdout (seed 42), all tuning via CV on
the training split only. Produces:
  - updated benchmark entries + final-model metrics in results.json
  - saved final model artifact + metadata (Prediction/models/final/)
  - clean regenerated artifacts: baseline_feature_importance.csv,
    cluster_profiles.csv / cluster_centers.csv, ensemble_metrics.csv
  - impact projections (results.json["impact"])
"""
import json
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler

SEED = 42
HERE = SCRATCH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(SCRATCH, "results.json")

df = pd.read_csv(os.path.join(REPO, "Analysis/data/telco_churn_cleaned.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
y = (df["Churn"] == "Yes").astype(int)

SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

def engineer(d):
    d = d.copy()
    d["NumServices"] = sum((d[c] == "Yes").astype(int) for c in SERVICES)
    d["FiberNoSupport"] = ((d["InternetService"] == "Fiber optic") &
                           (d["TechSupport"] == "No")).astype(int)
    d["MtmEcheck"] = ((d["Contract"] == "Month-to-month") &
                      (d["PaymentMethod"] == "Electronic check")).astype(int)
    d["AutoPay"] = d["PaymentMethod"].str.contains("automatic").astype(int)
    d["NewCustomer"] = (d["tenure"] <= 6).astype(int)
    d["AvgMonthlySpend"] = d["TotalCharges"] / d["tenure"].clip(lower=1)
    d["ChargeDelta"] = d["MonthlyCharges"] - d["AvgMonthlySpend"]
    d["TenureXMtm"] = d["tenure"] * (d["Contract"] == "Month-to-month")
    return d

raw = df.drop(columns=["customerID", "Churn"])
eng = engineer(raw)
X = pd.get_dummies(eng, drop_first=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                      random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

def holdout_metrics(prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"test_accuracy": float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred)),
            "recall": float(recall_score(yte, pred)),
            "f1": float(f1_score(yte, pred)),
            "roc_auc": float(roc_auc_score(yte, prob))}

# ---------------------------------------------------------------- optuna: LGBM
def lgbm_obj(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 200, 900),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 4, 24),
         "max_depth": trial.suggest_int("max_depth", 2, 5),
         "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True)}
    m = LGBMClassifier(random_state=SEED, verbosity=-1, **p)
    return cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

st = optuna.create_study(direction="maximize",
                         sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(lgbm_obj, n_trials=60, show_progress_bar=False)
lgbm = LGBMClassifier(random_state=SEED, verbosity=-1, **st.best_params).fit(Xtr, ytr)
m_lgbm = holdout_metrics(lgbm.predict_proba(Xte)[:, 1])
print("LightGBM (optuna) ", m_lgbm, flush=True)

# ---------------------------------------------------------------- optuna: XGB
def xgb_obj(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 200, 900),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
         "max_depth": trial.suggest_int("max_depth", 2, 5),
         "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20, log=True),
         "gamma": trial.suggest_float("gamma", 0, 5)}
    m = XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=2, **p)
    return cross_val_score(m, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

sx = optuna.create_study(direction="maximize",
                         sampler=optuna.samplers.TPESampler(seed=SEED))
sx.optimize(xgb_obj, n_trials=60, show_progress_bar=False)
xgb = XGBClassifier(random_state=SEED, eval_metric="logloss",
                    **sx.best_params).fit(Xtr, ytr)
m_xgb = holdout_metrics(xgb.predict_proba(Xte)[:, 1])
print("XGBoost (optuna)  ", m_xgb, flush=True)

# ---------------------------------------------------------------- CatBoost
cb = CatBoostClassifier(iterations=1200, learning_rate=0.03, depth=4,
                        l2_leaf_reg=6, random_seed=SEED, verbose=0)
cb.fit(Xtr, ytr)
m_cb = holdout_metrics(cb.predict_proba(Xte)[:, 1])
print("CatBoost          ", m_cb, flush=True)

# ---------------------------------------------------------------- final stack
stack = StackingClassifier(
    estimators=[("lgbm", LGBMClassifier(random_state=SEED, verbosity=-1,
                                        **st.best_params)),
                ("xgb", XGBClassifier(random_state=SEED, eval_metric="logloss",
                                      **sx.best_params)),
                ("gb", GradientBoostingClassifier(random_state=SEED,
                                                  learning_rate=0.03,
                                                  n_estimators=200, max_depth=2,
                                                  subsample=0.7,
                                                  min_samples_leaf=15)),
                ("rf", RandomForestClassifier(n_estimators=500,
                                              min_samples_leaf=4,
                                              random_state=SEED, n_jobs=-1)),
                ("lr", LogisticRegression(max_iter=2000, C=0.5,
                                          random_state=SEED))],
    final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
    cv=cv, n_jobs=-1, stack_method="predict_proba")
stack.fit(Xtr, ytr)
prob_stack = stack.predict_proba(Xte)[:, 1]
m_stack = holdout_metrics(prob_stack)
print("Final stack       ", m_stack, flush=True)

# CV-chosen accuracy-optimal threshold (train OOF only — no leakage)
oof = cross_val_predict(stack, Xtr, ytr, cv=cv, method="predict_proba",
                        n_jobs=-1)[:, 1]
thrs = np.linspace(0.3, 0.7, 81)
t_best = float(thrs[int(np.argmax([accuracy_score(ytr, (oof >= t).astype(int))
                                   for t in thrs]))])
m_stack_thr = holdout_metrics(prob_stack, t_best)
m_stack_thr["threshold"] = t_best
print(f"Final stack @thr {t_best:.2f}", m_stack_thr, flush=True)

# holdout lift metrics for the final model
order = np.argsort(-prob_stack)
yo = yte.values[order]
base = float(yte.mean())
lift = {}
for frac in [0.1, 0.2, 0.3]:
    k = int(len(yo) * frac)
    lift[f"top_{int(frac*100)}"] = {
        "precision": float(yo[:k].mean()),
        "lift": float(yo[:k].mean() / base),
        "capture": float(yo[:k].sum() / yo.sum())}
print("lift:", lift, flush=True)

# ---------------------------------------------------------------- save artifacts
final_dir = os.path.join(REPO, "Prediction/models/final")
os.makedirs(final_dir, exist_ok=True)
joblib.dump(stack, os.path.join(final_dir, "churn_model_stack_v2.joblib"))
meta = {"model": "StackingClassifier(LGBM+XGB+GB+RF+LR -> LR)",
        "trained": "2026-07-29", "seed": SEED,
        "split": "80/20 stratified holdout",
        "features": "one-hot + engineered (see tools/portfolio)",
        "holdout": m_stack, "holdout_at_tuned_threshold": m_stack_thr,
        "lift": lift,
        "lgbm_params": st.best_params, "xgb_params": sx.best_params}
with open(os.path.join(final_dir, "churn_model_stack_v2_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

# regenerate ensemble metrics CSV (replaces stale 75.8% artifact)
pd.DataFrame([{**{"model": "stacking_v2"}, **m_stack,
               "cv_auc_lgbm": st.best_value, "cv_auc_xgb": sx.best_value}]).to_csv(
    os.path.join(REPO, "Prediction/models/ensemble/ensemble_metrics.csv"),
    index=False)

# clean baseline feature importance (replaces customerID-polluted artifact)
sc = StandardScaler().fit(Xtr)
lr_clean = LogisticRegression(max_iter=4000, C=0.3, random_state=SEED).fit(
    sc.transform(Xtr), ytr)
imp = pd.DataFrame({"Feature": X.columns,
                    "Importance": np.abs(lr_clean.coef_[0]),
                    "Sign": np.sign(lr_clean.coef_[0])}).sort_values(
    "Importance", ascending=False)
imp.to_csv(os.path.join(REPO,
           "Prediction/models/baseline/baseline_feature_importance.csv"),
           index=False)

# clean segmentation artifacts (replaces 138-micro-cluster profile)
seg_feats = df[["tenure", "MonthlyCharges", "TotalCharges"]]
seg_sc = StandardScaler().fit(seg_feats)
km = KMeans(n_clusters=4, n_init=20, random_state=SEED).fit(
    seg_sc.transform(seg_feats))
df["_seg"] = km.labels_
prof = df.groupby("_seg").agg(
    Size=("_seg", "size"),
    Churn_Rate=("Churn", lambda s: (s == "Yes").mean() * 100),
    Avg_tenure=("tenure", "mean"),
    Avg_MonthlyCharges=("MonthlyCharges", "mean"),
    Avg_TotalCharges=("TotalCharges", "mean")).reset_index().rename(
    columns={"_seg": "Cluster"})
prof.insert(2, "Percentage", prof["Size"] / len(df) * 100)
prof.to_csv(os.path.join(REPO, "Analysis/data/cluster_profiles.csv"), index=False)
centers = pd.DataFrame(seg_sc.inverse_transform(km.cluster_centers_),
                       columns=["tenure", "MonthlyCharges", "TotalCharges"])
centers.insert(0, "Cluster", range(4))
centers.to_csv(os.path.join(REPO, "Analysis/data/cluster_centers.csv"), index=False)

# ---------------------------------------------------------------- impact model
R = json.load(open(RESULTS))
ab = {s["strategy_name"]: s for s in R["ab_tests"]}

pools = {}
def pool_stats(mask, name):
    sub = df[mask]
    ch = sub["Churn"] == "Yes"
    pools[name] = {
        "n": int(len(sub)), "churners": int(ch.sum()),
        "avg_monthly_churner": float(sub.loc[ch, "MonthlyCharges"].mean()),
        "monthly_rev_at_risk": float(sub.loc[ch, "MonthlyCharges"].sum())}

pool_stats(df["InternetService"] == "Fiber optic", "Free Premium Technical Support")
pool_stats(df["Contract"] == "Month-to-month", "Contract Incentive")
pool_stats((df["MonthlyCharges"] < 45), "Discount Offer")
pool_stats((df["MonthlyCharges"] >= 75) & (df["tenure"] < 30), "Free Service Upgrade")
pool_stats(df["tenure"] >= 48, "Enhanced Loyalty Program")

TOP30_CAPTURE = lift["top_30"]["capture"]      # share of churners the model flags
TARGET_FRAC = 0.30                             # contact top-30% risk scores
HORIZON = 12                                   # months of protected revenue

impact = []
for name, s in ab.items():
    p = pools[name]
    save_rate = ((s["treatment_retention"] - s["baseline_retention"]) /
                 (1 - s["baseline_retention"]))  # share of would-be churners saved
    for mode in ["blanket", "model-targeted"]:
        if mode == "blanket":
            contacted = p["n"]
            churners_reached = p["churners"]
        else:
            contacted = int(round(p["n"] * TARGET_FRAC))
            churners_reached = p["churners"] * TOP30_CAPTURE
        saved = churners_reached * save_rate
        rev_saved_mo = saved * p["avg_monthly_churner"]
        cost = contacted * s["cost_per_customer"]
        net = rev_saved_mo * HORIZON - cost
        impact.append({
            "strategy": s["strategy_name"], "target": s["target_segment"],
            "mode": mode, "pool": p["n"], "contacted": contacted,
            "pool_churners": p["churners"], "save_rate": float(save_rate),
            "customers_saved": float(saved),
            "monthly_revenue_saved": float(rev_saved_mo),
            "cost": float(cost), "horizon_months": HORIZON,
            "net_12mo": float(net),
            "roi_pct": float(net / cost * 100) if cost else 0.0})

R["impact"] = impact
R["impact_assumptions"] = {
    "save_rate": "(treatment_retention - baseline_retention) / (1 - baseline_retention) from simulated A/B results",
    "model_targeting": f"contact top {int(TARGET_FRAC*100)}% risk scores; captures {TOP30_CAPTURE*100:.1f}% of churners (final-model holdout)",
    "horizon_months": HORIZON}

# fold new models into benchmark for the dashboard
fpr, tpr, _ = roc_curve(yte, prob_stack)
keep = np.linspace(0, len(fpr) - 1, 120).astype(int)
cvacc = cross_val_score(stack, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
for nm, mm, mdl_prob in [("LightGBM (Optuna)", m_lgbm, lgbm.predict_proba(Xte)[:, 1]),
                         ("XGBoost (Optuna)", m_xgb, xgb.predict_proba(Xte)[:, 1]),
                         ("CatBoost", m_cb, cb.predict_proba(Xte)[:, 1]),
                         ("Stacked Ensemble v2", m_stack, prob_stack)]:
    fp, tp, _ = roc_curve(yte, mdl_prob)
    kp = np.linspace(0, len(fp) - 1, 120).astype(int)
    R["benchmark"][nm] = {**mm, "cv_accuracy_mean": float(cvacc.mean()),
                          "cv_accuracy_std": float(cvacc.std())}
    R["roc_curves"][nm] = {"fpr": fp[kp].round(4).tolist(),
                           "tpr": tp[kp].round(4).tolist()}
R["final_model"] = meta
R["final_lift"] = lift
with open(RESULTS, "w") as f:
    json.dump(R, f)
print("results.json updated; artifacts saved", flush=True)

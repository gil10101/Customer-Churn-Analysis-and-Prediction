#!/usr/bin/env python3
"""Master analysis driver: re-runs the churn pipeline end-to-end and dumps
every number the portfolio figures and dashboards need into results.json.
All values computed from Analysis/data/telco_churn_cleaned.csv (7,043 rows)."""
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter, CoxPHFitter

HERE = ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SEED = 42
rng = np.random.RandomState(SEED)

df = pd.read_csv(os.path.join(REPO, "Analysis/data/telco_churn_cleaned.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

out = {"seed": SEED, "n_customers": int(len(df))}

# ---------------------------------------------------------------- EDA
out["n_churned"] = int(df["ChurnFlag"].sum())
out["churn_rate"] = float(df["ChurnFlag"].mean())
out["monthly_revenue"] = float(df["MonthlyCharges"].sum())
out["monthly_revenue_at_risk"] = float(df.loc[df["ChurnFlag"] == 1, "MonthlyCharges"].sum())
out["avg_tenure"] = float(df["tenure"].mean())
out["avg_monthly_charges"] = float(df["MonthlyCharges"].mean())
out["median_tenure_churned"] = float(df.loc[df.ChurnFlag == 1, "tenure"].median())
out["median_tenure_retained"] = float(df.loc[df.ChurnFlag == 0, "tenure"].median())

def churn_by(col):
    g = df.groupby(col).agg(n=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"))
    g["rate"] = g["churned"] / g["n"]
    return {str(k): {"n": int(v["n"]), "churned": int(v["churned"]), "rate": float(v["rate"])}
            for k, v in g.iterrows()}

for col in ["Contract", "InternetService", "PaymentMethod", "TechSupport",
            "OnlineSecurity", "SeniorCitizen", "Partner", "Dependents",
            "PaperlessBilling", "StreamingTV", "gender"]:
    out[f"churn_by_{col}"] = churn_by(col)

# tenure bands
bands = [(0, 12, "0-12 mo"), (12, 24, "13-24 mo"), (24, 48, "25-48 mo"), (48, 72, "49-72 mo")]
out["churn_by_tenure_band"] = {}
for lo, hi, name in bands:
    m = (df["tenure"] > lo) & (df["tenure"] <= hi) if lo else (df["tenure"] <= hi)
    sub = df[m]
    out["churn_by_tenure_band"][name] = {
        "n": int(len(sub)), "churned": int(sub.ChurnFlag.sum()), "rate": float(sub.ChurnFlag.mean())}

# tenure histogram by churn (bin width 6)
edges = list(range(0, 78, 6))
h_ch, _ = np.histogram(df.loc[df.ChurnFlag == 1, "tenure"], bins=edges)
h_re, _ = np.histogram(df.loc[df.ChurnFlag == 0, "tenure"], bins=edges)
out["tenure_hist"] = {"edges": edges, "churned": h_ch.tolist(), "retained": h_re.tolist()}

# monthly charges histogram by churn
edges_mc = list(range(15, 125, 10))
mc_ch, _ = np.histogram(df.loc[df.ChurnFlag == 1, "MonthlyCharges"], bins=edges_mc)
mc_re, _ = np.histogram(df.loc[df.ChurnFlag == 0, "MonthlyCharges"], bins=edges_mc)
out["charges_hist"] = {"edges": edges_mc, "churned": mc_ch.tolist(), "retained": mc_re.tolist()}

# correlation of churn with encoded features (point-biserial via pearson on dummies)
enc = df.drop(columns=["customerID", "Churn"]).copy()
enc = pd.get_dummies(enc, drop_first=True)
corr = enc.corr(numeric_only=True)["ChurnFlag"].drop("ChurnFlag").sort_values()
out["churn_correlations"] = {k: float(v) for k, v in corr.items()}

# ---------------------------------------------------------------- modeling
X = enc.drop(columns=["ChurnFlag"])
y = enc["ChurnFlag"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
scaler = StandardScaler().fit(Xtr)
Xtr_s = pd.DataFrame(scaler.transform(Xtr), columns=X.columns, index=Xtr.index)
Xte_s = pd.DataFrame(scaler.transform(Xte), columns=X.columns, index=Xte.index)

models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, C=0.5, random_state=SEED), True),
    "Random Forest": (RandomForestClassifier(n_estimators=400, min_samples_leaf=4,
                                             random_state=SEED, n_jobs=-1), False),
    "Extra Trees": (ExtraTreesClassifier(n_estimators=400, min_samples_leaf=4,
                                         random_state=SEED, n_jobs=-1), False),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), False),
    "Hist Gradient Boosting": (HistGradientBoostingClassifier(random_state=SEED,
                                                              max_depth=3, learning_rate=0.08,
                                                              max_iter=400), False),
    "AdaBoost": (AdaBoostClassifier(n_estimators=300, random_state=SEED), False),
    "KNN": (KNeighborsClassifier(n_neighbors=25), True),
    "Naive Bayes": (GaussianNB(), True),
    "SVM (RBF)": (SVC(probability=True, random_state=SEED), True),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
bench = {}
roc_curves = {}
for name, (mdl, scaled) in models.items():
    A, B = (Xtr_s, Xte_s) if scaled else (Xtr, Xte)
    cvs = cross_val_score(mdl, A, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
    mdl.fit(A, ytr)
    prob = mdl.predict_proba(B)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(yte, prob)
    keep = np.linspace(0, len(fpr) - 1, 120).astype(int)
    bench[name] = {
        "cv_accuracy_mean": float(cvs.mean()), "cv_accuracy_std": float(cvs.std()),
        "test_accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred)),
        "recall": float(recall_score(yte, pred)),
        "f1": float(f1_score(yte, pred)),
        "roc_auc": float(roc_auc_score(yte, prob)),
    }
    roc_curves[name] = {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()}
    print(f"{name:26s} test_acc={bench[name]['test_accuracy']:.4f} auc={bench[name]['roc_auc']:.4f}")

# soft-voting ensemble of the three strongest families
ens = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000, C=0.5, random_state=SEED)),
        ("gb", GradientBoostingClassifier(random_state=SEED)),
        ("hgb", HistGradientBoostingClassifier(random_state=SEED, max_depth=3,
                                               learning_rate=0.08, max_iter=400)),
    ], voting="soft", n_jobs=-1)
cvs = cross_val_score(ens, Xtr_s, ytr, cv=cv, scoring="accuracy", n_jobs=-1)
ens.fit(Xtr_s, ytr)
prob = ens.predict_proba(Xte_s)[:, 1]
pred = (prob >= 0.5).astype(int)
fpr, tpr, _ = roc_curve(yte, prob)
keep = np.linspace(0, len(fpr) - 1, 120).astype(int)
bench["Soft-Voting Ensemble"] = {
    "cv_accuracy_mean": float(cvs.mean()), "cv_accuracy_std": float(cvs.std()),
    "test_accuracy": float(accuracy_score(yte, pred)),
    "precision": float(precision_score(yte, pred)),
    "recall": float(recall_score(yte, pred)),
    "f1": float(f1_score(yte, pred)),
    "roc_auc": float(roc_auc_score(yte, prob)),
}
roc_curves["Soft-Voting Ensemble"] = {"fpr": fpr[keep].round(4).tolist(),
                                      "tpr": tpr[keep].round(4).tolist()}
print(f"{'Soft-Voting Ensemble':26s} test_acc={bench['Soft-Voting Ensemble']['test_accuracy']:.4f}")

winner = max(bench, key=lambda k: bench[k]["test_accuracy"])
out["benchmark"] = bench
out["roc_curves"] = roc_curves
out["winner"] = winner

# winner confusion matrix + per-class detail
wm, wscaled = models.get(winner, (ens, True))
A, B = (Xtr_s, Xte_s) if (winner == "Soft-Voting Ensemble" or models[winner][1]) else (Xtr, Xte) \
    if winner in models else (Xtr_s, Xte_s)
if winner in models:
    wm = models[winner][0]
    B = Xte_s if models[winner][1] else Xte
else:
    wm = ens
    B = Xte_s
wprob = wm.predict_proba(B)[:, 1]
wpred = (wprob >= 0.5).astype(int)
cm = confusion_matrix(yte, wpred)
out["winner_confusion"] = cm.tolist()
out["test_size"] = int(len(yte))

# permutation-free importance: use GB feature importances (tree-based, interpretable)
gb = models["Gradient Boosting"][0]
imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
out["feature_importance_gb"] = {k: float(v) for k, v in imp.items()}

# ---------------------------------------------------------------- survival
km = KaplanMeierFitter()
kmx = np.linspace(0, 72, 73)
surv = {}
km.fit(df["tenure"], df["ChurnFlag"])
surv["All customers"] = km.survival_function_at_times(kmx).values.round(4).tolist()
for c in ["Month-to-month", "One year", "Two year"]:
    sub = df[df["Contract"] == c]
    km.fit(sub["tenure"], sub["ChurnFlag"])
    surv[c] = km.survival_function_at_times(kmx).values.round(4).tolist()
out["km_times"] = kmx.tolist()
out["km_contract"] = surv

# Cox PH on key covariates
cox_df = df[["tenure", "ChurnFlag", "MonthlyCharges", "SeniorCitizen", "Partner",
             "Dependents", "Contract", "InternetService", "TechSupport",
             "OnlineSecurity", "PaperlessBilling", "PaymentMethod"]].copy()
cox_df = pd.get_dummies(cox_df, drop_first=True)
cox_df = cox_df.loc[:, cox_df.std() > 0]
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_df, duration_col="tenure", event_col="ChurnFlag")
hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
out["cox_hazard_ratios"] = {
    k: {"hr": float(v["exp(coef)"]), "lo": float(v["exp(coef) lower 95%"]),
        "hi": float(v["exp(coef) upper 95%"]), "p": float(v["p"])}
    for k, v in hr.iterrows()}

# ---------------------------------------------------------------- segmentation
seg_feats = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
seg_scaled = StandardScaler().fit_transform(seg_feats)
kmeans = KMeans(n_clusters=4, n_init=20, random_state=SEED).fit(seg_scaled)
df["Segment"] = kmeans.labels_

seg_profile = df.groupby("Segment").agg(
    n=("ChurnFlag", "size"), churn_rate=("ChurnFlag", "mean"),
    tenure=("tenure", "mean"), monthly=("MonthlyCharges", "mean"),
    total=("TotalCharges", "mean"),
    mtm_share=("Contract", lambda s: (s == "Month-to-month").mean()),
    fiber_share=("InternetService", lambda s: (s == "Fiber optic").mean()),
    revenue_at_risk=("MonthlyCharges", "sum"),
)
# name segments by profile, ordered for stable naming
def seg_name(row):
    if row.tenure < 20 and row.monthly < 45:
        return "New · Basic service"
    if row.tenure < 30:
        return "New · High spend"
    if row.monthly >= 75:
        return "Established · Premium"
    return "Loyal · Value plans"

names = {}
for k, row in seg_profile.iterrows():
    names[k] = seg_name(row)
out["segments"] = {}
for k, row in seg_profile.iterrows():
    churned_rev = float(df[(df.Segment == k) & (df.ChurnFlag == 1)]["MonthlyCharges"].sum())
    out["segments"][names[k]] = {
        "n": int(row.n), "churn_rate": float(row.churn_rate),
        "avg_tenure": float(row.tenure), "avg_monthly": float(row.monthly),
        "avg_total": float(row.total), "mtm_share": float(row.mtm_share),
        "fiber_share": float(row.fiber_share),
        "monthly_revenue": float(row.revenue_at_risk),
        "monthly_revenue_churned": churned_rev,
    }

# KM per segment
out["km_segments"] = {}
for k, name in names.items():
    sub = df[df["Segment"] == k]
    km.fit(sub["tenure"], sub["ChurnFlag"])
    out["km_segments"][name] = km.survival_function_at_times(kmx).values.round(4).tolist()

# segment scatter sample for dashboards (400 pts)
samp = df.sample(700, random_state=SEED)
out["segment_scatter"] = [
    {"t": int(r.tenure), "m": float(r.MonthlyCharges), "seg": names[r.Segment],
     "churn": int(r.ChurnFlag)} for r in samp.itertuples()]

# ---------------------------------------------------------------- sankey flows
# Contract -> Tenure cohort -> Outcome
flows = df.groupby(["Contract"]).size().to_dict()
sankey = []
tb = pd.cut(df["tenure"], bins=[-1, 12, 48, 72], labels=["0-12 mo", "13-48 mo", "49-72 mo"])
df["TenureBand3"] = tb
g1 = df.groupby(["Contract", "TenureBand3"], observed=True).size()
for (c, t), n in g1.items():
    sankey.append({"src": c, "dst": str(t), "n": int(n)})
g2 = df.groupby(["TenureBand3", "Churn"], observed=True).size()
for (t, ch), n in g2.items():
    sankey.append({"src": str(t), "dst": "Churned" if ch == "Yes" else "Retained", "n": int(n)})
out["sankey"] = sankey

# internet service -> churn flows (secondary sankey option)
g3 = df.groupby(["InternetService", "Churn"]).size()
out["sankey_internet"] = [
    {"src": s, "dst": "Churned" if c == "Yes" else "Retained", "n": int(n)}
    for (s, c), n in g3.items()]

# ---------------------------------------------------------------- A/B tests (existing results)
ab = pd.read_csv(os.path.join(REPO, "Analysis/results/ab_test_strategy_results.csv"))
out["ab_tests"] = ab.to_dict(orient="records")

with open(os.path.join(ROOT, "results.json"), "w") as f:
    json.dump(out, f)
print("\nWinner:", winner, bench[winner])
print("Saved results.json,", os.path.getsize(os.path.join(ROOT, "results.json")), "bytes")

#!/usr/bin/env python3
"""Portfolio figures — all values read from results.json (regenerated pipeline).
Shared style: warm-neutral surface #fafaf9, slate/salmon/violet/gold palette,
2240x1440 px each (11.2 x 7.2 in @ 200 dpi)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "portfolio-assets")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))

# ---------------------------------------------------------------- shared style
SURFACE = "#fafaf9"
BLUE = "#3a6fa5"      # slot 1 — retained / primary series
SALMON = "#dd7f68"    # slot 2 — churn / risk
VIOLET = "#7a68b8"    # slot 3
GOLD = "#d99a3d"      # slot 4 — highlight
GRAY = "#b5b1a9"      # de-emphasis
NAVY = "#2e5a80"
INK = "#2b2a28"
INK2 = "#55534e"
MUTED = "#8a877f"
GRID = "#e7e5df"
BASE = "#c9c6bd"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.titlesize": 13.5, "axes.titleweight": "semibold",
    "axes.titlepad": 12, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5, "legend.frameon": False,
})


def new_fig():
    fig = plt.figure()
    fig.set_size_inches(11.2, 7.2)
    return fig


def style_ax(ax, xgrid=False, ygrid=True):
    ax.grid(axis="y" if ygrid and not xgrid else "x" if xgrid else "both",
            color=GRID, linewidth=0.8)
    if xgrid and not ygrid:
        ax.grid(axis="y", visible=False)
    if ygrid and not xgrid:
        ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(BASE)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("saved", name)


def pct(x, d=0):
    return f"{x*100:.{d}f}%"


# KM helpers -------------------------------------------------------------
KMX = np.array(R["km_times"])

def plot_km(ax, series, colors, lw=2.0, order=None, label_x=None):
    """Survival curves with direct end labels."""
    names = order or list(series.keys())
    for name in names:
        y = np.array(series[name])
        ax.plot(KMX, y, color=colors[name], lw=lw, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)
        ax.scatter([KMX[-1]], [y[-1]], s=26, color=colors[name], zorder=4,
                   edgecolor=SURFACE, linewidth=1.4)
    ax.set_xlim(0, 79)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks([0, 12, 24, 36, 48, 60, 72])


# ======================================================================
# 1. EXPLORATORY ANALYSIS — 2x2 grid
# ======================================================================
fig = new_fig()
gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38,
              left=0.135, right=0.965, top=0.93, bottom=0.09)

# (a) churn rate by contract + internet service (grouped feature bars)
ax = fig.add_subplot(gs[0, 0])
feats = [("Month-to-month", R["churn_by_Contract"]["Month-to-month"]["rate"]),
         ("One-year", R["churn_by_Contract"]["One year"]["rate"]),
         ("Two-year", R["churn_by_Contract"]["Two year"]["rate"]),
         ("Fiber optic", R["churn_by_InternetService"]["Fiber optic"]["rate"]),
         ("DSL", R["churn_by_InternetService"]["DSL"]["rate"]),
         ("E-check pay", R["churn_by_PaymentMethod"]["Electronic check"]["rate"]),
         ("Senior citizen", R["churn_by_SeniorCitizen"]["Yes"]["rate"]),
         ("No tech support", R["churn_by_TechSupport"]["No"]["rate"])]
feats.sort(key=lambda t: t[1])
labels = [f[0] for f in feats]
vals = [f[1] for f in feats]
overall = R["churn_rate"]
cols = [SALMON if v > overall else GRAY for v in vals]
bars = ax.barh(labels, vals, height=0.62, color=cols, zorder=3)
for b, v in zip(bars, vals):
    ax.text(v + 0.008, b.get_y() + b.get_height()/2, pct(v),
            va="center", ha="left", fontsize=9, color=INK2)
ax.axvline(overall, color=NAVY, lw=1.2, zorder=4)
ax.text(overall + 0.012, 0.55, f"overall {pct(overall,1)}", fontsize=8.5,
        color=NAVY, ha="left")
ax.set_xlim(0, 0.56)
ax.set_xticks([0, .1, .2, .3, .4, .5])
ax.set_xticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
ax.set_title("Churn rate by customer attribute")
style_ax(ax, xgrid=True, ygrid=False)

# (b) tenure distribution by outcome
ax = fig.add_subplot(gs[0, 1])
e = R["tenure_hist"]["edges"]
centers = [(e[i]+e[i+1])/2 for i in range(len(e)-1)]
w = (e[1]-e[0]) * 0.42
ax.bar([c - w/2 - 0.35 for c in centers], R["tenure_hist"]["retained"], width=w,
       color=BLUE, label="Retained", zorder=3)
ax.bar([c + w/2 + 0.35 for c in centers], R["tenure_hist"]["churned"], width=w,
       color=SALMON, label="Churned", zorder=3)
ax.set_title("Tenure distribution by outcome")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Customers")
ax.legend(loc="upper center")
med_c, med_r = R["median_tenure_churned"], R["median_tenure_retained"]
ax.annotate(f"median churn: {med_c:.0f} mo", xy=(med_c, 660), fontsize=9,
            color=INK2, ha="left",
            xytext=(20, 720), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
style_ax(ax)

# (c) top churn correlates — diverging bars
ax = fig.add_subplot(gs[1, 0])
corr = R["churn_correlations"]
name_map = {
    "InternetService_Fiber optic": "Fiber-optic internet",
    "PaymentMethod_Electronic check": "Electronic check",
    "MonthlyCharges": "Monthly charges",
    "PaperlessBilling_Yes": "Paperless billing",
    "SeniorCitizen_Yes": "Senior citizen",
    "tenure": "Tenure",
    "Contract_Two year": "Two-year contract",
    "OnlineSecurity_Yes": "Online security",
    "TechSupport_Yes": "Tech support",
    "Contract_One year": "One-year contract",
    "Dependents_Yes": "Has dependents",
    "TotalCharges": "Total charges",
}
picks_pos = ["InternetService_Fiber optic", "PaymentMethod_Electronic check",
             "MonthlyCharges", "PaperlessBilling_Yes", "SeniorCitizen_Yes"]
picks_neg = ["tenure", "Contract_Two year", "OnlineSecurity_Yes",
             "TechSupport_Yes", "Contract_One year"]
items = [(name_map[k], corr[k]) for k in picks_pos + picks_neg]
items.sort(key=lambda t: t[1])
labs = [i[0] for i in items]
vs = [i[1] for i in items]
cols = [SALMON if v > 0 else BLUE for v in vs]
bars = ax.barh(labs, vs, height=0.62, color=cols, zorder=3)
for b, v in zip(bars, vs):
    ax.text(v + (0.012 if v > 0 else -0.012), b.get_y() + b.get_height()/2,
            f"{v:+.2f}", va="center", ha="left" if v > 0 else "right",
            fontsize=8.6, color=INK2)
ax.axvline(0, color=BASE, lw=1.1)
ax.set_xlim(-0.47, 0.42)
ax.set_title("Correlation with churn (point-biserial)")
style_ax(ax, xgrid=True, ygrid=False)

# (d) monthly charges distribution by outcome
ax = fig.add_subplot(gs[1, 1])
e = R["charges_hist"]["edges"]
centers = [(e[i]+e[i+1])/2 for i in range(len(e)-1)]
w = (e[1]-e[0]) * 0.42
ax.bar([c - w/2 - 0.55 for c in centers], R["charges_hist"]["retained"], width=w,
       color=BLUE, label="Retained", zorder=3)
ax.bar([c + w/2 + 0.55 for c in centers], R["charges_hist"]["churned"], width=w,
       color=SALMON, label="Churned", zorder=3)
ax.set_title("Monthly charges by outcome")
ax.set_xlabel("Monthly charges ($)")
ax.set_ylabel("Customers")
ax.set_ylim(0, 1420)
ax.legend(loc="upper right")
style_ax(ax)

save(fig, "exploratory-analysis.png")

# ======================================================================
# 2. MODEL RESULTS — benchmark + confusion + ROC
# ======================================================================
fig = new_fig()
gs = GridSpec(2, 5, figure=fig, hspace=0.5, wspace=1.15,
              left=0.16, right=0.955, top=0.92, bottom=0.10)

bench = R["benchmark"]
order = sorted(bench, key=lambda k: bench[k]["test_accuracy"])
winner = R["winner"]

# (a) model comparison — full-height left panel
ax = fig.add_subplot(gs[:, :3])
vals = [bench[k]["test_accuracy"] for k in order]
cols = [GOLD if k == winner else GRAY for k in order]
bars = ax.barh(order, vals, height=0.6, color=cols, zorder=3)
for b, k, v in zip(bars, order, vals):
    weight = "bold" if k == winner else "normal"
    ax.text(v + 0.004, b.get_y() + b.get_height()/2, pct(v, 1),
            va="center", ha="left", fontsize=9.5, color=INK, fontweight=weight)
    ax.text(0.008, b.get_y() + b.get_height()/2,
            f"AUC {bench[k]['roc_auc']:.3f}", va="center", ha="left",
            fontsize=8, color=SURFACE if k == winner else INK2, alpha=0.95)
ax.set_xlim(0, 0.92)
ax.set_xticks([0, .2, .4, .6, .8])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"])
ax.set_title("Classifier benchmark — holdout accuracy (n=1,409)")
if R.get("winner_threshold"):
    ax.text(0, -0.085, "Stacked ensemble scored at its CV-tuned threshold "
            f"({R['winner_threshold']:.2f}); all others at 0.50.",
            transform=ax.transAxes, fontsize=8, color=MUTED)
style_ax(ax, xgrid=True, ygrid=False)

# (b) confusion matrix (winner)
ax = fig.add_subplot(gs[0, 3:])
cm = np.array(R["winner_confusion"])
cmn = cm / cm.sum()
seq = ["#e6edf6", "#b9cde6", "#7ba3cd", "#3a6fa5"]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq", ["#eef2f8", "#123c66"])
ax.imshow(cmn, cmap=cmap, vmin=0, vmax=0.72, aspect="auto")
for i in range(2):
    for j in range(2):
        share = cmn[i, j]
        color = "#ffffff" if share > 0.35 else INK
        lab = ["True negative", "False positive", "False negative", "True positive"][i*2+j]
        ax.text(j, i - 0.13, f"{cm[i, j]:,}", ha="center", va="center",
                fontsize=15, fontweight="semibold", color=color)
        ax.text(j, i + 0.20, f"{lab} · {share*100:.1f}%", ha="center",
                va="center", fontsize=8, color=color)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: stay", "Pred: churn"])
ax.set_yticklabels(["Stayed", "Churned"])
ax.set_title(f"Confusion matrix — {winner.lower()}")
ax.grid(visible=False)
ax.tick_params(length=0)

# (c) ROC curves — winner emphasized, rest context
ax = fig.add_subplot(gs[1, 3:])
for name in ["Random Forest", "Gradient Boosting", "AdaBoost", "KNN", "SVM (RBF)"]:
    rc = R["roc_curves"][name]
    ax.plot(rc["fpr"], rc["tpr"], color=GRAY, lw=1.1, alpha=0.75, zorder=2)
rc = R["roc_curves"][winner]
ax.plot(rc["fpr"], rc["tpr"], color=SALMON, lw=2.4, zorder=4,
        solid_capstyle="round")
ax.plot([0, 1], [0, 1], color=BASE, lw=1.0, ls=(0, (1, 2)), zorder=1)
ax.text(0.30, 0.83, f"{winner}\nAUC {bench[winner]['roc_auc']:.3f}",
        fontsize=9.5, color=INK, va="top")
ax.text(0.62, 0.47, "other models", fontsize=8.5, color=MUTED)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate")
ax.set_title("ROC — holdout set")
style_ax(ax)

save(fig, "model-results.png")

# ======================================================================
# 3. RETENTION DASHBOARD — segment KM + summary + interventions
# ======================================================================
fig = new_fig()
gs = GridSpec(2, 5, figure=fig, hspace=0.55, wspace=1.6,
              left=0.07, right=0.96, top=0.92, bottom=0.10)

SEG_COLORS = {"New · High spend": SALMON, "New · Basic service": GOLD,
              "Established · Premium": BLUE, "Loyal · Value plans": VIOLET}
seg_order = ["New · High spend", "New · Basic service",
             "Established · Premium", "Loyal · Value plans"]

# (a) KM by segment — left tall
ax = fig.add_subplot(gs[:, :3])
plot_km(ax, R["km_segments"], SEG_COLORS, lw=2.2, order=seg_order)
ends = {n: R["km_segments"][n][-1] for n in seg_order}
offsets = {"New · High spend": 0, "New · Basic service": 0,
           "Established · Premium": 0.02, "Loyal · Value plans": 0}
for n in seg_order:
    ax.text(73.5, ends[n] + offsets.get(n, 0) - 0.012, n, fontsize=9.2,
            color=INK2, va="center")
ax.set_xlim(0, 106)
ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
ax.set_title("Kaplan-Meier retention by customer segment")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Share of cohort retained")
ax.legend(handles=[plt.Line2D([], [], color=SEG_COLORS[n], lw=2.2, label=n)
                   for n in seg_order], loc="lower left", fontsize=8.6)
style_ax(ax)

# (b) segment churn + revenue at risk
ax = fig.add_subplot(gs[0, 3:])
segs = R["segments"]
rates = [segs[n]["churn_rate"] for n in seg_order]
bars = ax.barh(seg_order[::-1], rates[::-1], height=0.6,
               color=[SEG_COLORS[n] for n in seg_order[::-1]], zorder=3)
for b, n in zip(bars, seg_order[::-1]):
    v = segs[n]["churn_rate"]
    ax.text(v + 0.012, b.get_y() + b.get_height()/2,
            f"{pct(v,1)} · ${segs[n]['monthly_revenue_churned']/1000:.0f}K/mo lost",
            va="center", ha="left", fontsize=8.6, color=INK2)
ax.set_xlim(0, 0.75)
ax.set_xticks([0, .2, .4, .6])
ax.set_xticklabels(["0%", "20%", "40%", "60%"])
ax.set_title("Churn rate and revenue lost by segment")
style_ax(ax, xgrid=True, ygrid=False)

# (c) interventions — A/B tested strategies by simulated lift
ax = fig.add_subplot(gs[1, 3:])
ab = sorted(R["ab_tests"], key=lambda s: s["relative_lift"])
names = {"Discount Offer": "Discount offer",
         "Free Premium Technical Support": "Free tech support",
         "Free Service Upgrade": "Service upgrade",
         "Contract Incentive": "Contract incentive",
         "Enhanced Loyalty Program": "Loyalty program"}
labs = [names[s["strategy_name"]] for s in ab]
lifts = [s["relative_lift"] for s in ab]
rois = [s["estimated_roi"] for s in ab]
cols = [GOLD if r == max(rois) else GRAY for r in rois]
bars = ax.barh(labs, lifts, height=0.6, color=cols, zorder=3)
for b, s in zip(bars, ab):
    ax.text(s["relative_lift"] + 0.5, b.get_y() + b.get_height()/2,
            f"+{s['relative_lift']:.1f}% · ROI {s['estimated_roi']:.0f}%",
            va="center", ha="left", fontsize=8.6, color=INK2)
ax.set_xlim(0, 40)
ax.set_title("A/B-tested retention lift (simulated)")
ax.set_xlabel("Relative retention lift (%)")
style_ax(ax, xgrid=True, ygrid=False)

save(fig, "retention-dashboard.png")

# ======================================================================
# 4. HERO — 2x2 montage of strongest visuals
# ======================================================================
fig = new_fig()
gs = GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.34,
              left=0.125, right=0.965, top=0.93, bottom=0.09)

# (a) KM by contract — the money chart
ax = fig.add_subplot(gs[0, 0])
CONTRACT_COLORS = {"Two year": BLUE, "One year": VIOLET,
                   "Month-to-month": SALMON}
km_c = {k: v for k, v in R["km_contract"].items() if k != "All customers"}
plot_km(ax, km_c, CONTRACT_COLORS, lw=2.2,
        order=["Two year", "One year", "Month-to-month"])
for n, lab in [("Two year", "Two-year"), ("One year", "One-year"),
               ("Month-to-month", "Month-to-month")]:
    ax.text(73.5, km_c[n][-1] - 0.012, lab, fontsize=9, color=INK2)
ax.set_xlim(0, 102)
ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
ax.set_title("Retention by contract type (Kaplan-Meier)")
ax.set_ylabel("Share retained")
ax.set_xlabel("Tenure (months)")
ax.legend(handles=[plt.Line2D([], [], color=CONTRACT_COLORS[n], lw=2.2,
                              label=l) for n, l in
                   [("Two year", "Two-year"), ("One year", "One-year"),
                    ("Month-to-month", "Month-to-month")]],
          loc="lower left", fontsize=8.6)
style_ax(ax)

# (b) churn by tenure cohort
ax = fig.add_subplot(gs[0, 1])
tb = R["churn_by_tenure_band"]
names = list(tb.keys())
rates = [tb[n]["rate"] for n in names]
cols = [SALMON if r > R["churn_rate"] else GRAY for r in rates]
bars = ax.bar(names, rates, width=0.56, color=cols, zorder=3)
for b, r in zip(bars, rates):
    ax.text(b.get_x() + b.get_width()/2, r + 0.012, pct(r, 1), ha="center",
            fontsize=9.5, color=INK)
ax.axhline(R["churn_rate"], color=NAVY, lw=1.2, zorder=4)
ax.text(3.42, R["churn_rate"] + 0.012, f"overall {pct(R['churn_rate'],1)}",
        fontsize=8.5, color=NAVY, ha="right")
ax.set_ylim(0, 0.56)
ax.set_yticks([0, .1, .2, .3, .4, .5])
ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
ax.set_title("Churn rate by tenure cohort")
style_ax(ax)

# (c) model benchmark top 5
ax = fig.add_subplot(gs[1, 0])
top5 = sorted(bench, key=lambda k: -bench[k]["test_accuracy"])[:5][::-1]
vals = [bench[k]["test_accuracy"] for k in top5]
cols = [GOLD if k == winner else GRAY for k in top5]
bars = ax.barh(top5, vals, height=0.58, color=cols, zorder=3)
for b, k, v in zip(bars, top5, vals):
    ax.text(v + 0.004, b.get_y() + b.get_height()/2, pct(v, 1), va="center",
            fontsize=9.5, color=INK,
            fontweight="bold" if k == winner else "normal")
ax.set_xlim(0, 0.92)
ax.set_xticks([0, .2, .4, .6, .8])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"])
ax.set_title("Top classifiers — holdout accuracy")
style_ax(ax, xgrid=True, ygrid=False)

# (d) revenue at risk by segment
ax = fig.add_subplot(gs[1, 1])
rev = [segs[n]["monthly_revenue_churned"]/1000 for n in seg_order]
bars = ax.bar([n.replace(" · ", "\n") for n in seg_order], rev, width=0.56,
              color=[SEG_COLORS[n] for n in seg_order], zorder=3)
for b, v in zip(bars, rev):
    ax.text(b.get_x() + b.get_width()/2, v + 1.6, f"${v:.0f}K", ha="center",
            fontsize=9.5, color=INK)
ax.set_ylim(0, 104)
ax.set_ylabel("Monthly revenue churned ($K)")
ax.set_title("Revenue lost to churn by segment")
style_ax(ax)

save(fig, "hero.png")

# ---------------------------------------------------------------- verify
from PIL import Image
for f in ["hero.png", "exploratory-analysis.png", "model-results.png",
          "retention-dashboard.png"]:
    p = os.path.join(OUT, f)
    im = Image.open(p)
    print(f, im.size, f"{os.path.getsize(p)/1024:.0f}KB")

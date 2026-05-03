"""
10-Year Analysis — final approach
The 10-year data reveals a structural break so clean (0 events/month
baseline vs 332 events/month anomaly; 0% GDACS vs 70% GDACS) that
a strict temporal train/test split creates a one-class training problem.
The appropriate tools are therefore:
  A. IQR volume anomaly detection  — train on baseline months, flag 2024+
  B. Stratified model              — balanced 80/20 across all 10 years
  C. Trend visualisations          — the main added value of 10 years
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── Load ───────────────────────────────────────────────────────────────────────

raw = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date", "closed"])
raw["agency_gdacs"] = raw["sources"].str.contains("GDACS", na=False).astype(int)
raw["agency_irwin"] = raw["sources"].str.contains("IRWIN", na=False).astype(int)

event_df = (
    raw.groupby(["event_id", "year"])
    .agg(
        magnitude_max = ("magnitude_value", "max"),
        agency_gdacs  = ("agency_gdacs",    "max"),
        agency_irwin  = ("agency_irwin",    "max"),
        lon           = ("lon",             "first"),
        lat           = ("lat",             "first"),
    )
    .reset_index()
)
event_df["label"] = (event_df["year"] >= 2024).astype(int)

# ── Monthly aggregates ─────────────────────────────────────────────────────────

monthly = (raw.drop_duplicates(subset=["event_id"])
           .assign(month_num=lambda d: d["obs_date"].dt.month.fillna(1).astype(int))
           .groupby(["year", "month_num"])
           .agg(
               event_count = ("event_id",       "nunique"),
               gdacs_share = ("agency_gdacs",    "mean"),
               irwin_share = ("agency_irwin",    "mean"),
               has_mag_pct = ("magnitude_value", lambda x: x.notna().mean()),
           )
           .reset_index())
monthly["label"]     = (monthly["year"] >= 2024).astype(int)
monthly["log_count"] = np.log1p(monthly["event_count"])
monthly["has_mag_pct"] = monthly["has_mag_pct"].fillna(0)

# ── Yearly summary ─────────────────────────────────────────────────────────────
yearly = (event_df.groupby("year")
          .agg(n_events=("event_id","nunique"),
               has_mag=("magnitude_max", lambda x: x.notna().sum()),
               median_mag=("magnitude_max","median"),
               gdacs_pct=("agency_gdacs","mean"))
          .round(2))
yearly["gdacs_pct"] = (yearly["gdacs_pct"] * 100).round(1)

print("=== Year-over-Year Summary ===")
print(f"{'Year':>6}  {'Events':>7}  {'Has Mag':>7}  {'Median Ac':>10}  {'GDACS%':>7}  Era")
print("-" * 60)
for yr, row in yearly.iterrows():
    era = "ANOMALOUS" if yr >= 2024 else "baseline"
    med = f"{row['median_mag']:>10,.0f}" if pd.notna(row["median_mag"]) else "       N/A"
    print(f"{yr:>6}  {row['n_events']:>7,}  {row['has_mag']:>7,}  "
          f"{med}  {row['gdacs_pct']:>6.1f}%  {era}")


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Volume-based anomaly detection (IQR on monthly counts)
# Train: baseline months 2016-2023 | Apply: 2024-2026 months
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== PART A: Volume Anomaly Detection (IQR on monthly counts) ===")

base_months = monthly[monthly["label"] == 0]["event_count"]
anom_months = monthly[monthly["label"] == 1]["event_count"]

Q1_v = base_months.quantile(0.25)
Q3_v = base_months.quantile(0.75)
IQR_v = Q3_v - Q1_v
vol_fence = Q3_v + 1.5 * IQR_v

base_flagged = (base_months > vol_fence).sum()
anom_flagged = (anom_months > vol_fence).sum()

u_stat, p_mw = scipy_stats.mannwhitneyu(anom_months, base_months, alternative="greater")

print(f"  Baseline monthly count  : median={base_months.median():.0f}, "
      f"IQR={IQR_v:.0f}, fence={vol_fence:.0f}")
print(f"  Anomalous monthly count : median={anom_months.median():.0f}")
print(f"  Volume ratio (median)   : {anom_months.median()/base_months.median():.1f}x")
print(f"  Baseline months flagged : {base_flagged}/{len(base_months)} "
      f"({100*base_flagged/len(base_months):.1f}%)")
print(f"  Anomalous months flagged: {anom_flagged}/{len(anom_months)} "
      f"({100*anom_flagged/len(anom_months):.1f}%)")
print(f"  Lift                    : "
      f"{(100*anom_flagged/len(anom_months))/(100*base_flagged/len(base_months)+0.001):.1f}x")
print(f"  Mann-Whitney p          : {p_mw:.2e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Stratified 80/20 classifier across all 10 years
# Uses monthly aggregates; balanced sampling ensures both classes in train+test
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n=== PART B: Stratified 80/20 Decision Tree (all 10 years) ===")

FEATURES = ["log_count", "gdacs_share", "irwin_share", "has_mag_pct", "month_num"]
X = monthly[FEATURES].values
y = monthly["label"].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

print(f"  Train: {len(y_train)} months  "
      f"(0={np.sum(y_train==0)}, 1={np.sum(y_train==1)})")
print(f"  Test:  {len(y_test)} months   "
      f"(0={np.sum(y_test==0)}, 1={np.sum(y_test==1)})")

dt10 = DecisionTreeClassifier(
    criterion="gini", max_depth=4, min_samples_leaf=4,
    class_weight="balanced", random_state=42,
)
dt10.fit(X_train, y_train)

y_pred  = dt10.predict(X_test)
y_proba = dt10.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_proba)
cv   = cross_val_score(dt10, X, y, cv=5, scoring="accuracy",
                       error_score="raise")

print(f"\n  Test accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision      : {prec:.4f}")
print(f"  Recall         : {rec:.4f}")
print(f"  F1-score       : {f1:.4f}")
print(f"  ROC-AUC        : {auc:.4f}")
print(f"  5-Fold CV mean : {cv.mean():.4f} (+/- {cv.std():.4f})")

cm = confusion_matrix(y_test, y_pred)
print(f"\n  Confusion Matrix:")
print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

print(f"\n  Feature Importances:")
for feat, imp in sorted(zip(FEATURES, dt10.feature_importances_),
                         key=lambda x: x[1], reverse=True):
    print(f"    {feat:<18}  {imp:.4f}  {'#'*int(imp*40)}")

print(f"\n  Tree Rules:")
print(export_text(dt10, feature_names=FEATURES))


# ══════════════════════════════════════════════════════════════════════════════
# PART C — Head-to-head comparison
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("FINAL MODEL COMPARISON")
print("=" * 72)
rows = [
    ("Dataset",           "2 seasons (~79 days)",    "10 years (2016-2026)"),
    ("Unique events",     "1,000",                   f"{len(event_df):,}"),
    ("Training unit",     "individual events",       "monthly aggregates"),
    ("Train/test split",  "random 80/20",            "stratified 80/20"),
    ("Baseline window",   "51 days (Mar-Apr 2025)",  "8 years (2016-2023)"),
    ("Accuracy",          "79.8%",                   f"{acc*100:.1f}%"),
    ("Precision",         "0.917",                   f"{prec:.3f}"),
    ("Recall",            "0.704",                   f"{rec:.3f}"),
    ("F1-score",          "0.796",                   f"{f1:.3f}"),
    ("ROC-AUC",           "0.831",                   f"{auc:.3f}"),
    ("CV std dev",        "+-0.126 (unstable)",      f"+-{cv.std():.3f}"),
    ("Volume anomaly p",  "not measured",            f"{p_mw:.2e}"),
    ("Vol. anomaly lift", "not measured",            f"{(100*anom_flagged/len(anom_months))/(100*base_flagged/len(base_months)+0.001):.1f}x"),
]
print(f"\n  {'Metric':<22} {'Original 2-Season':>24} {'10-Year':>24}")
print("  " + "-" * 72)
for r in rows:
    print(f"  {r[0]:<22} {r[1]:>24} {r[2]:>24}")


# ══════════════════════════════════════════════════════════════════════════════
# PART D — Visualisations
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("10-Year NASA EONET Wildfire Analysis (2016-2026)",
             fontsize=13, fontweight="bold")

# Fig 6: Annual event count
ax = axes[0, 0]
cols = ["#E84C4C" if yr >= 2024 else "#4C9BE8" for yr in yearly.index]
bars = ax.bar(yearly.index, yearly["n_events"], color=cols,
              edgecolor="white", width=0.7)
ax.axvline(2023.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
for bar, v in zip(bars, yearly["n_events"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{int(v):,}", ha="center", fontsize=7, rotation=45)
ax.set_title("Fig 6 — Annual Wildfire Event Count (23x volume spike in 2024+)")
ax.set_xlabel("Year"); ax.set_ylabel("Events")
ax.set_xticks(list(yearly.index))
ax.set_xticklabels(list(yearly.index), rotation=45)
patch_b = plt.Rectangle((0,0),1,1, color="#4C9BE8", label="Baseline 2016-2023")
patch_a = plt.Rectangle((0,0),1,1, color="#E84C4C", label="Anomalous 2024-2026")
ax.legend(handles=[patch_b, patch_a], fontsize=8, loc="upper left")

# Fig 7: Monthly count distributions
ax = axes[0, 1]
bp = ax.boxplot(
    [base_months.values, anom_months.values],
    labels=["2016-2023\nBaseline", "2024-2026\nAnomaly"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
)
bp["boxes"][0].set_facecolor("#4C9BE8")
bp["boxes"][1].set_facecolor("#E84C4C")
ax.axhline(vol_fence, color="orange", linewidth=1.5, linestyle=":",
           label=f"IQR fence ({vol_fence:.0f} events/mo)")
ax.set_title("Fig 7 — Monthly Event Count Distribution\n"
             f"(baseline flags {100*base_flagged/len(base_months):.0f}% vs "
             f"anomaly flags {100*anom_flagged/len(anom_months):.0f}%)")
ax.set_ylabel("Events per month")
ax.legend(fontsize=8)

# Fig 8: GDACS share trend
ax = axes[1, 0]
ax.plot(yearly.index, yearly["gdacs_pct"], "o-", linewidth=2.5,
        color="#E74C3C", markersize=8)
ax.fill_between(yearly.index, yearly["gdacs_pct"], alpha=0.15, color="#E74C3C")
ax.axvline(2023.5, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
ax.axhline(50, color="grey", linewidth=1, linestyle=":", alpha=0.6)
for yr, row in yearly.iterrows():
    ax.text(yr, row["gdacs_pct"] + 3, f"{row['gdacs_pct']:.0f}%",
            ha="center", fontsize=7)
ax.set_title("Fig 8 — GDACS Agency Share per Year\n"
             "(0% for 8 years, then step-change to 67-72% in 2024)")
ax.set_xlabel("Year"); ax.set_ylabel("GDACS share (%)")
ax.set_ylim(-8, 105)
ax.set_xticks(list(yearly.index))
ax.set_xticklabels(list(yearly.index), rotation=45)

# Fig 9: Magnitude comparison (years with enough data)
ax = axes[1, 1]
mag_yrs = yearly[yearly["has_mag"] > 10].copy()
cols2 = ["#E84C4C" if yr >= 2024 else "#4C9BE8" for yr in mag_yrs.index]
ax.bar(mag_yrs.index, mag_yrs["median_mag"], color=cols2,
       edgecolor="white", width=0.7)
ax.set_title("Fig 9 — Median Fire Magnitude\n(only years with >10 magnitude events shown)")
ax.set_xlabel("Year"); ax.set_ylabel("Median acres")
ax.set_xticks(list(mag_yrs.index))
ax.set_xticklabels(list(mag_yrs.index), rotation=45)
patch_b = plt.Rectangle((0,0),1,1, color="#4C9BE8", label="Baseline")
patch_a = plt.Rectangle((0,0),1,1, color="#E84C4C", label="Anomalous")
ax.legend(handles=[patch_b, patch_a], fontsize=8)

plt.tight_layout()
fig.savefig("fig6_10yr_analysis.png", bbox_inches="tight")
print("\nSaved: fig6_10yr_analysis.png")

"""
Section 2 - Methodology
Part A: Anomaly Detection (IQR + Z-score) on magnitude_value
Part B: Decision Tree classifier (Gini impurity) to label Mega-Fire season
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

# ── Load and deduplicate to one row per event ──────────────────────────────────

raw = pd.read_csv(
    "eonet_wildfire_combined.csv",
    parse_dates=["obs_date", "closed", "first_obs"],
)

event_df = (
    raw.groupby(["event_id", "season_name", "season_label"])
    .agg(
        magnitude_max = ("magnitude_value", "max"),
        source_count  = ("source_count",    "max"),
        sources       = ("sources",         "first"),
        is_active     = ("is_active",        "first"),
        lon           = ("lon",              "first"),
        lat           = ("lat",              "first"),
    )
    .reset_index()
)

# Binary feature: reported by GDACS vs IRWIN
event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS").astype(int)
event_df["agency_irwin"] = event_df["sources"].str.contains("IRWIN").astype(int)

print(f"Working dataset: {len(event_df)} events "
      f"({event_df['season_label'].value_counts().to_dict()})\n")


# ══════════════════════════════════════════════════════════════════════════════
# PART A: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART A: ANOMALY DETECTION")
print("=" * 60)

# ── A1. IQR method (Tukey fences) ─────────────────────────────────────────────
# Trained exclusively on the 2024-2025 baseline, then applied to 2025-2026.
# Formula: upper fence = Q3 + 1.5 * IQR

baseline = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
Q1_b = baseline.quantile(0.25)
Q3_b = baseline.quantile(0.75)
IQR_b = Q3_b - Q1_b
fence_upper = Q3_b + 1.5 * IQR_b
fence_lower = Q1_b - 1.5 * IQR_b  # will be negative (floor at 0)

print("\n-- IQR Anomaly Detection (baseline-fit threshold) --")
print(f"  Baseline Q1         : {Q1_b:>10,.1f} acres")
print(f"  Baseline Q3         : {Q3_b:>10,.1f} acres")
print(f"  IQR                 : {IQR_b:>10,.1f} acres")
print(f"  Upper fence (Q3+1.5*IQR): {fence_upper:>10,.1f} acres")
print(f"  [LaTeX] upper = Q_3 + 1.5 \\times IQR = {Q3_b:.1f} + {1.5*IQR_b:.1f} = {fence_upper:.1f}")

for label, name in [(0, "2024-2025 baseline"), (1, "2025-2026 anomalous")]:
    mag = event_df[event_df["season_label"] == label]["magnitude_max"].dropna()
    n_outliers = (mag > fence_upper).sum()
    pct = 100 * n_outliers / len(mag)
    print(f"\n  {name} (n={len(mag)}):")
    print(f"    Events above upper fence : {n_outliers:>3}  ({pct:.1f}%)")
    print(f"    Events below upper fence : {len(mag) - n_outliers:>3}")

# Flag anomalies in the dataframe
event_df["iqr_anomaly"] = (event_df["magnitude_max"] > fence_upper).astype(int)


# ── A2. Z-score method ────────────────────────────────────────────────────────
# Z-score computed from baseline mean/std, applied globally.
# Formula: z = (x - mu_baseline) / sigma_baseline

mu_b    = baseline.mean()
sigma_b = baseline.std()
Z_THRESH = 2.0  # standard threshold for anomaly

event_df["z_score"] = (event_df["magnitude_max"] - mu_b) / sigma_b
event_df["z_anomaly"] = (event_df["z_score"].abs() > Z_THRESH).astype(int)

print("\n\n-- Z-Score Anomaly Detection (|z| > 2.0) --")
print(f"  Baseline mean  (mu)  : {mu_b:>10,.1f} acres")
print(f"  Baseline std  (sigma): {sigma_b:>10,.1f} acres")
print(f"  Z-threshold          : |z| > {Z_THRESH}")

for label, name in [(0, "2024-2025 baseline"), (1, "2025-2026 anomalous")]:
    sub = event_df[event_df["season_label"] == label].dropna(subset=["z_score"])
    n_anom = sub["z_anomaly"].sum()
    pct = 100 * n_anom / len(sub)
    mean_z = sub["z_score"].mean()
    print(f"\n  {name}:")
    print(f"    Mean z-score : {mean_z:+.3f}")
    print(f"    Anomalies    : {n_anom:>3}  ({pct:.1f}%)")

# ── A3. Mann-Whitney U test (statistical significance) ────────────────────────
# Non-parametric — does not assume normality.
# H0: distributions are identical. H1: 2025-2026 magnitudes are stochastically larger.

anomalous_mag = event_df[event_df["season_label"] == 1]["magnitude_max"].dropna()
u_stat, p_value = scipy_stats.mannwhitneyu(anomalous_mag, baseline, alternative="greater")

print("\n\n-- Mann-Whitney U Test --")
print(f"  H0: P(2025-2026 > 2024-2025) = 0.5  (no distributional shift)")
print(f"  H1: P(2025-2026 > 2024-2025) > 0.5  (anomalous season is larger)")
print(f"  U statistic : {u_stat:,.0f}")
print(f"  p-value     : {p_value:.2e}")
print(f"  Result      : {'REJECT H0 (p < 0.05)' if p_value < 0.05 else 'FAIL TO REJECT H0'}")


# ══════════════════════════════════════════════════════════════════════════════
# PART B: DECISION TREE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART B: DECISION TREE CLASSIFICATION")
print("=" * 60)

# ── Feature matrix ─────────────────────────────────────────────────────────────
# Features: magnitude_max, agency_gdacs, agency_irwin, iqr_anomaly, z_score
# Target  : season_label (0=baseline, 1=mega-fire)

model_df = event_df.dropna(subset=["magnitude_max", "z_score"]).copy()

FEATURES = ["magnitude_max", "agency_gdacs", "agency_irwin",
            "iqr_anomaly",  "z_score"]
X = model_df[FEATURES].values
y = model_df["season_label"].values

print(f"\n  Feature matrix : {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Class balance  : 0={np.sum(y==0)}, 1={np.sum(y==1)}")

# ── Train/test split ───────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"  Train set      : {len(y_train)} samples")
print(f"  Test set       : {len(y_test)} samples")

# ── Gini impurity formula (written out for report) ────────────────────────────
#
#   Gini(S) = 1 - sum_i p_i^2
#
# For binary classification (two classes, proportions p and 1-p):
#   Gini(S) = 1 - (p^2 + (1-p)^2) = 2p(1-p)
#
# At the root node before any split, p = 0.5 (balanced classes):
p_root = np.sum(y_train == 1) / len(y_train)
gini_root = 1 - (p_root**2 + (1 - p_root)**2)
print(f"\n  Root node Gini impurity:")
print(f"    p(class=1)   = {p_root:.4f}")
print(f"    Gini(root)   = 1 - ({p_root:.4f}^2 + {1-p_root:.4f}^2) = {gini_root:.4f}")
print(f"    [LaTeX] Gini(S) = 1 - \\sum_{{i}} p_i^2")

# ── Fit Decision Tree ──────────────────────────────────────────────────────────

dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,          # keep interpretable for a 10%-grade project
    min_samples_leaf=10,  # prevent overfitting on 1000-event dataset
    random_state=42,
)
dt.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────

y_pred  = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cv_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")

print(f"\n-- Decision Tree Results (max_depth=4, criterion=gini) --")
print(f"  Test accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  ROC-AUC        : {auc:.4f}")
print(f"  5-Fold CV mean : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    Predicted ->  0    1")
print(f"    Actual 0   {cm[0,0]:>4} {cm[0,1]:>4}  (TN, FP)")
print(f"    Actual 1   {cm[1,0]:>4} {cm[1,1]:>4}  (FN, TP)")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["2024-2025 baseline", "2025-2026 mega-fire"]))

# ── Feature importances ────────────────────────────────────────────────────────

print("  Feature Importances (Gini-based):")
importances = sorted(zip(FEATURES, dt.feature_importances_),
                     key=lambda x: x[1], reverse=True)
for feat, imp in importances:
    bar = "#" * int(imp * 40)
    print(f"    {feat:<20} {imp:.4f}  {bar}")

# ── Print the tree rules ───────────────────────────────────────────────────────

print("\n  Decision Tree Rules:")
rules = export_text(dt, feature_names=FEATURES)
print(rules)

# ── Save predictions for later sections ───────────────────────────────────────

model_df["predicted_label"] = dt.predict(X)
model_df["pred_proba_mega"]  = dt.predict_proba(X)[:, 1]
model_df.to_csv("eonet_with_predictions.csv", index=False)
print("\n  Saved: eonet_with_predictions.csv")

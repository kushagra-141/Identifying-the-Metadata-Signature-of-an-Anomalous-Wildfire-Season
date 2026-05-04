"""
Applies anomaly detection (IQR and z-score) and a Decision Tree classifier
to distinguish the 2024-2025 baseline from the 2025-2026 anomalous season.
Season windows are extracted from the decade dataset by obs_date.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

raw = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date", "closed"])

raw.loc[(raw["obs_date"] >= "2024-05-01") & (raw["obs_date"] <= "2025-04-30"), "season_label"] = 0
raw.loc[(raw["obs_date"] >= "2025-05-01") & (raw["obs_date"] <= "2026-04-30"), "season_label"] = 1
season_raw = raw[raw["season_label"].notna()].copy()
season_raw["season_label"] = season_raw["season_label"].astype(int)
season_raw["season_name"]  = season_raw["season_label"].map({0: "2024_2025", 1: "2025_2026"})

event_df = (
    season_raw.groupby(["event_id", "season_name", "season_label"])
    .agg(
        magnitude_max = ("magnitude_value", "max"),
        source_count  = ("source_count",    "max"),
        sources       = ("sources",         "first"),
        is_active     = ("is_active",       "first"),
        lon           = ("lon",             "first"),
        lat           = ("lat",             "first"),
    )
    .reset_index()
)
event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS", na=False).astype(int)
event_df["agency_irwin"] = event_df["sources"].str.contains("IRWIN", na=False).astype(int)

print(f"dataset: {len(event_df)} events  (0={event_df['season_label'].eq(0).sum()}, 1={event_df['season_label'].eq(1).sum()})")

# IQR threshold derived from baseline only
baseline    = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
Q1_b        = baseline.quantile(0.25)
Q3_b        = baseline.quantile(0.75)
IQR_b       = Q3_b - Q1_b
fence_upper = Q3_b + 1.5 * IQR_b

print(f"\nIQR anomaly detection (baseline-derived threshold):")
print(f"  Q1={Q1_b:,.1f}  Q3={Q3_b:,.1f}  IQR={IQR_b:,.1f}  fence={fence_upper:,.1f} acres")
for label, name in [(0, "2024-2025"), (1, "2025-2026")]:
    mag     = event_df[event_df["season_label"] == label]["magnitude_max"].dropna()
    n_above = (mag > fence_upper).sum()
    print(f"  {name}: {n_above}/{len(mag)} above fence ({100*n_above/len(mag):.1f}%)")

event_df["iqr_anomaly"] = (event_df["magnitude_max"] > fence_upper).astype(int)

# Robust z-score using median and MAD instead of mean/std.
# The baseline contains extreme outliers (max 288,690 acres) that inflate
# the standard deviation to 12,100 acres, making the standard z-score
# flag more baseline events than anomalous ones. MAD is resistant to outliers.
median_b = baseline.median()
mad_b    = (baseline - median_b).abs().median()
event_df["z_score"]   = (event_df["magnitude_max"] - median_b) / (1.4826 * mad_b)
event_df["z_anomaly"] = (event_df["z_score"] > 2.0).astype(int)

print(f"\nRobust z-score  (median={median_b:,.1f}, MAD={mad_b:,.1f}, threshold z>2.0):")
for label, name in [(0, "2024-2025"), (1, "2025-2026")]:
    sub = event_df[event_df["season_label"] == label].dropna(subset=["z_score"])
    print(f"  {name}: mean_z={sub['z_score'].mean():+.3f}  anomalies={sub['z_anomaly'].sum()} ({100*sub['z_anomaly'].mean():.1f}%)")

anomalous_mag = event_df[event_df["season_label"] == 1]["magnitude_max"].dropna()
u_stat, p_value = scipy_stats.mannwhitneyu(anomalous_mag, baseline, alternative="greater")
print(f"\nMann-Whitney U test:")
print(f"  U={u_stat:,.0f}  p={p_value:.2e}  {'reject H0' if p_value < 0.05 else 'fail to reject H0'}")

model_df = event_df.dropna(subset=["magnitude_max", "z_score"]).copy()
# agency_gdacs and agency_irwin have perfect correlation (-1.0); keep only one.
# iqr_anomaly is derived directly from magnitude_max; redundant when magnitude_max is present.
FEATURES  = ["magnitude_max", "agency_irwin", "z_score"]
X = model_df[FEATURES].values
y = model_df["season_label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

p_root    = np.sum(y_train == 1) / len(y_train)
gini_root = 1 - (p_root**2 + (1 - p_root)**2)
print(f"\nDecision Tree  (max_depth=4, criterion=gini):")
print(f"  root Gini = {gini_root:.4f}  train={len(y_train)}  test={len(y_test)}")

dt = DecisionTreeClassifier(criterion="gini", max_depth=4, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)

y_pred    = dt.predict(X_test)
y_proba   = dt.predict_proba(X_test)[:, 1]
acc       = accuracy_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_proba)
cv_scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
cm        = confusion_matrix(y_test, y_pred)

print(f"  accuracy={acc:.4f}  ROC-AUC={auc:.4f}  CV={cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n{classification_report(y_test, y_pred, target_names=['baseline','mega-fire'])}")

print("  feature importances:")
for feat, imp in sorted(zip(FEATURES, dt.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"    {feat:<20} {imp:.4f}")

print(f"\n  tree rules:\n{export_text(dt, feature_names=FEATURES)}")

model_df["predicted_label"] = dt.predict(X)
model_df["pred_proba_mega"]  = dt.predict_proba(X)[:, 1]
model_df.to_csv("eonet_with_predictions.csv", index=False)
print("saved: eonet_with_predictions.csv")

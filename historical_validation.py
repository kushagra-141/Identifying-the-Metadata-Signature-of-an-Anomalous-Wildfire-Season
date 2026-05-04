"""
Tests whether the Mega-Fire Signature generalises in both temporal directions.
Exp A: baseline-trained model applied to 2025-2026 (forward)
Exp B: mega-fire-trained model applied to 2024-2025 (reverse)
Exp C: hand-crafted rule applied to both seasons
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split

raw = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date", "closed"])

raw.loc[(raw["obs_date"] >= "2024-05-01") & (raw["obs_date"] <= "2025-04-30"), "season_label"] = 0
raw.loc[(raw["obs_date"] >= "2025-05-01") & (raw["obs_date"] <= "2026-04-30"), "season_label"] = 1
season_raw = raw[raw["season_label"].notna()].copy()
season_raw["season_label"] = season_raw["season_label"].astype(int)

event_df = (
    season_raw.groupby(["event_id", "season_label"])
    .agg(
        magnitude_max = ("magnitude_value", "max"),
        sources       = ("sources",         "first"),
        lon           = ("lon",             "first"),
        lat           = ("lat",             "first"),
    )
    .reset_index()
)
event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS", na=False).astype(int)
event_df["agency_irwin"] = event_df["sources"].str.contains("IRWIN", na=False).astype(int)

base_mag = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
FENCE    = base_mag.quantile(0.75) + 1.5 * (base_mag.quantile(0.75) - base_mag.quantile(0.25))

median_b = base_mag.median()
mad_b    = (base_mag - median_b).abs().median()
event_df["z_score"]     = (event_df["magnitude_max"] - median_b) / (1.4826 * mad_b)
event_df["iqr_anomaly"] = (event_df["magnitude_max"] > FENCE).astype(int)

model_df = event_df.dropna(subset=["magnitude_max"]).copy()
FEATURES = ["magnitude_max", "agency_irwin", "z_score"]

BASE_df = model_df[model_df["season_label"] == 0]
ANOM_df = model_df[model_df["season_label"] == 1]
X_base, y_base = BASE_df[FEATURES].values, BASE_df["season_label"].values
X_anom, y_anom = ANOM_df[FEATURES].values, ANOM_df["season_label"].values


# Exp A: train IQR pseudo-labels on baseline, apply to 2025-2026
y_base_pseudo = BASE_df["iqr_anomaly"].values
dt_forward    = DecisionTreeClassifier(criterion="gini", max_depth=4,
                                        min_samples_leaf=10, random_state=42)
dt_forward.fit(X_base, y_base_pseudo)
y_anom_pred = dt_forward.predict(X_anom)
pct_flagged = 100 * y_anom_pred.sum() / len(y_anom_pred)
lift        = pct_flagged / (y_base_pseudo.mean() * 100)

print("Exp A  (forward — baseline model applied to 2025-2026):")
print(f"  baseline outlier rate : {y_base_pseudo.mean()*100:.1f}%")
print(f"  2025-2026 flagged     : {y_anom_pred.sum():,}/{len(y_anom_pred):,} ({pct_flagged:.1f}%)")
print(f"  lift                  : {lift:.2f}x")


# Exp B: balanced tree trained on mega-fire + sampled baseline
# class_weight='balanced' corrects for class imbalance between the two pools
np.random.seed(42)
base_idx    = np.random.permutation(len(BASE_df))
# Use 70% of baseline events for training context, hold out 30% for testing
split       = int(len(BASE_df) * 0.70)
X_rev_train = np.vstack([X_anom, X_base[base_idx[:split]]])
y_rev_train = np.concatenate([y_anom, y_base[base_idx[:split]]])
X_rev_test  = X_base[base_idx[split:]]
y_rev_test  = y_base[base_idx[split:]]

dt_reverse = DecisionTreeClassifier(criterion="gini", max_depth=4,
                                     min_samples_leaf=10, class_weight="balanced",
                                     random_state=42)
dt_reverse.fit(X_rev_train, y_rev_train)
y_base_pred = dt_reverse.predict(X_rev_test)
tnr         = 100 * (y_base_pred == 0).sum() / len(y_rev_test)

print(f"\nExp B  (reverse — mega-fire model applied to baseline):")
print(f"  test set              : {len(y_rev_test):,} held-out baseline events")
print(f"  correctly rejected TN : {(y_base_pred==0).sum():,}/{len(y_rev_test):,} ({tnr:.1f}%)")
print(f"  false triggers FP     : {(y_base_pred==1).sum():,} ({100*(y_base_pred==1).mean():.1f}%)")


# Exp C: hand-crafted two-attribute rule
model_df = model_df.copy()
# Rule: not-IRWIN (i.e. GDACS) events are mega-fire;
# IRWIN events above the IQR fence are also mega-fire.
model_df["rule_pred"] = (
    (model_df["agency_irwin"] == 0) |
    ((model_df["agency_irwin"] == 1) & (model_df["magnitude_max"] > FENCE))
).astype(int)

y_rule = model_df["rule_pred"].values
y_true = model_df["season_label"].values
acc    = accuracy_score(y_true, y_rule)
prec   = precision_score(y_true, y_rule, zero_division=0)
rec    = recall_score(y_true, y_rule, zero_division=0)
f1     = f1_score(y_true, y_rule, zero_division=0)
cm     = confusion_matrix(y_true, y_rule)

print(f"\nExp C  (rule-based: NOT IRWIN OR (IRWIN AND magnitude > IQR fence)):")
print(f"  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}  FN={cm[1,0]:,}  TP={cm[1,1]:,}")


# Summary
X_all = model_df[FEATURES].values
y_all = model_df["season_label"].values
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.25,
                                           random_state=42, stratify=y_all)
dt_full = DecisionTreeClassifier(criterion="gini", max_depth=4,
                                  min_samples_leaf=10, random_state=42)
dt_full.fit(X_tr, y_tr)
y_te_pred = dt_full.predict(X_te)

print(f"\nvalidation summary:")
print(f"  {'experiment':<38} {'accuracy':>10}  {'precision':>10}  {'recall':>8}  {'f1':>6}")
print(f"  {'section 4 (80/20 split)':<38} {accuracy_score(y_te,y_te_pred)*100:>9.1f}%  {precision_score(y_te,y_te_pred):>10.3f}  {recall_score(y_te,y_te_pred):>8.3f}  {f1_score(y_te,y_te_pred):>6.3f}")
print(f"  {'exp A forward lift':<38} {pct_flagged:>9.1f}%  {'—':>10}  {'—':>8}  {'—':>6}")
print(f"  {'exp B reverse TN rate':<38} {tnr:>9.1f}%  {'—':>10}  {'—':>8}  {'—':>6}")
print(f"  {'exp C rule-based':<38} {acc*100:>9.1f}%  {prec:>10.3f}  {rec:>8.3f}  {f1:>6.3f}")

"""
Section 1 - Attribute Discovery
Analyzes EONET metadata fields to identify the "signature" of an anomalous fire season.
Fields examined: magnitude_value, duration_days, source_count, is_active, geometry spread.
"""

import pandas as pd
import numpy as np

# ── Load cached data ───────────────────────────────────────────────────────────

df = pd.read_csv(
    "eonet_wildfire_combined.csv",
    parse_dates=["obs_date", "closed", "first_obs"],
)

# ── Collapse to one row per event (take max magnitude, final source_count) ─────
# Each event has multiple geometry observations; we want per-event stats.

event_df = (
    df.groupby(["event_id", "season_name", "season_label"])
    .agg(
        title           = ("title",           "first"),
        magnitude_max   = ("magnitude_value", "max"),
        magnitude_mean  = ("magnitude_value", "mean"),
        obs_count       = ("obs_date",        "count"),  # how many position reports
        source_count    = ("source_count",    "max"),
        sources         = ("sources",         "first"),
        is_active       = ("is_active",       "first"),
        duration_days   = ("duration_days",   "first"),
        lon_std         = ("lon",             "std"),    # geographic spread of fire
        lat_std         = ("lat",             "std"),
        lon_range       = ("lon",             lambda x: x.max() - x.min()),
        lat_range       = ("lat",             lambda x: x.max() - x.min()),
    )
    .reset_index()
)

# Geographic spread proxy: diagonal of bounding box in degrees
event_df["geo_spread_deg"] = np.sqrt(
    event_df["lon_range"] ** 2 + event_df["lat_range"] ** 2
)

print("=== Per-event dataset shape ===")
print(event_df.shape)
print(f"Seasons: {event_df['season_name'].value_counts().to_dict()}\n")

# ── 1. Descriptive statistics per season ──────────────────────────────────────

FEATURES = ["magnitude_max", "duration_days", "source_count",
            "obs_count", "geo_spread_deg"]

print("=== Descriptive Statistics by Season ===")
for feat in FEATURES:
    grp = event_df.groupby("season_name")[feat]
    stats = grp.agg(["mean", "median", "std", "min", "max"]).round(2)
    print(f"\n  {feat}:")
    print(stats.to_string())

# ── 2. Active-event proportion ─────────────────────────────────────────────────

print("\n=== Active (null-closed) Event Rate ===")
active_rate = event_df.groupby("season_name")["is_active"].mean().round(4) * 100
print(active_rate.rename("active_pct (%)").to_string())

# ── 3. Source diversity ────────────────────────────────────────────────────────

print("\n=== Source Agency Diversity ===")
# Expand pipe-separated sources into individual agency IDs
agency_rows = []
for _, row in event_df.iterrows():
    for agency in str(row["sources"]).split("|"):
        agency_rows.append({"season_name": row["season_name"], "agency": agency.strip()})

agency_df = pd.DataFrame(agency_rows)
agency_dist = agency_df.groupby(["season_name", "agency"]).size().unstack(fill_value=0)
print(agency_dist.to_string())

print("\n  Multi-source events (source_count > 1):")
multi = event_df[event_df["source_count"] > 1].groupby("season_name").size()
print(multi.rename("count").to_string())

# ── 4. Magnitude distribution summary ─────────────────────────────────────────

print("\n=== Magnitude Percentile Distribution ===")
for season, grp in event_df.groupby("season_name"):
    mag = grp["magnitude_max"].dropna()
    pcts = np.percentile(mag, [25, 50, 75, 90, 95, 99])
    print(f"\n  {season}  (n={len(mag)} events with magnitude data)")
    for p, v in zip([25, 50, 75, 90, 95, 99], pcts):
        print(f"    P{p:>2}: {v:>10,.1f} acres")

# ── 5. Anomaly threshold (IQR method, preview for Section 2) ──────────────────

print("\n=== Outlier Count Preview (IQR x1.5, per season) ===")
for season, grp in event_df.groupby("season_name"):
    mag = grp["magnitude_max"].dropna()
    Q1, Q3 = mag.quantile(0.25), mag.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    outliers = (mag > upper).sum()
    print(f"  {season}: IQR={IQR:,.0f}  upper fence={upper:,.0f}  "
          f"outliers={outliers} ({100*outliers/len(mag):.1f}%)")

# ── 6. Signature summary ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ATTRIBUTE DISCOVERY SUMMARY")
print("=" * 60)

baseline   = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
anomalous  = event_df[event_df["season_label"] == 1]["magnitude_max"].dropna()

ratio_mean   = anomalous.mean()   / baseline.mean()
ratio_median = anomalous.median() / baseline.median()

print(f"  Mean magnitude ratio  (2025/2024): {ratio_mean:.2f}x")
print(f"  Median magnitude ratio(2025/2024): {ratio_median:.2f}x")
print(f"  Active-event rate 2024-2025: "
      f"{event_df[event_df['season_label']==0]['is_active'].mean()*100:.1f}%")
print(f"  Active-event rate 2025-2026: "
      f"{event_df[event_df['season_label']==1]['is_active'].mean()*100:.1f}%")
print()
print("  Proposed Mega-Fire Signature (3 attributes):")
print("    [1] magnitude_max > P75 of baseline (size)")
print("    [2] obs_count     > 1               (multi-report persistence)")
print("    [3] is_active     == True  OR  duration_days > 14  (longevity)")
print()

sig_mask = (
    (event_df["magnitude_max"]  > baseline.quantile(0.75)) &
    (event_df["obs_count"]      > 1) &
    ((event_df["is_active"]) | (event_df["duration_days"] > 14))
)
sig_counts = event_df[sig_mask].groupby("season_name").size()
print("  Events matching signature:")
print(sig_counts.rename("count").to_string())
hit_rate = sig_counts / event_df.groupby("season_name").size()
print("\n  Signature hit rate:")
print((hit_rate * 100).round(1).rename("pct (%)").to_string())

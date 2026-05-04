"""
Analyses EONET metadata fields to identify which attributes discriminate
between the 2024-2025 baseline and 2025-2026 anomalous fire seasons.
Season windows are extracted from the decade dataset by obs_date.
"""

import pandas as pd
import numpy as np

raw = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date", "closed"])

# Assign season labels by observation date
raw.loc[(raw["obs_date"] >= "2024-05-01") & (raw["obs_date"] <= "2025-04-30"), "season_label"] = 0
raw.loc[(raw["obs_date"] >= "2025-05-01") & (raw["obs_date"] <= "2026-04-30"), "season_label"] = 1
season_raw = raw[raw["season_label"].notna()].copy()
season_raw["season_label"] = season_raw["season_label"].astype(int)
season_raw["season_name"]  = season_raw["season_label"].map({0: "2024_2025", 1: "2025_2026"})

# Collapse to one row per event; the decade dataset stores one row per geometry observation
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

print(f"dataset: {event_df.shape}  seasons: {event_df['season_name'].value_counts().to_dict()}\n")

print("descriptive statistics by season:")
for feat in ["magnitude_max", "source_count"]:
    stats = event_df.groupby("season_name")[feat].agg(["mean","median","std","min","max"]).round(2)
    print(f"\n  {feat}:")
    print(stats.to_string())

print("\nactive event rate (null-closed):")
active_rate = event_df.groupby("season_name")["is_active"].mean().round(4) * 100
print(active_rate.rename("active_pct (%)").to_string())

print("\nsource agency counts:")
agency_rows = []
for _, row in event_df.iterrows():
    for agency in str(row["sources"]).split("|"):
        agency_rows.append({"season_name": row["season_name"], "agency": agency.strip()})
agency_df   = pd.DataFrame(agency_rows)
agency_dist = agency_df.groupby(["season_name", "agency"]).size().unstack(fill_value=0)
print(agency_dist.to_string())

print("\nmagnitude percentile distribution:")
for season, grp in event_df.groupby("season_name"):
    mag  = grp["magnitude_max"].dropna()
    pcts = np.percentile(mag, [25, 50, 75, 90, 95, 99])
    print(f"\n  {season}  (n={len(mag)})")
    for p, v in zip([25, 50, 75, 90, 95, 99], pcts):
        print(f"    P{p:>2}: {v:>12,.1f} acres")

print("\nIQR outlier count (baseline-derived fence):")
baseline  = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
anomalous = event_df[event_df["season_label"] == 1]["magnitude_max"].dropna()
Q1, Q3    = baseline.quantile(0.25), baseline.quantile(0.75)
fence     = Q3 + 1.5 * (Q3 - Q1)
for season, grp in event_df.groupby("season_name"):
    mag      = grp["magnitude_max"].dropna()
    outliers = (mag > fence).sum()
    print(f"  {season}: fence={fence:,.0f}  outliers={outliers} ({100*outliers/len(mag):.1f}%)")

print(f"\nmagnitude ratios (2025-2026 / 2024-2025):")
print(f"  mean   : {anomalous.mean()/baseline.mean():.2f}x")
print(f"  median : {anomalous.median()/baseline.median():.2f}x")

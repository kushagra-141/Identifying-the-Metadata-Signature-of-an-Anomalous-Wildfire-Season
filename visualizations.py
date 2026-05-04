"""
Generates four figures comparing the 2024-2025 baseline and 2025-2026
anomalous fire seasons using data extracted from the decade dataset.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

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
        sources       = ("sources",         "first"),
        lon           = ("lon",             "first"),
        lat           = ("lat",             "first"),
    )
    .reset_index()
)
event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS", na=False).astype(int)

PALETTE = {"2024_2025": "#4C9BE8", "2025_2026": "#E84C4C"}
LABELS  = {"2024_2025": "2024-2025 Baseline", "2025_2026": "2025-2026 Mega-Fire"}

baseline  = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
Q3_b      = baseline.quantile(0.75)
fence     = Q3_b + 1.5 * (Q3_b - baseline.quantile(0.25))


# Figure 1: Geospatial map
fig1, ax1 = plt.subplots(figsize=(14, 7))
ax1.set_facecolor("#D6EAF8")
for season, color in PALETTE.items():
    sub   = event_df[event_df["season_name"] == season].dropna(subset=["lon","lat","magnitude_max"])
    sizes = np.clip(sub["magnitude_max"] / 200, 5, 200)
    ax1.scatter(sub["lon"], sub["lat"], s=sizes, c=color,
                alpha=0.35, linewidths=0,
                label=f"{LABELS[season]}  (n={len(sub):,})")
for name, lon, lat in [("N.America",-100,45),("S.America",-60,-15),("Europe",15,52),
                        ("Africa",20,5),("Asia",100,45),("Australia",135,-25)]:
    ax1.text(lon, lat, name, fontsize=7, color="#555555", ha="center", alpha=0.6)
ax1.set_xlim(-180, 180); ax1.set_ylim(-90, 90)
ax1.set_xlabel("Longitude"); ax1.set_ylabel("Latitude")
ax1.set_title("Figure 1 — Global Wildfire Event Distribution by Season\n"
              "Point size proportional to magnitude (acres)", fontweight="bold")
ax1.legend(loc="lower left", framealpha=0.85)
fig1.tight_layout()
fig1.savefig("fig1_geospatial_map.png")
plt.close(fig1)
print("saved: fig1_geospatial_map.png")


# Figure 2: Magnitude distribution
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

ax_kde = axes[0]
for season, color in PALETTE.items():
    sub = event_df[event_df["season_name"] == season]["magnitude_max"].dropna()
    sns.kdeplot(sub, ax=ax_kde, color=color, linewidth=2.5,
                label=LABELS[season], fill=True, alpha=0.25)
    ax_kde.axvline(sub.median(), color=color, linewidth=1.5, linestyle="--", alpha=0.8)
ax_kde.axvline(fence, color="orange", linewidth=1.5, linestyle=":",
               label=f"IQR fence ({fence:,.0f} ac)")
ax_kde.set_xlabel("Magnitude (acres)"); ax_kde.set_ylabel("Density")
ax_kde.set_title("Magnitude KDE — Distribution Shift\n(dashed = medians)")
ax_kde.legend(fontsize=8); ax_kde.set_xlim(0, 50000)

ax_hist = axes[1]
bins = np.linspace(0, 50000, 40)
for season, color in PALETTE.items():
    sub = event_df[event_df["season_name"] == season]["magnitude_max"].dropna()
    ax_hist.hist(sub, bins=bins, color=color, alpha=0.55,
                 label=LABELS[season], edgecolor="none")
ax_hist.axvline(fence, color="orange", linewidth=1.5, linestyle=":")
ax_hist.set_yscale("log")
ax_hist.set_xlabel("Magnitude (acres)"); ax_hist.set_ylabel("Event count (log)")
ax_hist.set_title("Magnitude Histogram (log y-axis)")
ax_hist.legend(fontsize=8)

fig2.suptitle("Figure 2 — Wildfire Magnitude Distribution Comparison",
              fontweight="bold", y=1.01)
fig2.tight_layout()
fig2.savefig("fig2_magnitude_distribution.png", bbox_inches="tight")
plt.close(fig2)
print("saved: fig2_magnitude_distribution.png")


# Figure 3: Anomaly heatmap
fig3, ax3 = plt.subplots(figsize=(12, 4))
bin_edges  = [0, 1000, 2000, fence, 7000, 10000, 15000, 350001]
bin_labels = ["0-1k", "1k-2k", f"2k-{fence/1000:.1f}k\n(normal)",
              f"{fence/1000:.1f}k-7k\n(anomaly)", "7k-10k", "10k-15k", "15k+"]
heat_data  = {}
for season in ["2024_2025", "2025_2026"]:
    sub    = event_df[event_df["season_name"] == season]["magnitude_max"].dropna()
    counts, _ = np.histogram(sub, bins=bin_edges)
    heat_data[LABELS[season]] = counts
heat_df = pd.DataFrame(heat_data, index=bin_labels).T
sns.heatmap(heat_df, ax=ax3, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, cbar_kws={"label": "Event count"},
            annot_kws={"size": 11})
ax3.axvline(3, color="white", linewidth=3, linestyle="--")
ax3.set_title("Figure 3 — Anomaly Heatmap: Event Count by Magnitude Bin and Season",
              fontweight="bold")
ax3.set_xlabel("Magnitude Bin (acres)")
fig3.tight_layout()
fig3.savefig("fig3_anomaly_heatmap.png", bbox_inches="tight")
plt.close(fig3)
print("saved: fig3_anomaly_heatmap.png")


# Figure 4: Agency composition
fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5))
agency_counts = (
    event_df.groupby("season_name")["agency_gdacs"]
    .value_counts().unstack(fill_value=0)
    .rename(columns={0: "IRWIN (domestic)", 1: "GDACS (global)"})
)
agency_counts.plot(kind="bar", stacked=True, ax=axes4[0],
                   color=["#5DADE2","#E74C3C"], edgecolor="white")
axes4[0].set_xticklabels([LABELS[s] for s in agency_counts.index], rotation=15, ha="right")
axes4[0].set_title("Event Count by Reporting Agency")
axes4[0].set_ylabel("Number of events")
axes4[0].legend(loc="upper right")

agency_pct = agency_counts.div(agency_counts.sum(axis=1), axis=0) * 100
agency_pct.plot(kind="bar", stacked=True, ax=axes4[1],
                color=["#5DADE2","#E74C3C"], edgecolor="white")
axes4[1].set_xticklabels([LABELS[s] for s in agency_pct.index], rotation=15, ha="right")
axes4[1].set_title("Agency Share (%) — primary classifier signal")
axes4[1].set_ylabel("Percentage of events")
axes4[1].axhline(50, color="black", linewidth=1, linestyle="--", alpha=0.4)
axes4[1].legend(loc="upper right")

fig4.suptitle("Figure 4 — Reporting Agency Distribution by Season",
              fontweight="bold")
fig4.tight_layout()
fig4.savefig("fig4_agency_composition.png", bbox_inches="tight")
plt.close(fig4)
print("saved: fig4_agency_composition.png")

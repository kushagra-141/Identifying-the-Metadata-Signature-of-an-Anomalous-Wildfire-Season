"""
Section 3 - Data Visualization
Plots:
  1. Geospatial progression map  (lon/lat scatter, both seasons overlaid)
  2. Magnitude KDE + histogram   (distribution shift)
  3. Anomaly heatmap             (magnitude bins x season)
  4. Agency composition bar      (GDACS vs IRWIN per season)
All figures saved to PNG.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── Load deduplicated event data ───────────────────────────────────────────────

raw = pd.read_csv(
    "eonet_wildfire_combined.csv",
    parse_dates=["obs_date", "closed", "first_obs"],
)

event_df = (
    raw.groupby(["event_id", "season_name", "season_label"])
    .agg(
        magnitude_max = ("magnitude_value", "max"),
        sources       = ("sources",         "first"),
        lon           = ("lon",             "first"),
        lat           = ("lat",             "first"),
    )
    .reset_index()
)

event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS").astype(int)

BASE  = event_df[event_df["season_label"] == 0].dropna(subset=["magnitude_max"])
ANOM  = event_df[event_df["season_label"] == 1].dropna(subset=["magnitude_max"])

PALETTE = {"2024_2025": "#4C9BE8", "2025_2026": "#E84C4C"}
LABELS  = {"2024_2025": "2024-2025 Baseline", "2025_2026": "2025-2026 Mega-Fire"}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Geospatial Progression Map
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(14, 7))

# Draw a simple land outline using a world bounding box background
ax1.set_facecolor("#D6EAF8")
ax1.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
ax1.axvline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)

for season, color in PALETTE.items():
    sub = event_df[event_df["season_name"] == season].dropna(
        subset=["lon", "lat", "magnitude_max"]
    )
    sizes = np.clip(sub["magnitude_max"] / 80, 10, 300)
    ax1.scatter(
        sub["lon"], sub["lat"],
        s=sizes, c=color, alpha=0.45, linewidths=0,
        label=f"{LABELS[season]}  (n={len(sub)})",
        zorder=3,
    )

# IQR upper fence line annotation
ax1.text(
    -175, -70,
    "Point size = magnitude (acres)\nIQR anomaly threshold: 4,469 acres",
    fontsize=8, color="white",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", alpha=0.7),
)

ax1.set_xlim(-180, 180)
ax1.set_ylim(-90, 90)
ax1.set_xlabel("Longitude")
ax1.set_ylabel("Latitude")
ax1.set_title(
    "Figure 1 — Global Wildfire Event Distribution by Season\n"
    "(NASA EONET v3 | point size proportional to magnitude in acres)",
    fontweight="bold",
)
ax1.legend(loc="lower left", framealpha=0.85)

# Annotate continents lightly
for name, lon, lat in [
    ("N. America", -100, 45), ("S. America", -60, -15),
    ("Europe", 15, 52),       ("Africa", 20, 5),
    ("Asia", 100, 45),        ("Australia", 135, -25),
]:
    ax1.text(lon, lat, name, fontsize=7, color="#555555",
             ha="center", alpha=0.6)

fig1.tight_layout()
fig1.savefig("fig1_geospatial_map.png")
print("Saved: fig1_geospatial_map.png")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Magnitude Distribution (KDE + Histogram overlay)
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: overlaid KDE
ax_kde = axes[0]
for season, color in PALETTE.items():
    sub = event_df[event_df["season_name"] == season]["magnitude_max"].dropna()
    sns.kdeplot(sub, ax=ax_kde, color=color, linewidth=2.5,
                label=LABELS[season], fill=True, alpha=0.25)
    ax_kde.axvline(sub.median(), color=color, linewidth=1.5,
                   linestyle="--", alpha=0.8)

# IQR fence
fence_upper = 4469.1
ax_kde.axvline(fence_upper, color="orange", linewidth=1.5,
               linestyle=":", label=f"IQR upper fence ({fence_upper:,.0f} ac)")
ax_kde.set_xlabel("Magnitude (acres burned)")
ax_kde.set_ylabel("Density")
ax_kde.set_title("Magnitude KDE — Distribution Shift\n(dashed lines = medians)")
ax_kde.legend(fontsize=8)
ax_kde.set_xlim(0, 35000)

# Right: side-by-side histograms with log-y scale
ax_hist = axes[1]
bins = np.linspace(0, 35000, 40)
for season, color in PALETTE.items():
    sub = event_df[event_df["season_name"] == season]["magnitude_max"].dropna()
    ax_hist.hist(sub, bins=bins, color=color, alpha=0.55,
                 label=LABELS[season], edgecolor="none")
ax_hist.axvline(fence_upper, color="orange", linewidth=1.5,
                linestyle=":", label=f"IQR fence")
ax_hist.set_yscale("log")
ax_hist.set_xlabel("Magnitude (acres burned)")
ax_hist.set_ylabel("Event count (log scale)")
ax_hist.set_title("Magnitude Histogram (log y-axis)\nshows tail behaviour")
ax_hist.legend(fontsize=8)

fig2.suptitle("Figure 2 — Wildfire Magnitude Distribution Comparison",
              fontweight="bold", y=1.01)
fig2.tight_layout()
fig2.savefig("fig2_magnitude_distribution.png", bbox_inches="tight")
print("Saved: fig2_magnitude_distribution.png")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Anomaly Heatmap (magnitude bin x season, count)
# ══════════════════════════════════════════════════════════════════════════════

fig3, ax3 = plt.subplots(figsize=(12, 4))

bin_edges  = [0, 1000, 2000, 4469, 7000, 10000, 15000, 35001]
bin_labels = ["0-1k", "1k-2k", "2k-4.5k\n(normal)", "4.5k-7k\n(anomaly)",
              "7k-10k", "10k-15k", "15k+"]

heat_data = {}
for season in ["2024_2025", "2025_2026"]:
    sub = event_df[event_df["season_name"] == season]["magnitude_max"].dropna()
    counts, _ = np.histogram(sub, bins=bin_edges)
    heat_data[LABELS[season]] = counts

heat_df = pd.DataFrame(heat_data, index=bin_labels).T

sns.heatmap(
    heat_df,
    ax=ax3,
    annot=True, fmt="d",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Event count"},
    annot_kws={"size": 11},
)
ax3.set_title(
    "Figure 3 — Anomaly Heatmap: Event Count by Magnitude Bin and Season\n"
    "(bins left of dashed line are within baseline normal range)",
    fontweight="bold",
)
ax3.set_xlabel("Magnitude Bin (acres burned)")
ax3.set_ylabel("")

# Vertical line between normal and anomaly bins (after 3rd column)
ax3.axvline(3, color="white", linewidth=3, linestyle="--")

fig3.tight_layout()
fig3.savefig("fig3_anomaly_heatmap.png", bbox_inches="tight")
print("Saved: fig3_anomaly_heatmap.png")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Agency Composition (GDACS vs IRWIN per season)
# ══════════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 2, figsize=(11, 5))

agency_counts = (
    event_df.groupby("season_name")["agency_gdacs"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={0: "IRWIN (domestic)", 1: "GDACS (global)"})
)

# Stacked bar — left panel
agency_counts.plot(
    kind="bar", stacked=True, ax=axes4[0],
    color=["#5DADE2", "#E74C3C"], edgecolor="white", linewidth=0.5,
)
axes4[0].set_xticklabels(
    [LABELS[s] for s in agency_counts.index], rotation=15, ha="right"
)
axes4[0].set_title("Event Count by Reporting Agency")
axes4[0].set_ylabel("Number of events")
axes4[0].legend(loc="upper right")

# Proportional bar — right panel
agency_pct = agency_counts.div(agency_counts.sum(axis=1), axis=0) * 100
agency_pct.plot(
    kind="bar", stacked=True, ax=axes4[1],
    color=["#5DADE2", "#E74C3C"], edgecolor="white", linewidth=0.5,
)
axes4[1].set_xticklabels(
    [LABELS[s] for s in agency_pct.index], rotation=15, ha="right"
)
axes4[1].set_title("Agency Share (%) — the biggest classifier signal")
axes4[1].set_ylabel("Percentage of events")
axes4[1].axhline(50, color="black", linewidth=1, linestyle="--", alpha=0.4)
axes4[1].legend(loc="upper right")

# Annotate the inversion
axes4[1].annotate(
    "GDACS flips from\nminority to majority",
    xy=(1, 65), xytext=(0.55, 80),
    fontsize=8, color="#C0392B",
    arrowprops=dict(arrowstyle="->", color="#C0392B"),
)

for val, patch in zip(agency_pct["GDACS (global)"],
                      axes4[1].patches[len(agency_pct):]):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_y() + patch.get_height() / 2
    axes4[1].text(x, y, f"{val:.0f}%", ha="center", va="center",
                  fontsize=10, fontweight="bold", color="white")

fig4.suptitle("Figure 4 — Reporting Agency Distribution by Season\n"
              "(91.7% of Decision Tree Gini gain comes from this feature)",
              fontweight="bold")
fig4.tight_layout()
fig4.savefig("fig4_agency_composition.png", bbox_inches="tight")
print("Saved: fig4_agency_composition.png")
plt.close(fig4)

print("\nAll figures saved. Summary:")
print("  fig1_geospatial_map.png       — global fire locations, sized by magnitude")
print("  fig2_magnitude_distribution.png — KDE + histogram, distribution shift")
print("  fig3_anomaly_heatmap.png      — count per magnitude bin per season")
print("  fig4_agency_composition.png   — GDACS vs IRWIN inversion")

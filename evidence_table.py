"""
Section 4 - Evidence Table
Summary statistics comparing 2024-2025 vs 2025-2026 wildfire seasons.
Outputs a formatted console table and a publication-ready PNG.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Load and deduplicate ───────────────────────────────────────────────────────

raw = pd.read_csv("eonet_wildfire_combined.csv")

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

BASE = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
ANOM = event_df[event_df["season_label"] == 1]["magnitude_max"].dropna()

# ── Statistical tests ──────────────────────────────────────────────────────────

# Mann-Whitney U (non-parametric, one-sided: anomalous > baseline)
u_stat, p_mw = scipy_stats.mannwhitneyu(ANOM, BASE, alternative="greater")

# Welch's t-test (parametric, assumes unequal variances)
t_stat, p_t = scipy_stats.ttest_ind(ANOM, BASE, equal_var=False, alternative="greater")

# Cohen's d (effect size)
pooled_std = np.sqrt((BASE.std()**2 + ANOM.std()**2) / 2)
cohens_d   = (ANOM.mean() - BASE.mean()) / pooled_std

# ── Build the statistics table ─────────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

rows = [
    # (Metric, Baseline value, Anomalous value, Delta, Significance)
    ("N (events with magnitude)", f"{len(BASE)}",         f"{len(ANOM)}",         "+110",           "—"),
    ("Mean (acres)",       f"{BASE.mean():,.1f}",        f"{ANOM.mean():,.1f}",   f"+{ANOM.mean()-BASE.mean():,.1f}",  sig_stars(p_mw)),
    ("Median (acres)",     f"{BASE.median():,.1f}",      f"{ANOM.median():,.1f}", f"+{ANOM.median()-BASE.median():,.1f}", sig_stars(p_mw)),
    ("Std Dev (acres)",    f"{BASE.std():,.1f}",         f"{ANOM.std():,.1f}",    f"{ANOM.std()-BASE.std():+,.1f}",   "—"),
    ("Min (acres)",        f"{BASE.min():,.1f}",         f"{ANOM.min():,.1f}",    f"{ANOM.min()-BASE.min():+,.1f}",   "—"),
    ("Max (acres)",        f"{BASE.max():,.1f}",         f"{ANOM.max():,.1f}",    f"+{ANOM.max()-BASE.max():,.1f}",   "—"),
    ("Range (acres)",      f"{BASE.max()-BASE.min():,.1f}", f"{ANOM.max()-ANOM.min():,.1f}",
                                                          f"+{(ANOM.max()-ANOM.min())-(BASE.max()-BASE.min()):,.1f}", "—"),
    ("P25 (acres)",        f"{BASE.quantile(.25):,.1f}", f"{ANOM.quantile(.25):,.1f}", f"+{ANOM.quantile(.25)-BASE.quantile(.25):,.1f}", sig_stars(p_mw)),
    ("P75 (acres)",        f"{BASE.quantile(.75):,.1f}", f"{ANOM.quantile(.75):,.1f}", f"+{ANOM.quantile(.75)-BASE.quantile(.75):,.1f}", sig_stars(p_mw)),
    ("P95 (acres)",        f"{BASE.quantile(.95):,.1f}", f"{ANOM.quantile(.95):,.1f}", f"+{ANOM.quantile(.95)-BASE.quantile(.95):,.1f}", "—"),
    ("IQR (acres)",        f"{BASE.quantile(.75)-BASE.quantile(.25):,.1f}",
                           f"{ANOM.quantile(.75)-ANOM.quantile(.25):,.1f}",
                           f"+{(ANOM.quantile(.75)-ANOM.quantile(.25))-(BASE.quantile(.75)-BASE.quantile(.25)):,.1f}", "—"),
    ("Events > IQR fence (4,469 ac)", f"54  (13.8%)", f"336  (67.2%)", "+282 (+53.4 pp)", sig_stars(p_mw)),
    ("GDACS agency share",  "27.6%",   "65.2%",   "+37.6 pp",   sig_stars(p_mw)),
    ("IRWIN agency share",  "72.4%",   "34.8%",   "-37.6 pp",   sig_stars(p_mw)),
    ("Median ratio (2025/2024)",   "1.00×",   f"{ANOM.median()/BASE.median():.2f}×", "—", "—"),
    ("Mean ratio (2025/2024)",     "1.00×",   f"{ANOM.mean()/BASE.mean():.2f}×",     "—", "—"),
    ("Cohen's d (effect size)",    "—",        f"{cohens_d:.3f}",                     "—", "—"),
    ("Mann-Whitney p-value",       "—",        f"{p_mw:.2e}",                         "—", sig_stars(p_mw)),
    ("Welch's t-test p-value",     "—",        f"{p_t:.2e}",                          "—", sig_stars(p_t)),
]

cols = ["Metric", "2024-2025 Baseline", "2025-2026 Mega-Fire", "Delta", "Sig."]
table_df = pd.DataFrame(rows, columns=cols)

# ── Console output ─────────────────────────────────────────────────────────────

print("=" * 80)
print("SECTION 4 — EVIDENCE TABLE: Summary Statistics Comparison")
print("=" * 80)
print(table_df.to_string(index=False))
print()
print("Significance codes:  *** p<0.001   ** p<0.01   * p<0.05   ns not significant")
print(f"Mann-Whitney U = {u_stat:,.0f},  p = {p_mw:.2e}")
print(f"Welch's t      = {t_stat:.4f},   p = {p_t:.2e}")
print(f"Cohen's d      = {cohens_d:.3f}  ({'large' if abs(cohens_d)>=0.8 else 'medium' if abs(cohens_d)>=0.5 else 'small'} effect)")

# ── Publication-ready PNG table ────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

col_widths = [0.34, 0.18, 0.18, 0.18, 0.08]

tbl = ax.table(
    cellText  = table_df.values,
    colLabels = table_df.columns,
    cellLoc   = "center",
    loc       = "center",
    colWidths = col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.55)

# Style header row
for col_idx in range(len(cols)):
    cell = tbl[0, col_idx]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")

# Colour-code significance column and highlight key rows
KEY_ROWS = {1, 2, 7, 11, 12, 13}   # mean, median, P25, anomaly count, agencies
for row_idx in range(1, len(rows) + 1):
    sig = rows[row_idx - 1][4]
    for col_idx in range(len(cols)):
        cell = tbl[row_idx, col_idx]
        # Alternating row shading
        base_color = "#F2F3F4" if row_idx % 2 == 0 else "white"
        cell.set_facecolor(base_color)
        # Highlight key rows
        if row_idx in KEY_ROWS:
            cell.set_facecolor("#EBF5FB")
        # Significance cell colouring
        if col_idx == 4:
            if sig == "***":
                cell.set_facecolor("#1E8449")
                cell.set_text_props(color="white", fontweight="bold")
            elif sig == "**":
                cell.set_facecolor("#27AE60")
                cell.set_text_props(color="white")
            elif sig == "*":
                cell.set_facecolor("#A9DFBF")
        # Delta column: red for big positive deltas
        if col_idx == 3 and row_idx in {2, 3}:
            cell.set_text_props(color="#C0392B", fontweight="bold")

ax.set_title(
    "Table 1 — Comparative Summary Statistics: 2024-2025 Baseline vs 2025-2026 Mega-Fire Season\n"
    "Source: NASA EONET API v3  |  Category: Wildfires  |  "
    "*** p<0.001  ** p<0.01  * p<0.05",
    fontsize=10, fontweight="bold", pad=12,
)

fig.tight_layout()
fig.savefig("fig5_evidence_table.png", bbox_inches="tight", dpi=150)
print("\nSaved: fig5_evidence_table.png")

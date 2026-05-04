"""
Generates a summary statistics table comparing the 2024-2025 baseline
and 2025-2026 anomalous fire seasons with significance testing.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

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
    )
    .reset_index()
)
event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS", na=False).astype(int)
event_df["agency_irwin"] = event_df["sources"].str.contains("IRWIN", na=False).astype(int)

BASE = event_df[event_df["season_label"] == 0]["magnitude_max"].dropna()
ANOM = event_df[event_df["season_label"] == 1]["magnitude_max"].dropna()

u_stat, p_mw = scipy_stats.mannwhitneyu(ANOM, BASE, alternative="greater")
t_stat, p_t  = scipy_stats.ttest_ind(ANOM, BASE, equal_var=False, alternative="greater")
pooled_std   = np.sqrt((BASE.std()**2 + ANOM.std()**2) / 2)
cohens_d     = (ANOM.mean() - BASE.mean()) / pooled_std
fence        = BASE.quantile(0.75) + 1.5 * (BASE.quantile(0.75) - BASE.quantile(0.25))

def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

rows = [
    ("N (with magnitude)",        f"{len(BASE):,}",          f"{len(ANOM):,}",          f"+{len(ANOM)-len(BASE):,}",    "—"),
    ("Mean (acres)",               f"{BASE.mean():,.1f}",     f"{ANOM.mean():,.1f}",     f"+{ANOM.mean()-BASE.mean():,.1f}",   sig(p_mw)),
    ("Median (acres)",             f"{BASE.median():,.1f}",   f"{ANOM.median():,.1f}",   f"+{ANOM.median()-BASE.median():,.1f}", sig(p_mw)),
    ("Std Dev (acres)",            f"{BASE.std():,.1f}",      f"{ANOM.std():,.1f}",      f"{ANOM.std()-BASE.std():+,.1f}",    "—"),
    ("Min (acres)",                f"{BASE.min():,.1f}",      f"{ANOM.min():,.1f}",      f"{ANOM.min()-BASE.min():+,.1f}",    "—"),
    ("Max (acres)",                f"{BASE.max():,.1f}",      f"{ANOM.max():,.1f}",      f"{ANOM.max()-BASE.max():+,.1f}",    "—"),
    ("Range (acres)",              f"{BASE.max()-BASE.min():,.1f}", f"{ANOM.max()-ANOM.min():,.1f}", "—", "—"),
    ("P25 (acres)",                f"{BASE.quantile(.25):,.1f}", f"{ANOM.quantile(.25):,.1f}", f"+{ANOM.quantile(.25)-BASE.quantile(.25):,.1f}", sig(p_mw)),
    ("P75 (acres)",                f"{BASE.quantile(.75):,.1f}", f"{ANOM.quantile(.75):,.1f}", f"+{ANOM.quantile(.75)-BASE.quantile(.75):,.1f}", sig(p_mw)),
    ("P95 (acres)",                f"{BASE.quantile(.95):,.1f}", f"{ANOM.quantile(.95):,.1f}", f"+{ANOM.quantile(.95)-BASE.quantile(.95):,.1f}", "—"),
    ("IQR (acres)",                f"{BASE.quantile(.75)-BASE.quantile(.25):,.1f}", f"{ANOM.quantile(.75)-ANOM.quantile(.25):,.1f}", "—", "—"),
    (f"Events > IQR fence ({fence:,.0f})", f"{(BASE>fence).sum()} ({100*(BASE>fence).mean():.1f}%)", f"{(ANOM>fence).sum()} ({100*(ANOM>fence).mean():.1f}%)", "—", sig(p_mw)),
    ("GDACS share",                f"{event_df[event_df.season_label==0]['agency_gdacs'].mean():.1%}", f"{event_df[event_df.season_label==1]['agency_gdacs'].mean():.1%}", "—", sig(p_mw)),
    ("Median ratio (2025/2024)",   "1.00x",                   f"{ANOM.median()/BASE.median():.2f}x", "—", "—"),
    ("Mean ratio (2025/2024)",     "1.00x",                   f"{ANOM.mean()/BASE.mean():.2f}x",     "—", "—"),
    ("Cohen's d",                  "—",                       f"{cohens_d:.3f}",                      "—", "—"),
    ("Mann-Whitney p",             "—",                       f"{p_mw:.2e}",                          "—", sig(p_mw)),
    ("Welch's t p",                "—",                       f"{p_t:.2e}",                           "—", sig(p_t)),
]

cols     = ["Metric", "2024-2025 Baseline", "2025-2026 Mega-Fire", "Delta", "Sig."]
table_df = pd.DataFrame(rows, columns=cols)

print("summary statistics:")
print(table_df.to_string(index=False))
print(f"\n*** p<0.001  ** p<0.01  * p<0.05")
print(f"Mann-Whitney U={u_stat:,.0f}  p={p_mw:.2e}")
print(f"Welch's t={t_stat:.4f}  p={p_t:.2e}")
print(f"Cohen's d={cohens_d:.3f}  ({'large' if abs(cohens_d)>=0.8 else 'medium' if abs(cohens_d)>=0.5 else 'small'})")

# Publication table figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns,
               cellLoc="center", loc="center",
               colWidths=[0.34, 0.18, 0.18, 0.18, 0.08])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.55)
for col_idx in range(len(cols)):
    tbl[0, col_idx].set_facecolor("#2C3E50")
    tbl[0, col_idx].set_text_props(color="white", fontweight="bold")
for row_idx in range(1, len(rows) + 1):
    base_color = "#F2F3F4" if row_idx % 2 == 0 else "white"
    for col_idx in range(len(cols)):
        tbl[row_idx, col_idx].set_facecolor(base_color)
    if rows[row_idx-1][4] == "***":
        tbl[row_idx, 4].set_facecolor("#1E8449")
        tbl[row_idx, 4].set_text_props(color="white", fontweight="bold")
ax.set_title("Table 1 — Summary Statistics: 2024-2025 Baseline vs 2025-2026 Mega-Fire\n"
             "Source: NASA EONET API v3  |  *** p<0.001",
             fontsize=10, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig("fig5_evidence_table.png", bbox_inches="tight", dpi=150)
print("saved: fig5_evidence_table.png")

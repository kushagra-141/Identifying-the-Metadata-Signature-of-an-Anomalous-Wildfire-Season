"""
Fire Intelligence System — three real-world tools built on NASA EONET data
Uses only coordinates and event counts — 100% complete across all 11,712 events.
No magnitude data (78% missing pre-2024), no GDACS feature (API artifact).

Tool 1: Location Risk Scorer     — historical fire recurrence by lat/lon
Tool 2: Monthly Anomaly Detector — is this month abnormal vs 2016-2023?
Tool 3: Geographic Scope Monitor — is fire spreading into new regions?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── Load data ──────────────────────────────────────────────────────────────────

raw = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date"])
geo = (raw.dropna(subset=["lat", "lon"])
          .drop_duplicates(subset=["event_id"])
          .copy())

geo["month_num"] = geo["obs_date"].dt.month.fillna(1).astype(int)

print(f"Clean events with coordinates: {len(geo):,}")
print(f"Years: {sorted(geo['year'].unique())}\n")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Location Fire Risk Scorer
# Given a lat/lon, return historical fire recurrence probability
# Use case: insurance underwriting, evacuation pre-planning, land development
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("TOOL 1 — Location Fire Risk Scorer")
print("=" * 65)

GRID = 2.0   # 2-degree grid cells (~220 km at equator)

geo["cell_lat"] = (geo["lat"] // GRID) * GRID
geo["cell_lon"] = (geo["lon"] // GRID) * GRID
geo["cell"]     = (geo["cell_lat"].astype(str) + "_" +
                   geo["cell_lon"].astype(str))

# Count years each cell had fires
cell_years = geo.groupby("cell")["year"].nunique()
cell_coords = geo.groupby("cell")[["cell_lat","cell_lon"]].first()

# Recurrence: for each cell-year pair, did the cell also burn the following year?
cell_year_pivot = (geo.groupby(["cell","year"])
                      .size()
                      .unstack(fill_value=0)
                      .clip(upper=1))   # 1 = burned that year

recurrence_rows = []
for cell in cell_year_pivot.index:
    years_with_fire = [y for y in cell_year_pivot.columns
                       if cell_year_pivot.loc[cell, y] == 1]
    for yr in years_with_fire:
        recurred = (yr + 1) in years_with_fire and (yr + 1) in cell_year_pivot.columns
        recurrence_rows.append({
            "cell":       cell,
            "year":       yr,
            "n_yrs_fire": len(years_with_fire),
            "recurred":   int(recurred),
        })

rec_df = pd.DataFrame(recurrence_rows)

# Recurrence probability table
print("\n  Historical recurrence probability by fire frequency:")
print(f"  {'Years burned':>15}  {'Cells':>7}  {'Recur prob':>12}  {'Real-world meaning'}")
print("  " + "-" * 70)
thresholds = [(1,1), (2,3), (4,5), (6,8), (9,15)]
risk_table = {}
for lo, hi in thresholds:
    mask  = (rec_df["n_yrs_fire"] >= lo) & (rec_df["n_yrs_fire"] <= hi)
    sub   = rec_df[mask]
    cells = sub["cell"].nunique()
    prob  = sub["recurred"].mean() if len(sub) > 0 else 0
    label = f"{lo}" if lo == hi else f"{lo}-{hi}"
    if lo == 1 and hi == 1:
        meaning = "Isolated event — low structural risk"
    elif lo <= 3:
        meaning = "Occasional fire zone — moderate risk"
    elif lo <= 5:
        meaning = "Recurring fire zone — elevated risk"
    elif lo <= 8:
        meaning = "High-frequency zone — high risk"
    else:
        meaning = "Chronic fire zone — very high risk"
    risk_table[label] = prob
    print(f"  {label:>15}  {cells:>7}  {prob:>11.1%}  {meaning}")

# The scorer function
def score_location(lat: float, lon: float) -> dict:
    cell_lat = (lat // GRID) * GRID
    cell_lon = (lon // GRID) * GRID
    cell_id  = f"{cell_lat}_{cell_lon}"
    if cell_id not in cell_years.index:
        return {"risk_level": "UNKNOWN", "recurrence_prob": None,
                "years_burned": 0, "message": "No fire history in this cell (2016-2026)"}
    n_yrs = cell_years[cell_id]
    sub   = rec_df[rec_df["cell"] == cell_id]
    prob  = sub["recurred"].mean() if len(sub) > 0 else 0.0
    if n_yrs >= 9:   risk = "VERY HIGH"
    elif n_yrs >= 6: risk = "HIGH"
    elif n_yrs >= 4: risk = "ELEVATED"
    elif n_yrs >= 2: risk = "MODERATE"
    else:            risk = "LOW"
    return {"risk_level": risk, "recurrence_prob": round(prob, 3),
            "years_burned": int(n_yrs),
            "message": f"Cell burned in {n_yrs}/10 years; {prob:.0%} annual recurrence"}

# Demo: score real locations
print("\n  Sample location queries:")
locations = [
    ("Los Angeles, CA (wildland-urban interface)", 34.05, -118.24),
    ("Siberia, Russia (boreal forest)",            60.00,  100.00),
    ("Sydney, Australia (bushfire zone)",          -33.87,  151.21),
    ("Amazon, Brazil (tropical forest)",           -3.00,  -60.00),
    ("London, UK (low fire risk)",                 51.50,   -0.12),
    ("Northern California (high risk zone)",       39.00,  -121.00),
    ("Portugal (Mediterranean fire zone)",         39.50,   -8.00),
    ("Central Africa (savanna fires)",              0.00,   25.00),
]
print(f"\n  {'Location':<45} {'Risk':>10}  {'Prob':>6}  {'Yrs burned':>10}")
print("  " + "-" * 75)
for name, lat, lon in locations:
    result = score_location(lat, lon)
    prob   = f"{result['recurrence_prob']:.0%}" if result['recurrence_prob'] is not None else "  N/A"
    print(f"  {name:<45} {result['risk_level']:>10}  {prob:>6}  "
          f"{result['years_burned']:>10}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Monthly Anomaly Detector
# Is this month's fire activity abnormal vs 2016-2023 baseline?
# Use case: fire agency early warning, resource pre-positioning
# Features: log(event_count), irwin_share, month_num
# Deliberately excludes gdacs_share (EONET API artifact)
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 65)
print("TOOL 2 — Monthly Anomaly Detector")
print("=" * 65)

geo["agency_irwin"] = geo["sources"].str.contains("IRWIN", na=False).astype(int)

monthly = (geo.groupby(["year", "month_num"])
           .agg(
               event_count = ("event_id",    "nunique"),
               irwin_share = ("agency_irwin","mean"),
               n_countries = ("cell",        "nunique"),   # geographic spread
           )
           .reset_index())

monthly["log_count"]  = np.log1p(monthly["event_count"])
monthly["label"]      = (monthly["year"] >= 2024).astype(int)

# IQR anomaly threshold from baseline months only
base = monthly[monthly["label"] == 0]
anom = monthly[monthly["label"] == 1]

Q1, Q3 = base["event_count"].quantile(0.25), base["event_count"].quantile(0.75)
IQR    = Q3 - Q1
fence  = Q3 + 1.5 * IQR

print(f"\n  Baseline (2016-2023) monthly event stats:")
print(f"    Median  : {base['event_count'].median():.0f} events/month")
print(f"    IQR     : {IQR:.0f}")
print(f"    Fence   : {fence:.0f} events/month  (above = anomalous)")
print(f"\n  Anomalous (2024-2026) monthly event stats:")
print(f"    Median  : {anom['event_count'].median():.0f} events/month")

base_flagged = (base["event_count"] > fence).sum()
anom_flagged = (anom["event_count"] > fence).sum()
print(f"\n  IQR detection rate:")
print(f"    Baseline months flagged : {base_flagged}/{len(base)} "
      f"({100*base_flagged/len(base):.1f}%)  <- false alarm rate")
print(f"    Anomalous months flagged: {anom_flagged}/{len(anom)} "
      f"({100*anom_flagged/len(anom):.1f}%)  <- detection rate")

u_stat, p_mw = scipy_stats.mannwhitneyu(
    anom["event_count"], base["event_count"], alternative="greater"
)
print(f"\n  Mann-Whitney U={u_stat:,.0f},  p={p_mw:.2e}")

# Decision Tree anomaly detector
FEATURES = ["log_count", "irwin_share", "n_countries", "month_num"]
X = monthly[FEATURES].values
y = monthly["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt  = DecisionTreeClassifier(criterion="gini", max_depth=3,
                              min_samples_leaf=5, class_weight="balanced",
                              random_state=42)
cv  = cross_val_score(dt, X, y, cv=skf, scoring="accuracy")
dt.fit(X, y)

print(f"\n  Decision Tree (5-fold stratified CV):")
print(f"    CV mean accuracy : {cv.mean():.4f}  ({cv.mean()*100:.1f}%)")
print(f"    CV std dev       : +-{cv.std():.4f}")
print(f"    Fold scores      : {[round(s,3) for s in cv]}")

print(f"\n  Tree rules:")
print(export_text(dt, feature_names=FEATURES))

# Live scorer function
def score_month(event_count: int, irwin_share: float,
                n_countries: int, month_num: int) -> dict:
    log_c = np.log1p(event_count)
    pred  = dt.predict([[log_c, irwin_share, n_countries, month_num]])[0]
    prob  = dt.predict_proba([[log_c, irwin_share, n_countries, month_num]])[0][1]
    iqr   = event_count > fence
    return {
        "classifier_label": "ANOMALOUS" if pred == 1 else "normal",
        "anomaly_probability": round(float(prob), 3),
        "iqr_flag": "ANOMALOUS" if iqr else "normal",
        "event_count": event_count,
        "fence": int(fence),
    }

# Demo
print("  Sample month queries:")
demos = [
    ("Typical Jan 2020 (baseline)",  12, 0.95, 8,  1),
    ("Peak Aug 2020 (baseline)",    133, 0.90, 15,  8),
    ("Jan 2024 (anomalous start)",  302, 0.30, 35,  1),
    ("Jul 2025 (anomalous peak)",   446, 0.28, 42,  7),
    ("Hypothetical: 400 events",    400, 0.30, 40,  6),
    ("Hypothetical: 50 events",      50, 0.90, 12,  5),
]
print(f"\n  {'Scenario':<38} {'IQR flag':>10}  {'ML label':>10}  {'Prob':>6}")
print("  " + "-" * 70)
for name, cnt, irw, nc, mo in demos:
    r = score_month(cnt, irw, nc, mo)
    print(f"  {name:<38} {r['iqr_flag']:>10}  "
          f"{r['classifier_label']:>10}  {r['anomaly_probability']:>6.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Geographic Scope Monitor
# Is fire activity spreading into new regions this year?
# Use case: international aid pre-positioning, climate monitoring
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 65)
print("TOOL 3 — Geographic Scope Monitor")
print("=" * 65)

yearly_geo = (geo.groupby("year")
              .agg(
                  n_events    = ("event_id",  "nunique"),
                  mean_lat    = ("lat",        "mean"),
                  mean_lon    = ("lon",        "mean"),
                  abs_lat     = ("lat",        lambda x: x.abs().mean()),
                  lat_std     = ("lat",        "std"),
                  pct_north   = ("lat",        lambda x: (x > 0).mean()),
                  unique_cells= ("cell",       "nunique"),
              )
              .round(3))

# Linear regression on absolute latitude (distance from equator)
slope, intercept, r, p, se = scipy_stats.linregress(
    yearly_geo.index, yearly_geo["abs_lat"]
)
# Confidence interval for slope
t_crit = scipy_stats.t.ppf(0.975, df=len(yearly_geo)-2)
ci_lo  = slope - t_crit * se
ci_hi  = slope + t_crit * se

print(f"\n  Latitude trend (2016-2026):")
print(f"    Slope    : {slope:.3f} degrees/year")
print(f"    95% CI   : [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"    r²       : {r**2:.3f}")
print(f"    p-value  : {p:.3f}")
print(f"    Meaning  : fires moving "
      f"{'toward equator' if slope < 0 else 'toward poles'} "
      f"at {abs(slope):.2f} deg/yr")

print(f"\n  Year-by-year geographic summary:")
print(f"  {'Year':>6}  {'Events':>7}  {'MeanLat':>8}  "
      f"{'AbsLat':>7}  {'%North':>7}  {'Cells':>6}  Era")
print("  " + "-" * 65)
for yr, row in yearly_geo.iterrows():
    era = "ANOMALOUS" if yr >= 2024 else "baseline"
    print(f"  {yr:>6}  {row.n_events:>7,}  {row.mean_lat:>8.2f}  "
          f"{row.abs_lat:>7.2f}  {row.pct_north:>6.1%}  "
          f"{int(row.unique_cells):>6}  {era}")

# Scope scorer function
baseline_abs_lat = yearly_geo[yearly_geo.index < 2024]["abs_lat"]
lat_mean_b = baseline_abs_lat.mean()
lat_std_b  = baseline_abs_lat.std()

def score_geographic_scope(year_data_mean_abs_lat: float,
                            year_data_unique_cells: int) -> dict:
    z = (year_data_mean_abs_lat - lat_mean_b) / lat_std_b
    baseline_cells = yearly_geo[yearly_geo.index < 2024]["unique_cells"].mean()
    cell_ratio = year_data_unique_cells / baseline_cells
    if z < -2:    geo_status = "MAJOR EQUATORWARD SHIFT"
    elif z < -1:  geo_status = "MODERATE EQUATORWARD SHIFT"
    elif z < 1:   geo_status = "WITHIN NORMAL RANGE"
    else:         geo_status = "POLEWARD SHIFT"
    return {"geo_status": geo_status, "z_score": round(z, 2),
            "cell_ratio": round(cell_ratio, 2),
            "new_area_pct": f"{(cell_ratio-1)*100:+.0f}% vs baseline avg"}

print(f"\n  Sample year scope assessments:")
test_years = [(2020, yearly_geo.loc[2020,"abs_lat"], int(yearly_geo.loc[2020,"unique_cells"])),
              (2023, yearly_geo.loc[2023,"abs_lat"], int(yearly_geo.loc[2023,"unique_cells"])),
              (2024, yearly_geo.loc[2024,"abs_lat"], int(yearly_geo.loc[2024,"unique_cells"])),
              (2025, yearly_geo.loc[2025,"abs_lat"], int(yearly_geo.loc[2025,"unique_cells"]))]
print(f"\n  {'Year':>6}  {'Status':<28}  {'Z-score':>8}  {'Area vs baseline':>18}")
print("  " + "-" * 68)
for yr, al, uc in test_years:
    r = score_geographic_scope(al, uc)
    print(f"  {yr:>6}  {r['geo_status']:<28}  {r['z_score']:>8.2f}  "
          f"{r['new_area_pct']:>18}")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS — one figure per tool
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure A: Risk map (top 300 highest-recurrence cells) ────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#D6EAF8")
ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)

# All cells — colour by recurrence probability
cell_prob = (rec_df.groupby("cell")["recurred"].mean()
             .rename("prob").reset_index())
cell_prob = cell_prob.merge(
    cell_coords.reset_index(), on="cell", how="left"
).dropna()

sc = ax.scatter(
    cell_prob["cell_lon"], cell_prob["cell_lat"],
    c=cell_prob["prob"], cmap="YlOrRd",
    s=30, alpha=0.7, linewidths=0,
    vmin=0, vmax=1,
)
plt.colorbar(sc, ax=ax, label="Annual fire recurrence probability",
             shrink=0.6)
for name, lat, lon in [("N.America",-100,45),("S.America",-60,-15),
                        ("Europe",15,52),("Africa",20,5),
                        ("Asia",100,45),("Australia",135,-25)]:
    ax.text(lon, lat, name, fontsize=7, color="#333333",
            ha="center", alpha=0.5)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Tool 1 — Global Fire Risk Map (2016-2026)\n"
             "Colour = probability cell burns in any given year | "
             "Based on 10-year recurrence analysis",
             fontweight="bold")
fig.tight_layout()
fig.savefig("tool1_risk_map.png", bbox_inches="tight")
print("\n\nSaved: tool1_risk_map.png")
plt.close(fig)

# ── Figure B: Monthly anomaly detector ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for yr in sorted(monthly["year"].unique()):
    sub   = monthly[monthly["year"] == yr].sort_values("month_num")
    color = "#E84C4C" if yr >= 2024 else "#4C9BE8"
    alpha = 0.9 if yr >= 2024 else 0.35
    lw    = 2.0 if yr >= 2024 else 0.8
    ax.plot(sub["month_num"], sub["event_count"],
            color=color, alpha=alpha, linewidth=lw,
            label=str(yr) if yr >= 2024 else None)
ax.axhline(fence, color="orange", linewidth=2, linestyle="--",
           label=f"Anomaly fence ({fence:.0f} events)")
ax.set_title("Tool 2 — Monthly Event Count by Year\n"
             "Red = 2024-2026 | Blue = 2016-2023 baseline")
ax.set_xlabel("Month"); ax.set_ylabel("Events")
ax.set_xticks(range(1,13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.legend(fontsize=8)

ax = axes[1]
bp = ax.boxplot(
    [base["event_count"].values, anom["event_count"].values],
    tick_labels=["2016-2023\nBaseline", "2024-2026\nAnomaly"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
)
bp["boxes"][0].set_facecolor("#4C9BE8")
bp["boxes"][1].set_facecolor("#E84C4C")
ax.axhline(fence, color="orange", linewidth=2, linestyle="--",
           label=f"IQR fence ({fence:.0f})")
ax.set_title(f"Monthly Distribution: Baseline vs Anomaly\n"
             f"Detection rate {100*anom_flagged/len(anom):.0f}%  |  "
             f"False alarm rate {100*base_flagged/len(base):.0f}%")
ax.set_ylabel("Events per month")
ax.legend(fontsize=8)

fig.suptitle("Tool 2 — Monthly Fire Activity Anomaly Detector",
             fontweight="bold")
fig.tight_layout()
fig.savefig("tool2_monthly_detector.png", bbox_inches="tight")
print("Saved: tool2_monthly_detector.png")
plt.close(fig)

# ── Figure C: Geographic scope monitor ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
colors = ["#E84C4C" if yr >= 2024 else "#4C9BE8"
          for yr in yearly_geo.index]
ax.bar(yearly_geo.index, yearly_geo["abs_lat"],
       color=colors, edgecolor="white", width=0.7)
x_fit = np.array(yearly_geo.index)
ax.plot(x_fit, intercept + slope * x_fit,
        color="black", linewidth=2, linestyle="--",
        label=f"Trend: {slope:.2f} deg/yr (p={p:.3f})")
ax.fill_between(x_fit,
                (intercept + slope*x_fit) - t_crit*se*np.sqrt(len(x_fit)),
                (intercept + slope*x_fit) + t_crit*se*np.sqrt(len(x_fit)),
                alpha=0.15, color="black")
ax.set_title("Tool 3 — Mean Absolute Latitude of Fires per Year\n"
             "Decreasing = fires moving toward equator")
ax.set_xlabel("Year"); ax.set_ylabel("Mean absolute latitude (degrees)")
ax.set_xticks(list(yearly_geo.index))
ax.set_xticklabels(list(yearly_geo.index), rotation=45)
ax.legend(fontsize=8)
patch_b = plt.Rectangle((0,0),1,1,color="#4C9BE8",label="Baseline")
patch_a = plt.Rectangle((0,0),1,1,color="#E84C4C",label="Anomalous")
ax.legend(handles=[patch_b, patch_a,
                   plt.Line2D([0],[0],color="black",linestyle="--",
                              label=f"Trend {slope:.2f} deg/yr")],
          fontsize=8)

ax = axes[1]
ax.plot(yearly_geo.index, yearly_geo["pct_north"] * 100,
        "o-", linewidth=2.5, color="#8E44AD", markersize=8)
ax.axvline(2023.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
ax.axhline(50, color="grey", linewidth=1, linestyle=":", alpha=0.6)
ax.fill_between(yearly_geo.index, yearly_geo["pct_north"]*100,
                50, alpha=0.1, color="#8E44AD")
for yr, row in yearly_geo.iterrows():
    ax.text(yr, row.pct_north*100 + 2, f"{row.pct_north:.0%}",
            ha="center", fontsize=7)
ax.set_title("Northern Hemisphere Share of Fire Events per Year\n"
             "Drop in 2024+ = Southern Hemisphere fires entering dataset")
ax.set_xlabel("Year"); ax.set_ylabel("% events in Northern Hemisphere")
ax.set_ylim(0, 115)
ax.set_xticks(list(yearly_geo.index))
ax.set_xticklabels(list(yearly_geo.index), rotation=45)

fig.suptitle("Tool 3 — Geographic Scope Monitor",
             fontweight="bold")
fig.tight_layout()
fig.savefig("tool3_geographic_scope.png", bbox_inches="tight")
print("Saved: tool3_geographic_scope.png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 65)
print("SYSTEM SUMMARY")
print("=" * 65)
print(f"""
  Tool 1 — Location Fire Risk Scorer
    Input  : latitude, longitude
    Output : risk level + annual recurrence probability
    Accuracy: historical recurrence rate (not a classifier)
    Use case: insurance underwriting, land-use planning,
              evacuation pre-planning

  Tool 2 — Monthly Activity Anomaly Detector
    Input  : monthly event count + agency mix + geographic spread
    Output : normal / ANOMALOUS + probability score
    Accuracy: {cv.mean()*100:.1f}% (5-fold stratified CV, +-{cv.std()*100:.1f}%)
    IQR detection rate : {100*anom_flagged/len(anom):.0f}% of anomalous months caught
    IQR false alarm    : {100*base_flagged/len(base):.0f}% of baseline months mis-flagged
    Use case: fire agency resource pre-positioning,
              government emergency budget alerts

  Tool 3 — Geographic Scope Monitor
    Input  : full year of EONET event locations
    Output : geographic status vs historical baseline
    Trend  : {slope:.2f} deg/yr equatorward drift (p={p:.3f})
    Use case: international aid pre-positioning,
              climate change monitoring, reinsurance pricing
""")

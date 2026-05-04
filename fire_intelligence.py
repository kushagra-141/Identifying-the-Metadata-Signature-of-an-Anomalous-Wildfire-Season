"""
Three operational tools built on 10 years of NASA EONET wildfire data.
Uses coordinates and event counts — 100% complete across all events.
Magnitude (sparse pre-2024) and GDACS agency (integration artefact) excluded.
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
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

raw = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date"])
geo = (raw.dropna(subset=["lat", "lon"])
          .drop_duplicates(subset=["event_id"])
          .copy())
geo["month_num"]    = geo["obs_date"].dt.month.fillna(1).astype(int)
geo["agency_irwin"] = geo["sources"].str.contains("IRWIN", na=False).astype(int)

print(f"events with coordinates: {len(geo):,}  years: {sorted(geo['year'].unique())}")


# Tool 1: Location Fire Risk Scorer
# Divides the globe into 2-degree grid cells and computes the fraction of
# years each cell had at least one fire event (recurrence probability).

GRID = 2.0
geo["cell_lat"] = (geo["lat"] // GRID) * GRID
geo["cell_lon"] = (geo["lon"] // GRID) * GRID
geo["cell"]     = geo["cell_lat"].astype(str) + "_" + geo["cell_lon"].astype(str)

cell_years  = geo.groupby("cell")["year"].nunique()
cell_coords = geo.groupby("cell")[["cell_lat", "cell_lon"]].first()

cell_year_pivot = (geo.groupby(["cell", "year"]).size()
                      .unstack(fill_value=0).clip(upper=1))

recurrence_rows = []
for cell in cell_year_pivot.index:
    years_with_fire = [y for y in cell_year_pivot.columns if cell_year_pivot.loc[cell, y] == 1]
    for yr in years_with_fire:
        recurred = (yr + 1) in years_with_fire and (yr + 1) in cell_year_pivot.columns
        recurrence_rows.append({
            "cell": cell, "year": yr,
            "n_yrs_fire": len(years_with_fire),
            "recurred": int(recurred),
        })
rec_df = pd.DataFrame(recurrence_rows)

print("\nrecurrence probability by fire frequency (tool 1):")
print(f"  {'years burned':>12}  {'cells':>6}  {'recur prob':>10}")
for lo, hi in [(1,1),(2,3),(4,5),(6,8),(9,15)]:
    mask  = (rec_df["n_yrs_fire"] >= lo) & (rec_df["n_yrs_fire"] <= hi)
    sub   = rec_df[mask]
    label = f"{lo}" if lo == hi else f"{lo}-{hi}"
    prob  = sub["recurred"].mean() if len(sub) > 0 else 0
    print(f"  {label:>12}  {sub['cell'].nunique():>6}  {prob:>10.1%}")


def score_location(lat: float, lon: float) -> dict:
    cell_id = f"{(lat//GRID)*GRID}_{(lon//GRID)*GRID}"
    if cell_id not in cell_years.index:
        return {"risk_level": "UNKNOWN", "recurrence_prob": None, "years_burned": 0}
    n_yrs = cell_years[cell_id]
    prob  = rec_df[rec_df["cell"] == cell_id]["recurred"].mean() if n_yrs > 0 else 0.0
    risk  = "VERY HIGH" if n_yrs >= 9 else "HIGH" if n_yrs >= 6 else \
            "ELEVATED"  if n_yrs >= 4 else "MODERATE" if n_yrs >= 2 else "LOW"
    return {"risk_level": risk, "recurrence_prob": round(float(prob), 3), "years_burned": int(n_yrs)}


print("\nsample location risk scores:")
locations = [
    ("Los Angeles, CA",        34.05, -118.24),
    ("Northern California",    39.00, -121.00),
    ("Sydney, Australia",     -33.87,  151.21),
    ("London, UK",             51.50,   -0.12),
    ("Central Africa",          0.00,   25.00),
    ("Portugal",               39.50,   -8.00),
]
print(f"  {'location':<30} {'risk':>10}  {'prob':>6}  {'years':>6}")
for name, lat, lon in locations:
    r    = score_location(lat, lon)
    prob = f"{r['recurrence_prob']:.0%}" if r['recurrence_prob'] is not None else "N/A"
    print(f"  {name:<30} {r['risk_level']:>10}  {prob:>6}  {r['years_burned']:>6}")


# Tool 2: Monthly Activity Anomaly Detector
# GDACS excluded because it was only integrated into EONET around 2023-2024,
# making it unreliable as a feature across the full 10-year baseline.

monthly = (geo.groupby(["year", "month_num"])
           .agg(
               event_count = ("event_id",    "nunique"),
               irwin_share = ("agency_irwin","mean"),
               n_countries = ("cell",        "nunique"),
           )
           .reset_index())
monthly["log_count"] = np.log1p(monthly["event_count"])
monthly["label"]     = (monthly["year"] >= 2024).astype(int)

base = monthly[monthly["label"] == 0]
anom = monthly[monthly["label"] == 1]
Q1, Q3 = base["event_count"].quantile(0.25), base["event_count"].quantile(0.75)
fence  = Q3 + 1.5 * (Q3 - Q1)

base_flagged = (base["event_count"] > fence).sum()
anom_flagged = (anom["event_count"] > fence).sum()
u_stat, p_mw = scipy_stats.mannwhitneyu(anom["event_count"], base["event_count"], alternative="greater")

print(f"\nmonthly anomaly detector (tool 2):")
print(f"  baseline median : {base['event_count'].median():.0f} events/month")
print(f"  anomaly  median : {anom['event_count'].median():.0f} events/month")
print(f"  IQR fence       : {fence:.0f} events/month")
print(f"  detection rate  : {anom_flagged}/{len(anom)} ({100*anom_flagged/len(anom):.1f}%)")
print(f"  false alarm rate: {base_flagged}/{len(base)} ({100*base_flagged/len(base):.1f}%)")
print(f"  Mann-Whitney    : U={u_stat:,.0f}  p={p_mw:.2e}")

FEATURES = ["log_count", "irwin_share", "n_countries", "month_num"]
X = monthly[FEATURES].values
y = monthly["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt  = DecisionTreeClassifier(criterion="gini", max_depth=3,
                              min_samples_leaf=5, class_weight="balanced",
                              random_state=42)
cv  = cross_val_score(dt, X, y, cv=skf, scoring="accuracy")
dt.fit(X, y)

print(f"  CV accuracy     : {cv.mean():.4f} (+/-{cv.std():.4f})")
print(f"  fold scores     : {[round(s,3) for s in cv]}")
print(f"\n  tree rules:\n{export_text(dt, feature_names=FEATURES)}")


def score_month(event_count: int, irwin_share: float, n_countries: int, month_num: int) -> dict:
    x    = [[np.log1p(event_count), irwin_share, n_countries, month_num]]
    pred = dt.predict(x)[0]
    prob = dt.predict_proba(x)[0][1]
    return {
        "label":       "ANOMALOUS" if pred == 1 else "normal",
        "probability": round(float(prob), 3),
        "iqr_flag":    "ANOMALOUS" if event_count > fence else "normal",
    }


# Tool 3: Geographic Scope Monitor
# Linear regression on mean absolute latitude to detect equatorward drift.

yearly_geo = (geo.groupby("year")
              .agg(
                  n_events    = ("event_id",   "nunique"),
                  mean_lat    = ("lat",         "mean"),
                  abs_lat     = ("lat",         lambda x: x.abs().mean()),
                  pct_north   = ("lat",         lambda x: (x > 0).mean()),
                  unique_cells= ("cell",        "nunique"),
              )
              .round(3))

slope, intercept, r, p, se = scipy_stats.linregress(yearly_geo.index, yearly_geo["abs_lat"])
t_crit = scipy_stats.t.ppf(0.975, df=len(yearly_geo) - 2)

print(f"\ngeographic scope monitor (tool 3):")
print(f"  latitude trend  : {slope:.3f} deg/yr  95% CI=[{slope-t_crit*se:.3f}, {slope+t_crit*se:.3f}]")
print(f"  r²={r**2:.3f}  p={p:.3f}")

baseline_abs_lat = yearly_geo[yearly_geo.index < 2024]["abs_lat"]
lat_mean_b, lat_std_b = baseline_abs_lat.mean(), baseline_abs_lat.std()

print(f"\n  {'year':>6}  {'events':>7}  {'abs_lat':>8}  {'z-score':>8}  {'%north':>7}  era")
for yr, row in yearly_geo.iterrows():
    z   = (row.abs_lat - lat_mean_b) / lat_std_b
    era = "ANOMALOUS" if yr >= 2024 else "baseline"
    print(f"  {yr:>6}  {row.n_events:>7,}  {row.abs_lat:>8.2f}  {z:>8.2f}  {row.pct_north:>6.1%}  {era}")


def score_geographic_scope(mean_abs_lat: float, unique_cells: int) -> dict:
    z          = (mean_abs_lat - lat_mean_b) / lat_std_b
    cell_ratio = unique_cells / yearly_geo[yearly_geo.index < 2024]["unique_cells"].mean()
    status     = ("MAJOR EQUATORWARD SHIFT"    if z < -2 else
                  "MODERATE EQUATORWARD SHIFT" if z < -1 else
                  "WITHIN NORMAL RANGE"        if z <  1 else "POLEWARD SHIFT")
    return {"status": status, "z_score": round(z, 2), "cell_ratio": round(cell_ratio, 2)}


# Visualisations

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_facecolor("#D6EAF8")
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)
cell_prob = (rec_df.groupby("cell")["recurred"].mean()
             .rename("prob").reset_index()
             .merge(cell_coords.reset_index(), on="cell", how="left")
             .dropna())
sc = ax.scatter(cell_prob["cell_lon"], cell_prob["cell_lat"],
                c=cell_prob["prob"], cmap="YlOrRd", s=30, alpha=0.7,
                linewidths=0, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label="Annual fire recurrence probability", shrink=0.6)
for name, lat, lon in [("N.America",-100,45),("S.America",-60,-15),("Europe",15,52),
                        ("Africa",20,5),("Asia",100,45),("Australia",135,-25)]:
    ax.text(lon, lat, name, fontsize=7, color="#333333", ha="center", alpha=0.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Tool 1 — Global Fire Risk Map (2016-2026)\n"
             "Colour = probability cell burns in any given year", fontweight="bold")
fig.tight_layout()
fig.savefig("tool1_risk_map.png", bbox_inches="tight")
print("\nsaved: tool1_risk_map.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for yr in sorted(monthly["year"].unique()):
    sub   = monthly[monthly["year"] == yr].sort_values("month_num")
    color = "#E84C4C" if yr >= 2024 else "#4C9BE8"
    ax.plot(sub["month_num"], sub["event_count"], color=color,
            alpha=0.9 if yr >= 2024 else 0.35,
            linewidth=2.0 if yr >= 2024 else 0.8,
            label=str(yr) if yr >= 2024 else None)
ax.axhline(fence, color="orange", linewidth=2, linestyle="--",
           label=f"Anomaly fence ({fence:.0f} events)")
ax.set_title("Tool 2 — Monthly Event Count by Year\nRed = 2024-2026  |  Blue = 2016-2023")
ax.set_xlabel("Month")
ax.set_ylabel("Events")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.legend(fontsize=8)

ax = axes[1]
bp = ax.boxplot([base["event_count"].values, anom["event_count"].values],
                tick_labels=["2016-2023\nBaseline", "2024-2026\nAnomaly"],
                patch_artist=True, medianprops=dict(color="white", linewidth=2.5))
bp["boxes"][0].set_facecolor("#4C9BE8")
bp["boxes"][1].set_facecolor("#E84C4C")
ax.axhline(fence, color="orange", linewidth=2, linestyle="--", label=f"IQR fence ({fence:.0f})")
ax.set_title(f"Monthly Distribution\nDetection {100*anom_flagged/len(anom):.0f}%  |  False alarm {100*base_flagged/len(base):.0f}%")
ax.set_ylabel("Events per month")
ax.legend(fontsize=8)
fig.suptitle("Tool 2 — Monthly Fire Activity Anomaly Detector", fontweight="bold")
fig.tight_layout()
fig.savefig("tool2_monthly_detector.png", bbox_inches="tight")
print("saved: tool2_monthly_detector.png")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.bar(yearly_geo.index, yearly_geo["abs_lat"],
       color=["#E84C4C" if yr >= 2024 else "#4C9BE8" for yr in yearly_geo.index],
       edgecolor="white", width=0.7)
x_fit = np.array(yearly_geo.index)
ax.plot(x_fit, intercept + slope * x_fit, color="black", linewidth=2, linestyle="--",
        label=f"Trend {slope:.2f} deg/yr (p={p:.3f})")
ax.set_title("Tool 3 — Mean Absolute Latitude per Year\nDecreasing = fires moving toward equator")
ax.set_xlabel("Year")
ax.set_ylabel("Mean absolute latitude (degrees)")
ax.set_xticks(list(yearly_geo.index))
ax.set_xticklabels(list(yearly_geo.index), rotation=45)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(yearly_geo.index, yearly_geo["pct_north"] * 100,
        "o-", linewidth=2.5, color="#8E44AD", markersize=8)
ax.axvline(2023.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
ax.axhline(50, color="grey", linewidth=1, linestyle=":", alpha=0.6)
for yr, row in yearly_geo.iterrows():
    ax.text(yr, row.pct_north * 100 + 2, f"{row.pct_north:.0%}", ha="center", fontsize=7)
ax.set_title("Northern Hemisphere Share of Fire Events\nDrop in 2024+ = Southern Hemisphere fires entering dataset")
ax.set_xlabel("Year")
ax.set_ylabel("% events in Northern Hemisphere")
ax.set_ylim(0, 115)
ax.set_xticks(list(yearly_geo.index))
ax.set_xticklabels(list(yearly_geo.index), rotation=45)
fig.suptitle("Tool 3 — Geographic Scope Monitor", fontweight="bold")
fig.tight_layout()
fig.savefig("tool3_geographic_scope.png", bbox_inches="tight")
print("saved: tool3_geographic_scope.png")
plt.close(fig)

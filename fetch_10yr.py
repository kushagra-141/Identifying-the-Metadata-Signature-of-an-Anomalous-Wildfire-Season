"""
10-Year EONET Wildfire Fetch
Queries month-by-month from Jan 2016 to Apr 2026 to avoid the 500-event
recency ceiling. Each monthly window has far fewer than 500 events, so
every call returns the complete picture for that month.
"""

import requests
import pandas as pd
import time
from datetime import date
from calendar import monthrange

BASE_URL  = "https://eonet.gsfc.nasa.gov/api/v3/events"
CACHE     = "eonet_10yr_combined.csv"
START_YR, START_MO = 2016, 1
END_YR,   END_MO   = 2026, 4


# ── Month generator ────────────────────────────────────────────────────────────

def months_between(sy, sm, ey, em):
    y, m = sy, sm
    while (y, m) <= (ey, em):
        last_day = monthrange(y, m)[1]
        yield (f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last_day:02d}", y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


# ── Single-month fetch ─────────────────────────────────────────────────────────

def fetch_month(start: str, end: str) -> list[dict]:
    params = {
        "category": "wildfires",
        "status":   "all",
        "start":    start,
        "end":      end,
        "limit":    500,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except requests.RequestException as e:
        print(f"    [!] {start}: {e}")
        return []


# ── Parser (same logic as original) ───────────────────────────────────────────

def parse_events(raw_events: list[dict], year: int) -> pd.DataFrame:
    rows = []
    for evt in raw_events:
        closed_raw  = evt.get("closed")
        source_list = [s["id"] for s in evt.get("sources", [])]
        base = {
            "event_id":     evt.get("id"),
            "title":        evt.get("title"),
            "closed":       closed_raw,
            "is_active":    closed_raw is None,
            "source_count": len(source_list),
            "sources":      "|".join(source_list),
            "year":         year,
        }
        geometries = evt.get("geometry", [])
        if not geometries:
            rows.append({**base, "lon": None, "lat": None,
                         "obs_date": None,
                         "magnitude_value": None, "magnitude_unit": None})
            continue
        for geom in geometries:
            coords = geom.get("coordinates") or []
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None
            rows.append({
                **base,
                "lon":             lon,
                "lat":             lat,
                "obs_date":        geom.get("date"),
                "magnitude_value": geom.get("magnitudeValue"),
                "magnitude_unit":  geom.get("magnitudeUnit"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["obs_date"]        = pd.to_datetime(df["obs_date"], utc=True, errors="coerce")
        df["closed"]          = pd.to_datetime(df["closed"],   utc=True, errors="coerce")
        df["magnitude_value"] = pd.to_numeric(df["magnitude_value"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    return df


# ── Main fetch loop ────────────────────────────────────────────────────────────

def fetch_10yr(cache: bool = True) -> pd.DataFrame:
    if cache:
        try:
            df = pd.read_csv(CACHE, parse_dates=["obs_date", "closed"])
            print(f"[cache] Loaded {len(df):,} rows, "
                  f"{df['event_id'].nunique():,} unique events from {CACHE}")
            return df
        except FileNotFoundError:
            pass

    all_frames = []
    month_list = list(months_between(START_YR, START_MO, END_YR, END_MO))
    total      = len(month_list)

    print(f"Fetching {total} months ({START_YR}-{START_MO:02d} to "
          f"{END_YR}-{END_MO:02d}) ...\n")

    for i, (start, end, yr, mo) in enumerate(month_list, 1):
        raw    = fetch_month(start, end)
        frame  = parse_events(raw, year=yr)
        n_evt  = frame["event_id"].nunique() if not frame.empty else 0
        print(f"  [{i:>3}/{total}] {start[:7]}  ->  {n_evt:>4} events"
              f"  ({len(raw):>4} geometry rows)")
        if not frame.empty:
            all_frames.append(frame)
        time.sleep(0.25)   # polite rate-limiting

    combined = pd.concat(all_frames, ignore_index=True)

    # Drop duplicates — same event can appear in two monthly windows if
    # its obs_date straddles a month boundary (keep first occurrence)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["event_id", "obs_date", "lon", "lat"])
    print(f"\nDeduplication: {before:,} -> {len(combined):,} rows")

    combined.to_csv(CACHE, index=False)
    print(f"[cache] Saved {len(combined):,} rows to {CACHE}")
    return combined


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_10yr(cache=True)

    # One row per event per year
    event_df = (
        df.groupby(["event_id", "year"])
        .agg(magnitude_max=("magnitude_value", "max"),
             sources=("sources", "first"))
        .reset_index()
    )
    event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS").astype(int)

    print("\n=== Events and Mean Magnitude by Year ===")
    summary = (event_df.groupby("year")
               .agg(events         =("event_id",        "nunique"),
                    with_magnitude =("magnitude_max",    lambda x: x.notna().sum()),
                    mean_mag       =("magnitude_max",    "mean"),
                    median_mag     =("magnitude_max",    "median"),
                    gdacs_pct      =("agency_gdacs",     "mean"))
               .round(1))
    summary["gdacs_pct"] = (summary["gdacs_pct"] * 100).round(1)
    print(summary.rename(columns={
        "events":         "n_events",
        "with_magnitude": "has_mag",
        "mean_mag":       "mean_acres",
        "median_mag":     "median_acres",
        "gdacs_pct":      "gdacs_%"
    }).to_string())

    print(f"\nTotal unique events : {event_df['event_id'].nunique():,}")
    print(f"Years covered       : {sorted(event_df['year'].unique())}")

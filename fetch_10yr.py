"""
Fetches wildfire events month-by-month from Jan 2016 to Apr 2026.
Querying by month avoids the 500-event recency ceiling that affects
full-year queries, ensuring complete coverage for each month.
Includes exponential backoff for HTTP 429 rate-limit responses.
"""

import requests
import pandas as pd
import time
from calendar import monthrange

BASE_URL   = "https://eonet.gsfc.nasa.gov/api/v3/events"
CACHE_PATH = "eonet_10yr_combined.csv"
START_YR, START_MO = 2016, 1
END_YR,   END_MO   = 2026, 4


def months_between(sy, sm, ey, em):
    y, m = sy, sm
    while (y, m) <= (ey, em):
        last_day = monthrange(y, m)[1]
        yield (f"{y}-{m:02d}-01", f"{y}-{m:02d}-{last_day:02d}", y, m)
        m += 1
        if m > 12:
            m, y = 1, y + 1


def fetch_month(start: str, end: str, max_retries: int = 5) -> list[dict]:
    params = {"category": "wildfires", "status": "all",
              "start": start, "end": end, "limit": 500}
    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)
                print(f"  rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("events", [])
        except requests.RequestException as e:
            print(f"  error on {start}: {e}")
            time.sleep(10)
    return []


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
                         "obs_date": None, "magnitude_value": None,
                         "magnitude_unit": None})
            continue
        for geom in geometries:
            coords = geom.get("coordinates") or []
            rows.append({
                **base,
                "lon":             coords[0] if len(coords) > 0 else None,
                "lat":             coords[1] if len(coords) > 1 else None,
                "obs_date":        geom.get("date"),
                "magnitude_value": geom.get("magnitudeValue"),
                "magnitude_unit":  geom.get("magnitudeUnit"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["obs_date"]        = pd.to_datetime(df["obs_date"], utc=True, errors="coerce")
        df["closed"]          = pd.to_datetime(df["closed"],   utc=True, errors="coerce")
        df["magnitude_value"] = pd.to_numeric(df["magnitude_value"], errors="coerce")
        df["lon"]             = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"]             = pd.to_numeric(df["lat"], errors="coerce")
    return df


def fetch_10yr(cache: bool = True) -> pd.DataFrame:
    if cache:
        try:
            df = pd.read_csv(CACHE_PATH, parse_dates=["obs_date", "closed"])
            print(f"loaded {len(df):,} rows, {df['event_id'].nunique():,} unique events from cache")
            return df
        except FileNotFoundError:
            pass

    month_list = list(months_between(START_YR, START_MO, END_YR, END_MO))
    total      = len(month_list)
    all_frames = []

    print(f"fetching {total} months ({START_YR}-{START_MO:02d} to {END_YR}-{END_MO:02d})")

    for i, (start, end, yr, mo) in enumerate(month_list, 1):
        raw   = fetch_month(start, end)
        frame = parse_events(raw, year=yr)
        n     = frame["event_id"].nunique() if not frame.empty else 0
        print(f"  [{i:>3}/{total}] {start[:7]}  {n:>4} events")
        if not frame.empty:
            all_frames.append(frame)
        time.sleep(0.5)

    combined = pd.concat(all_frames, ignore_index=True)
    # An event observed near a month boundary may appear in two consecutive
    # monthly queries; keep only the first occurrence
    combined = combined.drop_duplicates(subset=["event_id", "obs_date", "lon", "lat"])
    combined.to_csv(CACHE_PATH, index=False)
    print(f"saved {len(combined):,} rows, {combined['event_id'].nunique():,} unique events")
    return combined


if __name__ == "__main__":
    df = fetch_10yr(cache=True)

    event_df = (
        df.groupby(["event_id", "year"])
        .agg(
            magnitude_max = ("magnitude_value", "max"),
            sources       = ("sources",         "first"),
        )
        .reset_index()
    )
    event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS", na=False).astype(int)

    summary = (
        event_df.groupby("year")
        .agg(
            n_events  = ("event_id",       "nunique"),
            has_mag   = ("magnitude_max",   lambda x: x.notna().sum()),
            median_mag= ("magnitude_max",   "median"),
            gdacs_pct = ("agency_gdacs",    "mean"),
        )
        .round(1)
    )
    summary["gdacs_pct"] = (summary["gdacs_pct"] * 100).round(1)
    print(summary.to_string())

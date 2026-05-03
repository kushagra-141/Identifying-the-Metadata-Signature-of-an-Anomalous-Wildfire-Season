"""
Patches missing 2021 months into eonet_10yr_combined.csv.
Uses exponential backoff to handle rate limiting.
"""

import requests
import pandas as pd
import time
from calendar import monthrange

BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"

MISSING_MONTHS = [
    ("2020-12-01", "2020-12-31", 2020, 12),
    ("2021-01-01", "2021-01-31", 2021,  1),
    ("2021-02-01", "2021-02-28", 2021,  2),
    ("2021-03-01", "2021-03-31", 2021,  3),
    ("2021-04-01", "2021-04-30", 2021,  4),
    ("2021-05-01", "2021-05-31", 2021,  5),
    ("2021-06-01", "2021-06-30", 2021,  6),
    ("2021-07-01", "2021-07-31", 2021,  7),
    ("2021-08-01", "2021-08-31", 2021,  8),
    ("2021-09-01", "2021-09-30", 2021,  9),
    ("2021-10-01", "2021-10-31", 2021, 10),
    ("2021-11-01", "2021-11-30", 2021, 11),
]


def fetch_with_retry(start, end, max_retries=5):
    params = {"category": "wildfires", "status": "all",
              "start": start, "end": end, "limit": 500}
    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = 10 * (2 ** attempt)
                print(f"    429 rate limit - waiting {wait}s ...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("events", [])
        except requests.RequestException as e:
            print(f"    Error: {e} - waiting 10s ...")
            time.sleep(10)
    return []


def parse_events(raw_events, year):
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
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    return df


# ── Load existing cache ────────────────────────────────────────────────────────
existing = pd.read_csv("eonet_10yr_combined.csv", parse_dates=["obs_date", "closed"])
existing_ids = set(existing["event_id"].unique())
print(f"Existing cache: {len(existing):,} rows, {len(existing_ids):,} unique events\n")

# ── Fetch missing months ───────────────────────────────────────────────────────
new_frames = []
for start, end, yr, mo in MISSING_MONTHS:
    print(f"  Fetching {start[:7]} ...", end=" ", flush=True)
    raw   = fetch_with_retry(start, end)
    frame = parse_events(raw, year=yr)
    n     = frame["event_id"].nunique() if not frame.empty else 0
    print(f"{n} events")
    if not frame.empty:
        new_frames.append(frame)
    time.sleep(3)   # conservative rate-limiting

# ── Merge and save ─────────────────────────────────────────────────────────────
if new_frames:
    new_df   = pd.concat(new_frames, ignore_index=True)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["event_id", "obs_date", "lon", "lat"])
    combined.to_csv("eonet_10yr_combined.csv", index=False)
    print(f"\nPatched cache: {len(combined):,} rows, "
          f"{combined['event_id'].nunique():,} unique events saved.")
else:
    print("\nNo new data fetched — cache unchanged.")

# ── Updated year summary ───────────────────────────────────────────────────────
df = pd.read_csv("eonet_10yr_combined.csv")
event_df = (df.groupby(["event_id","year"])
              .agg(magnitude_max=("magnitude_value","max"),
                   sources=("sources","first"))
              .reset_index())
event_df["agency_gdacs"] = event_df["sources"].str.contains("GDACS").astype(int)

print("\n=== Updated Events by Year ===")
summary = (event_df.groupby("year")
           .agg(n_events=("event_id","nunique"),
                has_mag=("magnitude_max", lambda x: x.notna().sum()),
                mean_mag=("magnitude_max","mean"),
                median_mag=("magnitude_max","median"),
                gdacs_pct=("agency_gdacs","mean"))
           .round(1))
summary["gdacs_pct"] = (summary["gdacs_pct"]*100).round(1)
print(summary.to_string())

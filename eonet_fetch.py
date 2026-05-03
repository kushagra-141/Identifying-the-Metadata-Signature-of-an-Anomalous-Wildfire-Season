"""
NASA EONET Wildfire Data Retrieval
Big Data Analytics Final Project
Compares 2024-2025 (baseline) vs 2025-2026 (anomalous) fire seasons.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timezone

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
CATEGORY = "wildfires"
PAGE_LIMIT = 500  # EONET max per request

# Fire season windows (Northern + Southern Hemisphere combined → full calendar year)
SEASONS = {
    "2024_2025": {
        "start": "2024-05-01",
        "end":   "2025-04-30",
        "label": 0,            # 0 = baseline
    },
    "2025_2026": {
        "start": "2025-05-01",
        "end":   "2026-04-30",
        "label": 1,            # 1 = anomalous / mega-fire season
    },
}

# ── Fetcher ────────────────────────────────────────────────────────────────────

def fetch_season(start: str, end: str, max_pages: int = 20) -> list[dict]:
    """
    Pages through EONET events for a given date window.
    Returns a flat list of raw event dicts.
    """
    all_events = []
    params = {
        "category": CATEGORY,
        "status":   "all",        # include both open and closed events
        "start":    start,
        "end":      end,
        "limit":    PAGE_LIMIT,
        "offset":   0,
    }

    for page in range(max_pages):
        params["offset"] = page * PAGE_LIMIT
        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [!] Request failed on page {page}: {e}")
            break

        payload = resp.json()
        events  = payload.get("events", [])
        if not events:
            break

        all_events.extend(events)
        print(f"  Page {page + 1}: fetched {len(events)} events "
              f"(total so far: {len(all_events)})")

        # EONET doesn't always paginate — stop if we got fewer than the limit
        if len(events) < PAGE_LIMIT:
            break

        time.sleep(0.3)  # polite rate-limiting

    return all_events


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_events(raw_events: list[dict], season_label: int) -> pd.DataFrame:
    """
    Flattens the nested EONET event structure into one row per geometry point.
    Each row represents a single positional observation of an event.

    Key columns extracted:
      event_id        — unique EONET identifier
      title           — fire name
      closed          — ISO timestamp if event ended, else None  (null = still active)
      is_active       — bool: True if closed is null
      source_count    — number of distinct reporting agencies (IRWIN, CALFIRE, etc.)
      sources         — pipe-separated list of agency IDs
      lon, lat        — geometry coordinates
      obs_date        — timestamp of this geometry observation
      magnitude_value — acres burned at this observation (may be NaN)
      magnitude_unit  — unit string (should be "acres")
      season_label    — 0=baseline 2024-2025, 1=anomalous 2025-2026
    """
    rows = []

    for evt in raw_events:
        closed_raw   = evt.get("closed")
        source_list  = [s["id"] for s in evt.get("sources", [])]

        base = {
            "event_id":     evt.get("id"),
            "title":        evt.get("title"),
            "closed":       closed_raw,
            "is_active":    closed_raw is None,
            "source_count": len(source_list),
            "sources":      "|".join(source_list),
            "season_label": season_label,
        }

        geometries = evt.get("geometry", [])
        if not geometries:
            # Keep the event even without geometry (magnitude will be NaN)
            rows.append({**base,
                         "lon": None, "lat": None,
                         "obs_date": None,
                         "magnitude_value": None,
                         "magnitude_unit": None})
            continue

        for geom in geometries:
            coords = geom.get("coordinates", [None, None])
            rows.append({
                **base,
                "lon":             coords[0],
                "lat":             coords[1],
                "obs_date":        geom.get("date"),
                "magnitude_value": geom.get("magnitudeValue"),
                "magnitude_unit":  geom.get("magnitudeUnit"),
            })

    df = pd.DataFrame(rows)

    # Type coercions
    if not df.empty:
        df["obs_date"] = pd.to_datetime(df["obs_date"], utc=True, errors="coerce")
        df["closed"]   = pd.to_datetime(df["closed"],   utc=True, errors="coerce")
        df["magnitude_value"] = pd.to_numeric(df["magnitude_value"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

        # Derived: event duration in days (NaN for still-active events)
        first_obs = df.groupby("event_id")["obs_date"].min().rename("first_obs")
        df = df.merge(first_obs, on="event_id", how="left")
        df["duration_days"] = (df["closed"] - df["first_obs"]).dt.total_seconds() / 86400

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def load_all_seasons(cache: bool = True) -> pd.DataFrame:
    """
    Fetches both seasons and returns a combined DataFrame.
    Pass cache=True to save/load from CSV so the API is only hit once.
    """
    cache_path = "eonet_wildfire_combined.csv"

    if cache:
        try:
            df = pd.read_csv(cache_path, parse_dates=["obs_date", "closed", "first_obs"])
            print(f"[cache] Loaded {len(df):,} rows from {cache_path}")
            return df
        except FileNotFoundError:
            pass

    frames = []
    for name, cfg in SEASONS.items():
        print(f"\n-- Fetching season: {name} ({cfg['start']} to {cfg['end']}) --")
        raw    = fetch_season(cfg["start"], cfg["end"])
        season_df = parse_events(raw, season_label=cfg["label"])
        season_df["season_name"] = name
        print(f"   -> {len(season_df):,} observation rows, "
              f"{season_df['event_id'].nunique()} unique events")
        frames.append(season_df)

    combined = pd.concat(frames, ignore_index=True)

    if cache:
        combined.to_csv(cache_path, index=False)
        print(f"\n[cache] Saved {len(combined):,} rows to {cache_path}")

    return combined


# ── Quick Sanity Check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_seasons(cache=True)

    print("\n=== Dataset Shape ===")
    print(df.shape)

    print("\n=== Column Dtypes ===")
    print(df.dtypes)

    print("\n=== Events per Season ===")
    summary = (df.groupby("season_name")
                 .agg(
                     unique_events   = ("event_id",        "nunique"),
                     obs_rows        = ("event_id",        "count"),
                     active_events   = ("is_active",       "sum"),
                     mean_magnitude  = ("magnitude_value", "mean"),
                     median_magnitude= ("magnitude_value", "median"),
                     max_magnitude   = ("magnitude_value", "max"),
                     mean_sources    = ("source_count",    "mean"),
                 )
                 .round(2))
    print(summary.to_string())

    print("\n=== Sample Rows ===")
    print(df[["event_id", "title", "season_name", "magnitude_value",
              "source_count", "is_active", "lon", "lat"]].head(10).to_string())

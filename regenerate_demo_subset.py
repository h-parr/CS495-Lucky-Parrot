"""Regenerate a high-quality demo subset with strict trip validity checks.

Rules enforced for each selected trip:
- Long trip distance (cumulative haversine distance)
- Non-trivial speed changes
- Weight data present and positive for most pings
- End location far from start location

If strict thresholds leave too few trips, thresholds are relaxed in controlled tiers
until enough valid trips are found.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SEGMENTED_PATH = PROJECT_ROOT / "data" / "segmented_trips.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "demo_trips_subset.csv"
TARGET_TRIPS = 20
PREFERRED_VIN = "Truck 18"
PREFERRED_MIN_TRIPS = 2


@dataclass(frozen=True)
class Thresholds:
    name: str
    min_pings: int
    min_total_km: float
    min_end_to_end_km: float
    min_speed_range: float
    min_speed_std: float
    min_weight_nonnull_ratio: float
    min_weight_positive_ratio: float
    min_valid_coord_ratio: float
    max_initial_idle_min: float
    min_moving_ratio: float
    min_path_efficiency: float


TIERS: list[Thresholds] = [
    Thresholds("strict", 90, 220.0, 140.0, 28.0, 9.0, 0.99, 0.99, 0.99, 20.0, 0.80, 0.12),
    Thresholds("balanced", 70, 150.0, 90.0, 22.0, 7.0, 0.97, 0.97, 0.97, 30.0, 0.70, 0.08),
    Thresholds("fallback", 50, 100.0, 60.0, 16.0, 5.0, 0.94, 0.94, 0.94, 45.0, 0.60, 0.05),
]


def haversine_km_np(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine in kilometers."""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def meets_thresholds(m: dict[str, float], t: Thresholds) -> bool:
    return (
        m["pings"] >= t.min_pings
        and m["total_km"] >= t.min_total_km
        and m["end_to_end_km"] >= t.min_end_to_end_km
        and m["speed_range"] >= t.min_speed_range
        and m["speed_std"] >= t.min_speed_std
        and m["weight_nonnull_ratio"] >= t.min_weight_nonnull_ratio
        and m["weight_positive_ratio"] >= t.min_weight_positive_ratio
        and m["coord_valid_ratio"] >= t.min_valid_coord_ratio
    )


def build_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trip_id"] = out["trip_id"].astype(str)
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
    out = out.sort_values(["trip_id", "Timestamp"]).reset_index(drop=True)

    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["Speed"] = pd.to_numeric(out["Speed"], errors="coerce")
    out["Weight_lbs"] = pd.to_numeric(out["Weight_lbs"], errors="coerce")
    if "VIN" not in out.columns:
        out["VIN"] = out["trip_id"].str.split("_").str[0]

    coord_valid = (
        out["Latitude"].notna()
        & out["Longitude"].notna()
        & out["Latitude"].between(-90, 90)
        & out["Longitude"].between(-180, 180)
    )
    out["coord_valid"] = coord_valid

    prev_trip = out["trip_id"].shift(1)
    same_trip_as_prev = out["trip_id"] == prev_trip
    prev_lat = out["Latitude"].shift(1)
    prev_lon = out["Longitude"].shift(1)
    prev_valid = coord_valid.shift(1, fill_value=False)
    has_prev_segment = same_trip_as_prev & coord_valid & prev_valid

    segment_km = np.zeros(len(out), dtype=float)
    idx = np.where(has_prev_segment.to_numpy())[0]
    if len(idx) > 0:
        segment_km[idx] = haversine_km_np(
            prev_lat.iloc[idx].to_numpy(),
            prev_lon.iloc[idx].to_numpy(),
            out["Latitude"].iloc[idx].to_numpy(),
            out["Longitude"].iloc[idx].to_numpy(),
        )
    out["segment_km"] = segment_km

    pings = out.groupby("trip_id").size().rename("pings")
    vin = out.groupby("trip_id")["VIN"].first().rename("vin")
    coord_valid_ratio = out.groupby("trip_id")["coord_valid"].mean().rename("coord_valid_ratio")
    total_km = out.groupby("trip_id")["segment_km"].sum().rename("total_km")

    speed_range = (out.groupby("trip_id")["Speed"].max() - out.groupby("trip_id")["Speed"].min()).fillna(0.0).rename("speed_range")
    speed_std = out.groupby("trip_id")["Speed"].std().fillna(0.0).rename("speed_std")

    weight_nonnull_ratio = out.groupby("trip_id")["Weight_lbs"].apply(lambda s: float(s.notna().mean())).rename("weight_nonnull_ratio")
    weight_positive_ratio = out.groupby("trip_id")["Weight_lbs"].apply(lambda s: float((s.fillna(0) > 0).mean())).rename("weight_positive_ratio")
    weight_range = (out.groupby("trip_id")["Weight_lbs"].max() - out.groupby("trip_id")["Weight_lbs"].min()).fillna(0.0).rename("weight_range")

    moving_speed_threshold = 8.0
    out["is_moving"] = out["Speed"].fillna(0.0) >= moving_speed_threshold
    moving_ratio = out.groupby("trip_id")["is_moving"].mean().rename("moving_ratio")
    first_ts = out.groupby("trip_id")["Timestamp"].min().rename("first_ts")
    first_moving_ts = out[out["is_moving"]].groupby("trip_id")["Timestamp"].min().rename("first_moving_ts")
    timing = pd.concat([first_ts, first_moving_ts], axis=1)
    initial_idle_minutes = (
        (timing["first_moving_ts"] - timing["first_ts"]).dt.total_seconds().div(60.0)
    ).fillna(1e9).rename("initial_idle_minutes")

    valid_coords = out[out["coord_valid"]].copy()
    first_coords = valid_coords.groupby("trip_id", as_index=False).first()[["trip_id", "Latitude", "Longitude"]]
    last_coords = valid_coords.groupby("trip_id", as_index=False).last()[["trip_id", "Latitude", "Longitude"]]
    endpoints = first_coords.merge(last_coords, on="trip_id", suffixes=("_start", "_end"))
    if len(endpoints) > 0:
        endpoints["end_to_end_km"] = haversine_km_np(
            endpoints["Latitude_start"].to_numpy(),
            endpoints["Longitude_start"].to_numpy(),
            endpoints["Latitude_end"].to_numpy(),
            endpoints["Longitude_end"].to_numpy(),
        )
        end_to_end_km = endpoints.set_index("trip_id")["end_to_end_km"]
    else:
        end_to_end_km = pd.Series(dtype=float, name="end_to_end_km")

    metrics = pd.concat(
        [
            pings,
            vin,
            coord_valid_ratio,
            total_km,
            speed_range,
            speed_std,
            weight_nonnull_ratio,
            weight_positive_ratio,
            weight_range,
            moving_ratio,
            initial_idle_minutes,
        ],
        axis=1,
    ).reset_index()
    metrics = metrics.merge(end_to_end_km.reset_index(), on="trip_id", how="left")
    metrics["end_to_end_km"] = metrics["end_to_end_km"].fillna(0.0)
    metrics["path_efficiency"] = np.where(
        metrics["total_km"] > 0,
        metrics["end_to_end_km"] / metrics["total_km"],
        0.0,
    )
    return metrics


def main() -> None:
    print("Loading segmented trips...")
    df = pd.read_csv(SEGMENTED_PATH, parse_dates=["Timestamp"])
    metrics = build_metrics_table(df)

    print(f"Scanned {len(metrics)} trips")
    chosen_tier: Thresholds | None = None
    eligible = pd.DataFrame()
    for tier in TIERS:
        mask = (
            (metrics["pings"] >= tier.min_pings)
            & (metrics["total_km"] >= tier.min_total_km)
            & (metrics["end_to_end_km"] >= tier.min_end_to_end_km)
            & (metrics["speed_range"] >= tier.min_speed_range)
            & (metrics["speed_std"] >= tier.min_speed_std)
            & (metrics["weight_nonnull_ratio"] >= tier.min_weight_nonnull_ratio)
            & (metrics["weight_positive_ratio"] >= tier.min_weight_positive_ratio)
            & (metrics["coord_valid_ratio"] >= tier.min_valid_coord_ratio)
            & (metrics["initial_idle_minutes"] <= tier.max_initial_idle_min)
            & (metrics["moving_ratio"] >= tier.min_moving_ratio)
            & (metrics["path_efficiency"] >= tier.min_path_efficiency)
        )
        candidates = metrics[mask].copy()
        print(f"Tier '{tier.name}' valid trips: {len(candidates)}")
        if len(candidates) >= TARGET_TRIPS:
            chosen_tier = tier
            eligible = candidates
            break

    if chosen_tier is None:
        # Use the loosest tier; if still short, keep all valid and stop.
        chosen_tier = TIERS[-1]
        mask = (
            (metrics["pings"] >= chosen_tier.min_pings)
            & (metrics["total_km"] >= chosen_tier.min_total_km)
            & (metrics["end_to_end_km"] >= chosen_tier.min_end_to_end_km)
            & (metrics["speed_range"] >= chosen_tier.min_speed_range)
            & (metrics["speed_std"] >= chosen_tier.min_speed_std)
            & (metrics["weight_nonnull_ratio"] >= chosen_tier.min_weight_nonnull_ratio)
            & (metrics["weight_positive_ratio"] >= chosen_tier.min_weight_positive_ratio)
            & (metrics["coord_valid_ratio"] >= chosen_tier.min_valid_coord_ratio)
            & (metrics["initial_idle_minutes"] <= chosen_tier.max_initial_idle_min)
            & (metrics["moving_ratio"] >= chosen_tier.min_moving_ratio)
            & (metrics["path_efficiency"] >= chosen_tier.min_path_efficiency)
        )
        eligible = metrics[mask].copy()

    if eligible.empty:
        raise RuntimeError("No trips satisfy even fallback validity thresholds. Check source data.")

    # Rank by long-distance quality and smooth motion changes.
    eligible = eligible.sort_values(
        by=[
            "moving_ratio",
            "path_efficiency",
            "end_to_end_km",
            "total_km",
            "speed_range",
            "speed_std",
            "weight_range",
            "pings",
        ],
        ascending=False,
    )

    selected = eligible.head(TARGET_TRIPS).copy()
    preferred_pool = eligible[eligible["vin"].astype(str).str.casefold() == PREFERRED_VIN.casefold()].copy()
    selected_preferred = selected[selected["vin"].astype(str).str.casefold() == PREFERRED_VIN.casefold()]
    need_preferred = max(0, PREFERRED_MIN_TRIPS - len(selected_preferred))
    if need_preferred > 0 and not preferred_pool.empty:
        preferred_to_add = preferred_pool[~preferred_pool["trip_id"].isin(selected["trip_id"])].head(need_preferred)
        if not preferred_to_add.empty:
            replace_count = len(preferred_to_add)
            selected = pd.concat([selected.iloc[:-replace_count], preferred_to_add], ignore_index=True)

    selected_ids = selected["trip_id"].tolist()
    if len(selected_ids) < TARGET_TRIPS:
        print(
            f"Warning: only {len(selected_ids)} valid trips after filtering "
            f"(target {TARGET_TRIPS})."
        )

    demo_df = df[df["trip_id"].astype(str).isin(selected_ids)].copy()
    demo_df = demo_df.drop(columns=["Device_Type", "Source"], errors="ignore")
    demo_df = demo_df.sort_values(["trip_id", "Timestamp"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nUsing tier: {chosen_tier.name}")
    print(f"Wrote {len(demo_df)} rows across {len(selected_ids)} trips to {OUTPUT_PATH}")
    print("\nSelected trips (verification):")
    for _, row in selected.iterrows():
        print(
            f"  {row['trip_id']}: {int(row['pings']):4d} pings | "
            f"total={row['total_km']:7.1f} km | end2end={row['end_to_end_km']:7.1f} km | "
            f"speed_range={row['speed_range']:5.1f} | speed_std={row['speed_std']:4.1f} | "
            f"moving_ratio={row['moving_ratio']:.2f} | idle_start_min={row['initial_idle_minutes']:.1f} | "
            f"eff={row['path_efficiency']:.3f} | weight_nonnull={row['weight_nonnull_ratio']:.2f}"
        )

    print("\nSelected subset minimums:")
    print(f"  min total_km      = {selected['total_km'].min():.1f}")
    print(f"  min end_to_end_km = {selected['end_to_end_km'].min():.1f}")
    print(f"  min speed_range   = {selected['speed_range'].min():.1f}")
    print(f"  min speed_std     = {selected['speed_std'].min():.1f}")
    print(f"  min moving_ratio  = {selected['moving_ratio'].min():.2f}")
    print(f"  max idle_start    = {selected['initial_idle_minutes'].max():.1f} min")
    print(f"  min path_eff      = {selected['path_efficiency'].min():.3f}")
    print(f"  min weight ratio  = {selected['weight_nonnull_ratio'].min():.2f}")


if __name__ == "__main__":
    main()

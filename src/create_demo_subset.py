"""Create a small, real-data trip subset CSV for the Streamlit demo.

Usage:
    python src/create_demo_subset.py --input <path-to-segmented_trips.csv>

This script selects trips with enough pings and writes a compact subset to:
    data/demo_trips_subset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("data/demo_trips_subset.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small trip subset for demo use")
    parser.add_argument("--input", required=True, help="Path to full segmented trips CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--trips", type=int, default=20, help="Number of trips to keep")
    parser.add_argument("--min-pings", type=int, default=80, help="Minimum rows per trip to be eligible")
    parser.add_argument(
        "--trip-column",
        default="trip_id",
        help="Trip identifier column name (default: trip_id)",
    )
    parser.add_argument(
        "--timestamp-column",
        default="Timestamp",
        help="Timestamp column name used for sorting (default: Timestamp)",
    )
    parser.add_argument(
        "--vin-column",
        default="VIN",
        help="Vehicle identifier column used to derive trips if trip_id is missing (default: VIN)",
    )
    parser.add_argument(
        "--gap-minutes",
        type=float,
        default=45.0,
        help="Time gap threshold to split trips when deriving trip_id (default: 45)",
    )
    return parser.parse_args()


def derive_trip_ids(
    df: pd.DataFrame,
    trip_column: str,
    vin_column: str,
    timestamp_column: str,
    gap_minutes: float,
) -> pd.DataFrame:
    """Create trip ids from VIN + timestamp gaps when trip ids are not present."""
    if vin_column not in df.columns:
        raise ValueError(f"Missing VIN column required to derive trips: {vin_column}")
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column required to derive trips: {timestamp_column}")

    out = df.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column], errors="coerce")
    out = out.sort_values([vin_column, timestamp_column]).reset_index(drop=True)

    # New trip starts at VIN boundary, missing timestamp, or large timestamp gap.
    vin_change = out[vin_column] != out[vin_column].shift(1)
    delta_min = out[timestamp_column].diff().dt.total_seconds().div(60)
    large_gap = delta_min.isna() | (delta_min > gap_minutes)
    new_trip = vin_change | large_gap

    out["_trip_seq"] = new_trip.groupby(out[vin_column]).cumsum().astype(int)
    out[trip_column] = out[vin_column].astype(str) + "_trip_" + out["_trip_seq"].astype(str)
    out = out.drop(columns=["_trip_seq"])
    return out


def build_subset(
    input_csv: Path,
    output_csv: Path,
    trips: int,
    min_pings: int,
    trip_column: str,
    timestamp_column: str,
    vin_column: str,
    gap_minutes: float,
) -> tuple[int, int]:
    df = pd.read_csv(input_csv)

    if trip_column not in df.columns:
        df = derive_trip_ids(
            df=df,
            trip_column=trip_column,
            vin_column=vin_column,
            timestamp_column=timestamp_column,
            gap_minutes=gap_minutes,
        )

    counts = df.groupby(trip_column).size().sort_values(ascending=False)
    selected_ids = counts[counts >= min_pings].head(trips).index

    if len(selected_ids) == 0:
        raise ValueError(
            "No eligible trips found. Try lowering --min-pings or check input columns."
        )

    subset = df[df[trip_column].isin(selected_ids)].copy()

    if timestamp_column in subset.columns:
        subset[timestamp_column] = pd.to_datetime(subset[timestamp_column], errors="coerce")
        subset = subset.sort_values([trip_column, timestamp_column]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_csv, index=False)

    return int(len(selected_ids)), int(len(subset))


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_csv = Path(args.output)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    selected_trip_count, row_count = build_subset(
        input_csv=input_csv,
        output_csv=output_csv,
        trips=args.trips,
        min_pings=args.min_pings,
        trip_column=args.trip_column,
        timestamp_column=args.timestamp_column,
        vin_column=args.vin_column,
        gap_minutes=args.gap_minutes,
    )

    print(f"Wrote {row_count:,} rows across {selected_trip_count} trips -> {output_csv}")


if __name__ == "__main__":
    main()

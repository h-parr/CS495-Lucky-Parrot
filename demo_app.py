"""Interactive Streamlit demo for truck ETA quantile predictions.

Run with:
    streamlit run demo_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "demo_trips_subset.csv"

PLOT_BG = "#0f1117"
PAPER_BG = "#080a0f"
COLOR_PRIMARY = "#58a6ff"
COLOR_ACCENT = "#34d399"
COLOR_WARNING = "#f59e0b"
COLOR_ROUTE = "#4b89dc"
COLOR_MUTED = "#4b5563"
COLOR_TEXT = "#f3f6ff"
MAX_ANIMATION_FRAMES = 140
ANIMATION_FPS = 60
MAP_ANIMATION_FPS = 40


st.set_page_config(
    page_title="Truck ETA Demo",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --app-bg: #080a0f;
            --panel-bg: #0f1117;
            --card-bg: #141924;
            --text-main: #f3f6ff;
            --text-soft: #c4cddf;
            --accent: #58a6ff;
        }
        .stApp {
            background: var(--app-bg);
            color: var(--text-main);
        }
        .block-container {
            max-width: 1380px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
            color: var(--text-main);
        }
        section[data-testid="stSidebar"] {
            background: var(--panel-bg);
            border-right: 1px solid rgba(88, 166, 255, 0.15);
        }
        section[data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }
        section[data-testid="stSidebar"] input {
            color: var(--text-main) !important;
            background-color: #1a1f2a !important;
        }
        section[data-testid="stSidebar"] input::placeholder {
            color: #8b95b0 !important;
        }
        section[data-testid="stSidebar"] div[class*="stSelectbox"] input {
            color: var(--text-main) !important;
            background-color: #1a1f2a !important;
        }
        section[data-testid="stSidebar"] div[class*="stTextInput"] input {
            color: var(--text-main) !important;
            background-color: #1a1f2a !important;
        }
        section[data-testid="stSidebar"] div[class*="stNumberInput"] input {
            color: var(--text-main) !important;
            background-color: #1a1f2a !important;
        }
        section[data-testid="stSidebar"] div[class*="stSlider"] {
            color: var(--text-main) !important;
        }
        [data-testid="stMetric"] {
            background: var(--card-bg);
            border: 1px solid rgba(88, 166, 255, 0.18);
            border-radius: 16px;
            padding: 12px 14px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"], [data-testid="stMetricValue"] {
            color: var(--text-main);
        }
        .hero-card {
            background: linear-gradient(135deg, #121722 0%, #0f141e 100%);
            border: 1px solid rgba(88, 166, 255, 0.16);
            border-radius: 24px;
            padding: 22px 26px;
            box-shadow: 0 16px 40px rgba(2, 6, 23, 0.45);
            margin-bottom: 1rem;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            color: #8ea2c7;
            margin-bottom: 0.35rem;
        }
        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0;
        }
        .hero-subtitle {
            color: var(--text-soft);
            font-size: 1.02rem;
            margin-top: 0.5rem;
        }
        .metric-card {
            background: var(--card-bg);
            padding: 18px 20px;
            border-radius: 18px;
            border: 1px solid rgba(88, 166, 255, 0.18);
            box-shadow: 0 12px 30px rgba(2, 6, 23, 0.45);
            color: var(--text-main);
        }
        .metric-card * {
            color: var(--text-main);
        }
        .section-label {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #90a5cc;
            margin-bottom: 0.45rem;
        }
        .meta-line {
            color: var(--text-soft);
            font-size: 0.97rem;
            margin-top: 0.4rem;
        }
        .distance-metric {
            background: var(--card-bg);
            padding: 16px 18px;
            border-radius: 14px;
            border: 1px solid rgba(88, 166, 255, 0.18);
            margin-bottom: 1rem;
            text-align: center;
        }
        .distance-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #34d399;
            margin: 0;
        }
        .distance-label {
            font-size: 0.85rem;
            color: #8b95b0;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> dict[str, object]:
    model_dir = PROJECT_ROOT / "models"
    return {
        "p10": joblib.load(model_dir / "lgb_q10.pkl"),
        "p50": joblib.load(model_dir / "lgb_q50.pkl"),
        "p90": joblib.load(model_dir / "lgb_q90.pkl"),
    }


def resolve_data_path(data_path: str | None = None) -> Path:
    candidate_paths: list[Path] = []
    if data_path:
        raw_path = Path(data_path)
        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append(Path.cwd() / raw_path)
            candidate_paths.append(PROJECT_ROOT / raw_path)
    else:
        candidate_paths.append(DEFAULT_DATA_PATH)

    existing_path = next((path for path in candidate_paths if path.exists()), None)
    if existing_path is None:
        checked = "\n".join(f"- {path}" for path in candidate_paths)
        raise FileNotFoundError(
            "Could not find trip data CSV. Checked:\n"
            f"{checked}\n"
            "Use the sidebar to upload a CSV or provide a valid local path."
        )
    return existing_path


@st.cache_data
def load_sample_data(
    data_path: str | None = None,
    uploaded_bytes=None,
    file_mtime_ns: int | None = None,
) -> pd.DataFrame:
    if uploaded_bytes is not None:
        df = pd.read_csv(uploaded_bytes)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df

    # file_mtime_ns is intentionally part of the cache key to invalidate cache when CSV changes.
    _ = file_mtime_ns
    existing_path = resolve_data_path(data_path)

    df = pd.read_csv(existing_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import asin, cos, radians, sin, sqrt

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def prettify_trip_labels(df: pd.DataFrame) -> dict[str, str]:
    trip_meta = (
        df[["VIN", "trip_id", "Timestamp"]]
        .dropna(subset=["trip_id"])
        .copy()
    )
    trip_meta["trip_id"] = trip_meta["trip_id"].astype(str)
    trip_meta = (
        trip_meta.groupby(["VIN", "trip_id"], as_index=False)["Timestamp"]
        .min()
        .sort_values(["VIN", "Timestamp", "trip_id"])
    )
    trip_meta["trip_number"] = trip_meta.groupby("VIN").cumcount() + 1
    trip_meta["trip_count"] = trip_meta.groupby("VIN")["trip_id"].transform("count")
    trip_meta["label"] = np.where(
        trip_meta["trip_count"] == 1,
        trip_meta["VIN"].astype(str),
        trip_meta["VIN"].astype(str) + "_" + trip_meta["trip_number"].astype(str),
    )
    return dict(zip(trip_meta["trip_id"], trip_meta["label"]))


def detect_stops(trip_df: pd.DataFrame, speed_threshold: float = 2.0, min_duration: float = 60) -> list[dict]:
    stops: list[dict] = []
    ordered = trip_df.sort_values("Timestamp").reset_index(drop=True)
    in_stop = False
    stop_start = None

    for idx, row in ordered.iterrows():
        speed = row["Speed"] if pd.notna(row["Speed"]) else 0.0
        if speed < speed_threshold:
            if not in_stop:
                in_stop = True
                stop_start = idx
        elif in_stop:
            stop_end = idx - 1
            if stop_start is not None and stop_end >= stop_start:
                duration_min = (
                    ordered.iloc[stop_end]["Timestamp"] - ordered.iloc[stop_start]["Timestamp"]
                ).total_seconds() / 60
                if duration_min >= min_duration:
                    stops.append(
                        {
                            "start_idx": stop_start,
                            "end_idx": stop_end,
                            "start_time": ordered.iloc[stop_start]["Timestamp"],
                            "duration_min": duration_min,
                            "lat": ordered.iloc[stop_start]["Latitude"],
                            "lon": ordered.iloc[stop_start]["Longitude"],
                        }
                    )
            in_stop = False

    return stops


def add_trip_summary_columns(trip_df: pd.DataFrame) -> pd.DataFrame:
    out = trip_df.copy().sort_values("Timestamp").reset_index(drop=True)
    step_km = [0.0]
    for idx in range(1, len(out)):
        step_km.append(
            haversine_distance(
                out.iloc[idx - 1]["Latitude"],
                out.iloc[idx - 1]["Longitude"],
                out.iloc[idx]["Latitude"],
                out.iloc[idx]["Longitude"],
            )
        )
    out["step_distance_km"] = step_km
    out["cumulative_distance_km"] = out["step_distance_km"].cumsum()
    total_distance_km = float(out["cumulative_distance_km"].iloc[-1])
    out["distance_remaining_km"] = total_distance_km - out["cumulative_distance_km"]
    trip_end = out["Timestamp"].max()
    out["actual_eta_min"] = (trip_end - out["Timestamp"]).dt.total_seconds() / 60.0
    out["elapsed_min"] = (out["Timestamp"] - out["Timestamp"].min()).dt.total_seconds() / 60.0
    return out


@st.cache_data(show_spinner=False)
def prepare_trip_data(trip_data: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], str | None]:
    """Build per-trip features/predictions once per trip selection for faster reruns."""
    trip_data = trip_data.copy().sort_values("Timestamp").reset_index(drop=True)
    if len(trip_data) == 0:
        return trip_data, [], "Selected trip not found."

    trip_data = add_trip_summary_columns(trip_data)
    stops = detect_stops(trip_data)

    try:
        trip_with_features, feature_cols = build_features_for_trip(trip_data)
        trip_with_features = add_trip_summary_columns(trip_with_features)
        trip_with_features = enrich_predictions(trip_with_features, feature_cols, load_models())
    except Exception as exc:
        trip_with_features = trip_data.copy()
        trip_with_features["pred_p50_min"] = trip_with_features["actual_eta_min"]
        trip_with_features["pred_p10_min"] = np.clip(trip_with_features["actual_eta_min"] - 5, 0, None)
        trip_with_features["pred_p90_min"] = trip_with_features["actual_eta_min"] + 5
        return trip_with_features, stops, f"Feature building had issues: {exc}. Falling back to actual ETA curves."

    return trip_with_features, stops, None


def build_features_for_trip(trip_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from src.features import FEATURE_COLS, build_features

    trip_with_features = build_features(trip_df.copy())
    return trip_with_features, FEATURE_COLS


def enrich_predictions(trip_features: pd.DataFrame, feature_cols: list[str], models: dict[str, object]) -> pd.DataFrame:
    out = trip_features.copy()
    feature_frame = out[feature_cols].ffill().fillna(0.0)
    pred_p10 = models["p10"].predict(feature_frame)
    pred_p50 = models["p50"].predict(feature_frame)
    pred_p90 = models["p90"].predict(feature_frame)
    pred_stack = np.vstack([pred_p10, pred_p50, pred_p90]).T
    pred_stack.sort(axis=1)
    out["pred_p10_min"] = np.clip(pred_stack[:, 0], 0, None)
    out["pred_p50_min"] = np.clip(pred_stack[:, 1], 0, None)
    out["pred_p90_min"] = np.clip(pred_stack[:, 2], 0, None)
    return out


def build_eta_figure(
    trip_df: pd.DataFrame,
    playback_step: int,
    target_duration_s: int,
) -> go.Figure:
    """Build ETA quantile band figure with per-ping smooth animation."""
    frame_indices = list(range(len(trip_df)))

    max_eta = float(np.nanmax(np.nan_to_num(trip_df[["actual_eta_min", "pred_p90_min"]].to_numpy(), nan=0.0)))
    initial_idx = frame_indices[0]

    fig = go.Figure()
    
    # P90-P10 band
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["pred_p90_min"],
            mode="lines",
            line=dict(color="rgba(66, 99, 235, 0)", width=0),
            hoverinfo="skip",
            showlegend=False,
            name="P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["pred_p10_min"],
            mode="lines",
            line=dict(color="rgba(66, 99, 235, 0)", width=0),
            fill="tonexty",
            fillcolor="rgba(66, 99, 235, 0.18)",
            name="P10-P90 interval",
            hoverinfo="skip",
        )
    )
    
    # P50 median
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["pred_p50_min"],
            mode="lines",
            line=dict(color=COLOR_ACCENT, width=4),
            name="Median prediction (P50)",
        )
    )
    
    # Actual ETA
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["actual_eta_min"],
            mode="lines",
            line=dict(color=COLOR_PRIMARY, width=2, dash="dash"),
            name="Actual remaining ETA",
        )
    )
    
    # Current position marker
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[trip_df.iloc[initial_idx]["pred_p50_min"]],
            mode="markers",
            marker=dict(size=11, color="#111827", line=dict(color="white", width=1.5)),
            name="Current prediction",
            hovertemplate="Current P50<br>%{y:.1f} min<extra></extra>",
        )
    )
    
    # Vertical line
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"], trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[0, max_eta],
            mode="lines",
            line=dict(color="#1f2937", width=1.5, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Build smooth animation frames (only update moving marker/line)
    frames: list[go.Frame] = []
    for idx in frame_indices:
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(x=[trip_df.iloc[idx]["elapsed_min"]], y=[trip_df.iloc[idx]["pred_p50_min"]]),
                    go.Scatter(x=[trip_df.iloc[idx]["elapsed_min"], trip_df.iloc[idx]["elapsed_min"]], y=[0, max_eta]),
                ],
                traces=[4, 5],
            )
        )
    
    eta_target_duration_s = 20
    frame_duration_ms = max(1, int((eta_target_duration_s * 1000) / max(len(frame_indices), 1)))
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text="Remaining ETA with Prediction Interval",
            font=dict(size=20, color=COLOR_TEXT),
            x=0.5,
            xanchor="center",
        ),
        height=500,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(color=COLOR_TEXT, size=12),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)", bordercolor="#4b5563", borderwidth=1),
        hovermode="x unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration_ms, "redraw": False},
                                "transition": {"duration": frame_duration_ms, "easing": "linear"},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    fig.update_xaxes(title_text="Elapsed minutes", gridcolor="#2b3548", zeroline=False)
    fig.update_yaxes(title_text="Remaining ETA (min)", gridcolor="#2b3548", zeroline=False, range=[0, max(1.0, max_eta * 1.08)])
    return fig


def compute_map_bounds(trip_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute robust map bounds around trip points."""
    lons = trip_df["Longitude"].dropna().astype(float)
    lats = trip_df["Latitude"].dropna().astype(float)

    if len(lons) < 2 or len(lats) < 2:
        return -125.0, -66.0, 24.0, 50.0

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    lon_span = max(lon_max - lon_min, 0.05)
    lat_span = max(lat_max - lat_min, 0.05)
    lon_pad = max(lon_span * 0.25, 0.6)
    lat_pad = max(lat_span * 0.25, 0.5)

    return (
        max(lon_min - lon_pad, -180.0),
        min(lon_max + lon_pad, 180.0),
        max(lat_min - lat_pad, -90.0),
        min(lat_max + lat_pad, 90.0),
    )


def compute_start_centered_map_bounds(trip_df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Center map around the starting ping, zoomed out enough to show context."""
    lon_min, lon_max, lat_min, lat_max = compute_map_bounds(trip_df)
    start_lon = float(trip_df.iloc[0]["Longitude"])
    start_lat = float(trip_df.iloc[0]["Latitude"])

    span_lon = max(lon_max - lon_min, 1.0)
    span_lat = max(lat_max - lat_min, 1.0)

    # Keep start point centered while zooming out slightly.
    half_lon = min(max(span_lon * 0.75, 1.5), 25.0)
    half_lat = min(max(span_lat * 0.75, 1.2), 18.0)

    # Keep the same vertical zoom while widening horizontal view so the map
    # does not render as a narrow portrait strip.
    target_width_to_height = 1.55
    lat_for_scale = max(0.2, np.cos(np.radians(start_lat)))
    required_half_lon = (half_lat * target_width_to_height) / lat_for_scale
    half_lon = max(half_lon, required_half_lon)

    return (
        max(start_lon - half_lon, -180.0),
        min(start_lon + half_lon, 180.0),
        max(start_lat - half_lat, -90.0),
        min(start_lat + half_lat, 90.0),
    )


def _is_mostly_us_trip(trip_df: pd.DataFrame) -> bool:
    """Detect whether coordinates are within continental US bounds."""
    valid = trip_df[["Latitude", "Longitude"]].dropna()
    if valid.empty:
        return True

    in_us = (
        valid["Latitude"].between(24.0, 50.0)
        & valid["Longitude"].between(-125.0, -66.0)
    )
    return bool(in_us.mean() >= 0.8)


def get_major_cities_in_bounds(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> list[dict]:
    """Return major US cities within the given bounds."""
    major_cities = [
        {"name": "New York", "lon": -74.0, "lat": 40.7},
        {"name": "Los Angeles", "lon": -118.2, "lat": 34.0},
        {"name": "Chicago", "lon": -87.6, "lat": 41.9},
        {"name": "Dallas", "lon": -96.8, "lat": 32.8},
        {"name": "Houston", "lon": -95.4, "lat": 29.8},
        {"name": "Phoenix", "lon": -112.1, "lat": 33.4},
        {"name": "Philadelphia", "lon": -75.2, "lat": 39.9},
        {"name": "San Antonio", "lon": -98.5, "lat": 29.4},
        {"name": "San Diego", "lon": -117.2, "lat": 32.7},
        {"name": "San Francisco", "lon": -122.4, "lat": 37.8},
        {"name": "Denver", "lon": -104.9, "lat": 39.7},
        {"name": "Boston", "lon": -71.1, "lat": 42.4},
        {"name": "Atlanta", "lon": -84.4, "lat": 33.7},
        {"name": "Miami", "lon": -80.2, "lat": 25.8},
        {"name": "Seattle", "lon": -122.3, "lat": 47.6},
        {"name": "Detroit", "lon": -83.0, "lat": 42.3},
        {"name": "Minneapolis", "lon": -93.3, "lat": 44.9},
        {"name": "St. Louis", "lon": -90.2, "lat": 38.6},
        {"name": "Nashville", "lon": -86.8, "lat": 36.2},
        {"name": "Memphis", "lon": -90.0, "lat": 35.1},
    ]
    return [c for c in major_cities if lon_min <= c["lon"] <= lon_max and lat_min <= c["lat"] <= lat_max]


def build_map_figure(
    trip_df: pd.DataFrame,
    playback_step: int,
    target_duration_s: int,
) -> go.Figure:
    """Build trip path map figure with per-ping smooth animation."""
    # Validate and sanitize coordinates.
    cleaned = trip_df.copy()
    cleaned = cleaned[
        cleaned[["Latitude", "Longitude"]].notna().all(axis=1)
        & cleaned["Latitude"].between(-90, 90)
        & cleaned["Longitude"].between(-180, 180)
    ].reset_index(drop=True)

    valid_coords = len(cleaned)
    if valid_coords < 2:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient valid coordinates for map visualization", showarrow=False)
        fig.update_layout(height=600, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG)
        return fig

    trip_df = cleaned
    
    frame_indices = list(range(len(trip_df)))
    
    lon_min, lon_max, lat_min, lat_max = compute_start_centered_map_bounds(trip_df)
    major_cities = get_major_cities_in_bounds(lon_min, lon_max, lat_min, lat_max)
    
    fig = go.Figure()
    
    # Full route (gray background)
    fig.add_trace(
        go.Scattergeo(
            lon=trip_df["Longitude"],
            lat=trip_df["Latitude"],
            mode="lines",
            line=dict(color="#2e3f58", width=3),
            name="Full route",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Completed route (animated)
    initial_idx = frame_indices[0]
    fig.add_trace(
        go.Scattergeo(
            lon=trip_df["Longitude"][: initial_idx + 1],
            lat=trip_df["Latitude"][: initial_idx + 1],
            mode="lines",
            line=dict(color=COLOR_ROUTE, width=5),
            name="Traveled route",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Start marker
    fig.add_trace(
        go.Scattergeo(
            lon=[trip_df.iloc[0]["Longitude"]],
            lat=[trip_df.iloc[0]["Latitude"]],
            mode="markers",
            marker=dict(size=12, color=COLOR_ACCENT, line=dict(color="white", width=1.5)),
            name="Start",
            hovertemplate="Start<extra></extra>",
            showlegend=False,
        )
    )
    
    # End marker
    fig.add_trace(
        go.Scattergeo(
            lon=[trip_df.iloc[-1]["Longitude"]],
            lat=[trip_df.iloc[-1]["Latitude"]],
            mode="markers",
            marker=dict(size=12, color="#111827", symbol="diamond", line=dict(color="white", width=1.5)),
            name="End",
            hovertemplate="End<extra></extra>",
            showlegend=False,
        )
    )
    
    # Current position (animated)
    fig.add_trace(
        go.Scattergeo(
            lon=[trip_df.iloc[initial_idx]["Longitude"]],
            lat=[trip_df.iloc[initial_idx]["Latitude"]],
            mode="markers",
            marker=dict(size=18, color="#d33d3d", line=dict(color="white", width=2.0)),
            name="Current position",
            hovertemplate="Current<extra></extra>",
            showlegend=False,
        )
    )
    
    # City labels
    if major_cities:
        city_lons = [c["lon"] for c in major_cities]
        city_lats = [c["lat"] for c in major_cities]
        city_names = [c["name"] for c in major_cities]
        fig.add_trace(
            go.Scattergeo(
                lon=city_lons,
                lat=city_lats,
                mode="text",
                text=city_names,
                textposition="top center",
                textfont=dict(size=10, color="#8b95b0"),
                hoverinfo="text",
                hovertext=city_names,
                showlegend=False,
            )
        )
    
    # Build animation frames for every ping.
    frames: list[go.Frame] = []
    for idx in frame_indices:
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scattergeo(lon=trip_df["Longitude"][: idx + 1], lat=trip_df["Latitude"][: idx + 1]),
                    go.Scattergeo(lon=[trip_df.iloc[idx]["Longitude"]], lat=[trip_df.iloc[idx]["Latitude"]]),
                ],
                traces=[1, 4],
            )
        )
    
    frame_duration_ms = max(1, int(1000 / MAP_ANIMATION_FPS))
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text="Trip Path Map",
            font=dict(size=20, color=COLOR_TEXT),
            x=0.5,
            xanchor="center",
        ),
        height=650,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=20, r=20, t=80, b=60),
        font=dict(color=COLOR_TEXT, size=11),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.12,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    # Keep a stable world camera to avoid map rendering regressions,
    # and enable subunits so US state borders are visible when available.
    fig.update_geos(
        scope="world",
        projection_type="mercator",
        showland=True,
        landcolor="#111827",
        showcountries=True,
        countrycolor="#2b3548",
        showsubunits=True,
        subunitcolor="#3f5a8a",
        subunitwidth=1.1,
        showcoastlines=True,
        coastlinecolor="#2b3548",
        lonaxis_range=[lon_min, lon_max],
        lataxis_range=[lat_min, lat_max],
        bgcolor=PLOT_BG,
    )
    fig.update_annotations(font=dict(color=COLOR_TEXT))
    return fig


def build_speed_weight_figure(
    trip_df: pd.DataFrame,
    playback_step: int,
    target_duration_s: int,
) -> go.Figure:
    """Build combined Speed and Weight figure with smooth animation."""
    frame_indices = list(range(len(trip_df)))
    
    weight_series = trip_df["Weight_lbs"].ffill().fillna(0.0)
    max_speed = float(np.nanmax(np.nan_to_num(trip_df["Speed"].to_numpy(), nan=0.0)))
    max_weight = float(np.nanmax(np.nan_to_num(weight_series.to_numpy(), nan=0.0)))
    initial_idx = frame_indices[0]
    
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Speed over trip", "Weight over trip"),
        horizontal_spacing=0.12,
    )
    
    # Speed - background
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["Speed"],
            mode="lines",
            line=dict(color=COLOR_MUTED, width=2),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    
    # Speed - filled
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"][: initial_idx + 1],
            y=trip_df["Speed"][: initial_idx + 1],
            mode="lines",
            line=dict(color=COLOR_PRIMARY, width=3),
            fill="tozeroy",
            fillcolor="rgba(88, 166, 255, 0.16)",
            name="Speed",
        ),
        row=1,
        col=1,
    )
    
    # Speed - marker
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[trip_df.iloc[initial_idx]["Speed"]],
            mode="markers",
            marker=dict(size=10, color=COLOR_PRIMARY),
            hovertemplate="Speed<br>%{y:.1f} mph<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    
    # Weight - background
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=weight_series,
            mode="lines",
            line=dict(color=COLOR_MUTED, width=2),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    
    # Weight - filled
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"][: initial_idx + 1],
            y=weight_series[: initial_idx + 1],
            mode="lines",
            line=dict(color=COLOR_WARNING, width=3),
            name="Weight",
        ),
        row=1,
        col=2,
    )
    
    # Weight - marker
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[weight_series.iloc[initial_idx]],
            mode="markers",
            marker=dict(size=10, color=COLOR_WARNING),
            hovertemplate="Weight<br>%{y:,.0f} lbs<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    
    # Build smooth animation frames (update only changing traces for better performance)
    frames: list[go.Frame] = []
    for idx in frame_indices:
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(x=trip_df["elapsed_min"][: idx + 1], y=trip_df["Speed"][: idx + 1]),
                    go.Scatter(x=[trip_df.iloc[idx]["elapsed_min"]], y=[trip_df.iloc[idx]["Speed"]]),
                    go.Scatter(x=trip_df["elapsed_min"][: idx + 1], y=weight_series[: idx + 1]),
                    go.Scatter(x=[trip_df.iloc[idx]["elapsed_min"]], y=[weight_series.iloc[idx]]),
                ],
                traces=[1, 2, 4, 5],
            )
        )
    
    frame_duration_ms = max(1, int(1000 / ANIMATION_FPS))
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text="Speed and Weight Tracking",
            font=dict(size=20, color=COLOR_TEXT),
            x=0.5,
            xanchor="center",
        ),
        height=450,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(color=COLOR_TEXT, size=12),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration_ms, "redraw": False},
                                "transition": {"duration": frame_duration_ms, "easing": "linear"},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    fig.update_xaxes(title_text="Elapsed minutes", gridcolor="#2b3548", zeroline=False)
    fig.update_yaxes(title_text="Speed (mph)", gridcolor="#2b3548", zeroline=False, range=[0, max(10.0, max_speed * 1.1)], row=1, col=1)
    fig.update_yaxes(title_text="Weight (lbs)", gridcolor="#2b3548", zeroline=False, range=[0, max(1000.0, max_weight * 1.08)], row=1, col=2)
    fig.update_annotations(font=dict(color=COLOR_TEXT))
    return fig


def build_distance_figure(
    trip_df: pd.DataFrame,
    playback_step: int,
    target_duration_s: int,
) -> go.Figure:
    """Build animated distance traveled figure so progress visibly changes."""
    frame_indices = list(range(len(trip_df)))

    initial_idx = frame_indices[0]
    total_km = float(trip_df["cumulative_distance_km"].iloc[-1])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["cumulative_distance_km"],
            mode="lines",
            line=dict(color=COLOR_MUTED, width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"][: initial_idx + 1],
            y=trip_df["cumulative_distance_km"][: initial_idx + 1],
            mode="lines",
            line=dict(color=COLOR_ACCENT, width=4),
            fill="tozeroy",
            fillcolor="rgba(52, 211, 153, 0.15)",
            name="Distance traveled",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[trip_df.iloc[initial_idx]["cumulative_distance_km"]],
            mode="markers+text",
            marker=dict(size=12, color=COLOR_ACCENT, line=dict(color="white", width=1.5)),
            text=[f"{trip_df.iloc[initial_idx]['cumulative_distance_km']:.1f} km"],
            textposition="top center",
            textfont=dict(color=COLOR_TEXT, size=12),
            name="Current distance",
            hovertemplate="Distance<br>%{y:.2f} km<extra></extra>",
            showlegend=False,
        )
    )

    frames: list[go.Frame] = []
    for idx in frame_indices:
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(
                        x=trip_df["elapsed_min"][: idx + 1],
                        y=trip_df["cumulative_distance_km"][: idx + 1],
                    ),
                    go.Scatter(
                        x=[trip_df.iloc[idx]["elapsed_min"]],
                        y=[trip_df.iloc[idx]["cumulative_distance_km"]],
                        text=[f"{trip_df.iloc[idx]['cumulative_distance_km']:.1f} km"],
                    ),
                ],
                traces=[1, 2],
            )
        )

    frame_duration_ms = max(1, int(1000 / ANIMATION_FPS))
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text="Distance Traveled Over Trip",
            font=dict(size=20, color=COLOR_TEXT),
            x=0.5,
            xanchor="center",
        ),
        height=340,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(color=COLOR_TEXT, size=12),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration_ms, "redraw": False},
                                "transition": {"duration": frame_duration_ms, "easing": "linear"},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        annotations=[
            dict(
                text=f"Total distance: {total_km:.1f} km",
                xref="paper",
                yref="paper",
                x=0.99,
                y=1.14,
                showarrow=False,
                xanchor="right",
                font=dict(color="#8b95b0", size=11),
            )
        ],
    )
    fig.update_xaxes(title_text="Elapsed minutes", gridcolor="#2b3548", zeroline=False)
    fig.update_yaxes(title_text="Distance (km)", gridcolor="#2b3548", zeroline=False, range=[0, max(1.0, total_km * 1.08)])
    return fig


def build_metrics_figure(
    trip_df: pd.DataFrame,
    playback_step: int,
    target_duration_s: int,
) -> go.Figure:
    """Build performance metrics figure with smooth animation - updates metrics every few pings."""
    # Use adaptive step to keep animation smooth (max ~140 frames)
    adaptive_step = max(1, int(np.ceil(len(trip_df) / MAX_ANIMATION_FRAMES)))
    frame_indices = list(range(0, len(trip_df), adaptive_step))
    if frame_indices[-1] != len(trip_df) - 1:
        frame_indices.append(len(trip_df) - 1)
    
    initial_idx = frame_indices[0]
    
    fig = go.Figure()
    
    # Absolute error over time
    fig.add_trace(
        go.Scatter(
            x=trip_df["elapsed_min"],
            y=trip_df["abs_error_min"],
            mode="lines+markers",
            line=dict(color="#f97316", width=3),
            marker=dict(size=5, opacity=0.8),
            name="Absolute Error",
        )
    )
    
    # Current error marker
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[trip_df.iloc[initial_idx]["abs_error_min"]],
            mode="markers",
            marker=dict(size=14, color="#d33d3d", line=dict(color="white", width=2)),
            name="Current error",
            hovertemplate="Error: %{y:.2f} min<extra></extra>",
        )
    )
    
    # Vertical line
    fig.add_trace(
        go.Scatter(
            x=[trip_df.iloc[initial_idx]["elapsed_min"], trip_df.iloc[initial_idx]["elapsed_min"]],
            y=[0, trip_df["abs_error_min"].max()],
            mode="lines",
            line=dict(color="#1f2937", width=1.5, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    
    # Build smooth animation frames - update every adaptive_step pings
    frames: list[go.Frame] = []
    for idx in frame_indices:
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(x=trip_df["elapsed_min"], y=trip_df["abs_error_min"]),
                    go.Scatter(x=[trip_df.iloc[idx]["elapsed_min"]], y=[trip_df.iloc[idx]["abs_error_min"]]),
                    go.Scatter(x=[trip_df.iloc[idx]["elapsed_min"], trip_df.iloc[idx]["elapsed_min"]], y=[0, trip_df["abs_error_min"].max()]),
                ],
                traces=[0, 1, 2],
            )
        )
    
    frame_duration_ms = max(30, int((target_duration_s * 1000) / max(len(frame_indices), 1)))
    fig.frames = frames
    fig.update_layout(
        title=dict(
            text="Absolute Prediction Error Over Time",
            font=dict(size=20, color=COLOR_TEXT),
            x=0.5,
            xanchor="center",
        ),
        height=400,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        margin=dict(l=60, r=20, t=80, b=60),
        font=dict(color=COLOR_TEXT, size=12),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)", bordercolor="#4b5563", borderwidth=1),
        hovermode="x unified",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.01,
                "y": 1.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
    )
    fig.update_xaxes(title_text="Elapsed minutes", gridcolor="#2b3548", zeroline=False)
    fig.update_yaxes(title_text="Absolute Error (min)", gridcolor="#2b3548", zeroline=False)
    return fig


st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">Real-time Truck ETA Visualization</div>
        <h1 class="hero-title">ETA Prediction Intervals</h1>
        <div class="hero-subtitle">Replay a real trip, follow the truck across a clean USA map, and inspect how the prediction interval evolves without the interface flashing on each ping.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    models = load_models()
except Exception as exc:
    st.error(f"Error loading models: {exc}")
    st.stop()

with st.sidebar:
    st.header("Demo Controls")
    data_mode = st.radio("Choose how to load trip data:", ["Use local CSV path", "Upload CSV"], index=0)
    local_data_path = st.text_input(
        "CSV path",
        value=str(DEFAULT_DATA_PATH),
        help="Absolute path or path relative to the project root",
        disabled=data_mode != "Use local CSV path",
    )
    uploaded_file = st.file_uploader(
        "Upload segmented trips CSV",
        type=["csv"],
        disabled=data_mode != "Upload CSV",
    )

try:
    if data_mode == "Upload CSV":
        if uploaded_file is None:
            st.info("Upload a CSV in the sidebar to start the demo.")
            st.stop()
        all_data = load_sample_data(uploaded_bytes=uploaded_file)
    else:
        resolved_local_path = resolve_data_path(local_data_path)
        all_data = load_sample_data(
            data_path=str(resolved_local_path),
            file_mtime_ns=resolved_local_path.stat().st_mtime_ns,
        )
except Exception as exc:
    st.error(f"Error loading models or data: {exc}")
    st.caption("Tip: generate a committed demo subset with: python src/create_demo_subset.py --input <path-to-segmented_trips.csv>")
    st.caption("Expected columns include: VIN, Timestamp, Latitude, Longitude, Speed, Weight_lbs, trip_id")
    st.stop()

with st.sidebar:
    trip_label_map = prettify_trip_labels(all_data)
    all_trips = sorted(
        all_data["trip_id"].dropna().astype(str).unique(),
        key=lambda trip_id: trip_label_map.get(trip_id, trip_id),
    )
    if not all_trips:
        st.error("No trip_id values found in loaded data.")
        st.stop()

    selected_trip = st.selectbox(
        "Trip",
        all_trips,
        format_func=lambda trip_id: trip_label_map.get(trip_id, trip_id),
        help="Choose a trip to visualize",
    )
    playback_step = st.slider(
        "Animation frame step",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher values skip more pings to keep playback near 20-30 seconds",
    )
    animation_duration_s = st.slider(
        "Trip playback duration (seconds)",
        min_value=20,
        max_value=30,
        value=24,
        help="Target time for the full trip animation",
    )
    st.divider()
    st.subheader("Model Context")
    st.markdown(
        """
        **Models used:**
        - **P10**: 10th percentile (optimistic)
        - **P50**: Median (best estimate)
        - **P90**: 90th percentile (conservative)

        **Features:**
        - Elapsed time, rolling speed stats
        - Distance remaining, time encoding
        - Weight flag, trip progress
        - Load state indicators
        """
    )

trip_data = all_data[all_data["trip_id"].astype(str) == selected_trip].copy()
trip_data = trip_data.sort_values("Timestamp").reset_index(drop=True)
if len(trip_data) == 0:
    st.error("Selected trip not found.")
    st.stop()

trip_data = add_trip_summary_columns(trip_data)
trip_start_time = trip_data.iloc[0]["Timestamp"]
trip_end_time = trip_data.iloc[-1]["Timestamp"]
trip_duration_hours = (trip_end_time - trip_start_time).total_seconds() / 3600
if trip_duration_hours <= 0:
    trip_duration_hours = 1e-9
total_distance_km = float(trip_data["cumulative_distance_km"].iloc[-1])
trip_with_features, stops, prep_warning = prepare_trip_data(trip_data)
if prep_warning:
    st.warning(prep_warning)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
with summary_col1:
    st.metric("Trip", trip_label_map.get(selected_trip, selected_trip), f"{len(trip_data)} pings")
with summary_col2:
    st.metric("Trip window", f"{trip_duration_hours:.1f} h", trip_start_time.strftime("Start %H:%M"))
with summary_col3:
    st.metric("Distance", f"{total_distance_km:.1f} km", f"{(total_distance_km / trip_duration_hours):.1f} km/h avg")
with summary_col4:
    st.metric("Stops", len(stops), f"{sum(s['duration_min'] for s in stops):.0f} min total" if stops else "No extended stops")

st.markdown(
    f"""
    <div class="metric-card" style="margin-bottom: 1rem;">
        <div class="section-label">Start and end details</div>
        <div class="meta-line"><strong>Start time:</strong> {trip_start_time.strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="meta-line"><strong>End time:</strong> {trip_end_time.strftime('%Y-%m-%d %H:%M:%S')}</div>
        <div class="meta-line"><strong>Start location:</strong> {trip_data.iloc[0]['Latitude']:.4f}, {trip_data.iloc[0]['Longitude']:.4f}</div>
        <div class="meta-line"><strong>End location:</strong> {trip_data.iloc[-1]['Latitude']:.4f}, {trip_data.iloc[-1]['Longitude']:.4f}</div>
        <div class="meta-line"><strong>Replay length:</strong> about {animation_duration_s} seconds</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if stops:
    stop_cols = st.columns(min(3, len(stops)))
    for idx, stop in enumerate(sorted(stops, key=lambda item: item["duration_min"], reverse=True)[:3]):
        with stop_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card" style="padding: 14px 16px; margin-bottom: 0.8rem;">
                    <div class="section-label">Stop {idx + 1}</div>
                    <div class="meta-line">{stop['start_time'].strftime('%H:%M:%S')}</div>
                    <div class="meta-line">Duration {stop['duration_min']:.1f} min</div>
                    <div class="meta-line">Lat {stop['lat']:.4f}, Lon {stop['lon']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.plotly_chart(
    build_eta_figure(
        trip_df=trip_with_features,
        playback_step=playback_step,
        target_duration_s=animation_duration_s,
    ),
    use_container_width=True,
    config={"displaylogo": False, "responsive": True},
)

st.plotly_chart(
    build_distance_figure(
        trip_df=trip_with_features,
        playback_step=playback_step,
        target_duration_s=animation_duration_s,
    ),
    use_container_width=True,
    config={"displaylogo": False, "responsive": True},
)

st.plotly_chart(
    build_map_figure(
        trip_df=trip_with_features,
        playback_step=playback_step,
        target_duration_s=animation_duration_s,
    ),
    use_container_width=True,
    config={"displaylogo": False, "responsive": True},
)

st.plotly_chart(
    build_speed_weight_figure(
        trip_df=trip_with_features,
        playback_step=playback_step,
        target_duration_s=animation_duration_s,
    ),
    use_container_width=True,
    config={"displaylogo": False, "responsive": True},
)

st.divider()

with st.expander("How This Demo Works"):
    st.markdown(
        """
        ### Playback

        The demo has three separate visualizations, each with their own **Play** and **Pause** buttons:

        1. **ETA with Quantile Band** - Replay the prediction interval evolution
        2. **Trip Path Map** - Watch the truck's position move across the map
        3. **Speed and Weight Tracking** - Monitor velocity and load changes

        Each plays independently so you can focus on the visualization you care about.

        ### Panels Explained

        - **ETA Panel**: Shows the actual remaining ETA (dashed), the median prediction (solid green), and the P10-P90 confidence interval (shaded band).
        - **Distance Traveled**: Animated line and current marker show distance progression through the trip.
        - **Trip Path Map**: Auto-zooms to the trip region, shows major cities, marks start/end points, and animates the truck's position.
        - **Speed Panel**: Displays velocity over time with the traveled portion highlighted.
        - **Weight Panel**: Shows load changes with time, useful for understanding stop patterns.

        ### Model Interpretation

        - **P10**: Optimistic ETA estimate (10th percentile - truck arrives earlier than this 10% of the time).
        - **P50**: Median ETA estimate (best single prediction).
        - **P90**: Conservative ETA estimate (truck takes longer than this 10% of the time).

        The interval between P10 and P90 represents the 80% confidence band of the prediction.
        """
    )

st.markdown(
    """
    ---
    <div style="text-align: center; color: #666; margin-top: 20px;">
        <p><strong>CS495 Real-Time Truck ETA Prediction</strong></p>
        <p>Model: LightGBM Quantile Regression | Dataset: Segmented Truck Telemetry</p>
    </div>
    """,
    unsafe_allow_html=True,
)

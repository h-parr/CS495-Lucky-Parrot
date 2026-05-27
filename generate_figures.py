"""Generate publication-quality evaluation figures for the CS495 truck ETA project.

Produces three figures saved as high-resolution PNGs in the figures/ directory:
  Figure 1 — MAE Comparison Across ETA Prediction Models (bar chart)
  Figure 2 — Predicted vs Actual ETA for LightGBM Quantile Model (scatter)
  Figure 3 — Quantile Prediction Intervals During a Truck Trip (line chart)

Usage:
    python generate_figures.py

Requires trained models in models/ and segmented_trips.csv in data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SRC_DIR = ROOT / "src"

# ── Canonical full-run metrics (from complete 4.9M-row training run) ───────────
# These are used for Figure 1 so the bar chart always reflects the real numbers
# regardless of what sample size is used for Figures 2 & 3.
CANONICAL = {
    "mae_naive":    137.07,
    "mae_hgb":       77.60,
    "mae_lgbm":      56.55,
    "mae_lgbm_cal":  56.55,  # calibration widens interval, not median
    "rmse_naive":   429.78,
    "rmse_hgb":     103.04,
    "rmse_lgbm":     95.33,
    "coverage_raw":   0.561,
    "coverage_cal":   0.780,
}
sys.path.insert(0, str(SRC_DIR))

from features import FEATURE_COLS, build_features
from metrics import evaluate
from eta_model import add_eta_label, naive_speed_predict, trip_split, conformal_interval_adjustment

KM_PER_MILE = 1.60934
RANDOM_STATE = 42

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

PALETTE = {
    "naive":       "#d62728",   # red
    "hgb":         "#ff7f0e",   # orange
    "lgbm":        "#1f77b4",   # blue
    "lgbm_cal":    "#2ca02c",   # green
    "actual":      "#333333",
    "p50":         "#1f77b4",
    "p10_p90":     "#aec7e8",
    "band_edge":   "#7ab4e8",
    "grid":        "#e0e0e0",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path, sample_trips: int | None = None) -> pd.DataFrame:
    """Load CSV, build features, add label, drop NaNs.

    If sample_trips is given, a random subset of trip IDs is used (speeds up
    the script for development).  Pass None to use the full dataset.
    """
    print(f"Loading {csv_path.name} …")
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df.columns = df.columns.str.strip()

    if sample_trips is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        all_trips = df["trip_id"].unique()
        chosen = rng.choice(all_trips, size=min(sample_trips, len(all_trips)), replace=False)
        df = df[df["trip_id"].isin(chosen)].reset_index(drop=True)
        print(f"  Using {len(chosen)} trips ({len(df):,} rows) for figure generation.")

    print("Building features …")
    df = build_features(df)
    df = add_eta_label(df)
    df = df.dropna(subset=FEATURE_COLS + ["eta_remaining_s"]).reset_index(drop=True)
    df["eta_remaining_min"] = df["eta_remaining_s"] / 60.0
    return df


def load_models() -> dict:
    """Load all persisted models. Returns a dict keyed by model name."""
    models = {}
    for name, fname in [
        ("hgb",    "baseline_hgb.pkl"),
        ("lgb_q10", "lgb_q10.pkl"),
        ("lgb_q50", "lgb_q50.pkl"),
        ("lgb_q90", "lgb_q90.pkl"),
    ]:
        path = MODEL_DIR / fname
        if not path.exists():
            print(f"  WARNING: {fname} not found — skipping.")
            models[name] = None
        else:
            models[name] = joblib.load(path)
            print(f"  Loaded {fname}")
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — MAE Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def make_figure1(
    mae_naive: float,
    mae_hgb: float,
    mae_lgbm: float,
    mae_lgbm_cal: float,
    rmse_naive: float,
    rmse_hgb: float,
    rmse_lgbm: float,
    coverage_raw: float,
    coverage_cal: float,
) -> None:
    """Figure 1: MAE Comparison Across ETA Prediction Models."""
    labels = [
        "Naive\nBaseline",
        "HistGradientBoosting",
        "LightGBM\nP50",
        "LightGBM\n+ Calibration",
    ]
    mae_values = [mae_naive, mae_hgb, mae_lgbm, mae_lgbm_cal]
    colors = [PALETTE["naive"], PALETTE["hgb"], PALETTE["lgbm"], PALETTE["lgbm_cal"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, mae_values, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=1.2)

    # Value labels on top of each bar
    for bar, val in zip(bars, mae_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.5,
            f"{val:.1f} min",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
        )

    # Improvement annotations
    improvement_vs_naive = (mae_naive - mae_lgbm) / mae_naive * 100
    ax.annotate(
        f"−{improvement_vs_naive:.0f}% vs Naive",
        xy=(x[2], mae_lgbm),
        xytext=(x[2] + 0.55, mae_lgbm + 60),
        fontsize=10, color=PALETTE["lgbm"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["lgbm"], lw=1.4),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Absolute Error (minutes)", fontsize=12)
    ax.set_title("Figure 1: MAE Comparison Across ETA Prediction Models", pad=28)
    ax.yaxis.grid(True, color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, mae_naive * 1.18)

    # Coverage annotation below chart
    fig.text(
        0.5, -0.04,
        f"P10–P90 interval coverage:  raw {coverage_raw*100:.1f}%  →  calibrated {coverage_cal*100:.1f}%   "
        f"(target 80%)",
        ha="center", va="top", fontsize=10, color="#555555",
    )

    out = FIGURES_DIR / "figure1_mae_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Predicted vs Actual Scatter
# ─────────────────────────────────────────────────────────────────────────────

def make_figure2(y_true: np.ndarray, p50: np.ndarray, max_points: int = 10_000) -> None:
    """Figure 2: Predicted vs Actual ETA for LightGBM Quantile Model."""
    rng = np.random.default_rng(RANDOM_STATE)
    if len(y_true) > max_points:
        idx = rng.choice(len(y_true), size=max_points, replace=False)
        y_true = y_true[idx]
        p50 = p50[idx]

    # Clip to sensible display range (0–600 min = 10 h)
    cap = 600.0
    mask = (y_true <= cap) & (p50 <= cap) & (y_true >= 0) & (p50 >= 0)
    y_true = y_true[mask]
    p50 = p50[mask]

    errors = p50 - y_true
    mae_val = float(np.mean(np.abs(errors)))

    fig, ax = plt.subplots(figsize=(7, 7))

    # Diagonal reference
    lim = cap
    ax.plot([0, lim], [0, lim], color="#888888", linewidth=1.2, linestyle="--",
            label="Perfect prediction", zorder=2)

    # Scatter
    sc = ax.scatter(
        y_true, p50,
        c=np.abs(errors), cmap="plasma_r", vmin=0, vmax=120,
        alpha=0.35, s=8, linewidths=0, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("|Error| (min)", fontsize=10)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual ETA (minutes)")
    ax.set_ylabel("Predicted ETA — P50 (minutes)")
    ax.set_title("Figure 2: Predicted vs Actual ETA\n(LightGBM Quantile Model, P50)", pad=12)
    ax.yaxis.grid(True, color=PALETTE["grid"], zorder=0)
    ax.xaxis.grid(True, color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)

    ax.text(
        0.04, 0.96,
        f"MAE = {mae_val:.1f} min\nn = {len(y_true):,}",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )
    ax.legend(fontsize=10, loc="lower right")

    out = FIGURES_DIR / "figure2_predicted_vs_actual.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Quantile Prediction Intervals Along a Trip
# ─────────────────────────────────────────────────────────────────────────────

def make_figure3(
    df_full: pd.DataFrame,
    models: dict,
    qhat: float,
    trip_id: str | None = None,
) -> None:
    """Figure 3: Quantile Prediction Intervals During a Truck Trip.

    Picks a long, representative trip and plots actual ETA alongside
    P10 / P50 / P90 predictions ping-by-ping.
    """
    if trip_id is None:
        # Select the trip whose total distance is closest to the 75th percentile
        # — representative of a medium-to-long haul.
        trip_lens = df_full.groupby("trip_id")["eta_remaining_min"].first()
        trip_id = str(trip_lens[trip_lens >= trip_lens.quantile(0.70)].idxmax())
        print(f"  Auto-selected trip_id for Figure 3: {trip_id}")

    trip = df_full[df_full["trip_id"] == trip_id].sort_values("eta_remaining_min", ascending=False).copy()
    trip = trip.reset_index(drop=True)

    if len(trip) < 10:
        print(f"  WARNING: trip {trip_id} has only {len(trip)} pings — picking another.")
        trip_lens = df_full.groupby("trip_id")["eta_remaining_min"].first()
        candidates = trip_lens[trip_lens >= trip_lens.quantile(0.50)].sort_values(ascending=False)
        for tid in candidates.index:
            t = df_full[df_full["trip_id"] == tid]
            if len(t) >= 20:
                trip_id = str(tid)
                trip = t.sort_values("eta_remaining_min", ascending=False).reset_index(drop=True)
                print(f"  Fallback trip_id: {trip_id}  ({len(trip)} pings)")
                break

    X_trip = trip[FEATURE_COLS].to_numpy(dtype=np.float64)
    p10_raw = models["lgb_q10"].predict(X_trip)
    p50_raw = models["lgb_q50"].predict(X_trip)
    p90_raw = models["lgb_q90"].predict(X_trip)

    # Apply conformal calibration (widen interval by qhat on both sides)
    p10_cal = p10_raw - qhat
    p90_cal = p90_raw + qhat

    y_actual = trip["eta_remaining_min"].values
    # x-axis: elapsed time in minutes from trip start
    elapsed_min = trip["elapsed_s"].values / 60.0

    coverage_raw = float(np.mean((y_actual >= p10_raw) & (y_actual <= p90_raw)))
    coverage_cal = float(np.mean((y_actual >= p10_cal) & (y_actual <= p90_cal)))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Calibrated band (shaded)
    ax.fill_between(
        elapsed_min, np.maximum(p10_cal, 0), p90_cal,
        alpha=0.25, color=PALETTE["p10_p90"], label="P10–P90 (calibrated)",
        zorder=1,
    )
    # Raw band edges as dashed lines for reference
    ax.plot(elapsed_min, np.maximum(p10_raw, 0), color=PALETTE["band_edge"],
            linewidth=0.8, linestyle=":", zorder=2)
    ax.plot(elapsed_min, p90_raw, color=PALETTE["band_edge"],
            linewidth=0.8, linestyle=":", zorder=2, label="P10–P90 (raw)")

    # P50 predicted
    ax.plot(elapsed_min, p50_raw, color=PALETTE["p50"],
            linewidth=2.0, label="P50 prediction", zorder=3)

    # Actual ETA
    ax.plot(elapsed_min, y_actual, color=PALETTE["actual"],
            linewidth=2.0, linestyle="-", label="Actual ETA", zorder=4)

    ax.set_xlabel("Elapsed trip time (minutes)")
    ax.set_ylabel("ETA remaining (minutes)")
    ax.set_title(
        "Figure 3: Quantile Prediction Intervals During a Truck Trip\n"
        f"(trip {trip_id},  {len(trip)} pings)",
        pad=12,
    )
    ax.yaxis.grid(True, color=PALETTE["grid"], zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(elapsed_min[0], elapsed_min[-1])
    ax.set_ylim(bottom=0)

    # Coverage annotation
    ax.text(
        0.99, 0.97,
        f"Coverage:  raw {coverage_raw*100:.1f}%  |  calibrated {coverage_cal*100:.1f}%",
        transform=ax.transAxes, va="top", ha="right",
        fontsize=10, color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(0.99, 0.87))

    out = FIGURES_DIR / "figure3_prediction_intervals.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    csv_path = DATA_DIR / "segmented_trips.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found.")

    # Use a sample of trips to keep things fast for figure generation.
    # Increase sample_trips or pass None to use the full dataset.
    df = load_and_prepare(csv_path, sample_trips=200)

    print("\nSplitting data (trip-level) …")
    train_df, val_df, test_df = trip_split(df)
    X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_train = train_df["eta_remaining_min"].values
    X_val   = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_val   = val_df["eta_remaining_min"].values
    X_test  = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_test  = test_df["eta_remaining_min"].values
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    print("\nLoading models …")
    models = load_models()

    # ── Predictions ──────────────────────────────────────────
    naive_pred = naive_speed_predict(test_df)

    hgb_pred  = models["hgb"].predict(X_test) if models["hgb"] else np.zeros_like(y_test)
    p10_raw   = models["lgb_q10"].predict(X_test)
    p50_raw   = models["lgb_q50"].predict(X_test)
    p90_raw   = models["lgb_q90"].predict(X_test)

    p10_val   = models["lgb_q10"].predict(X_val)
    p90_val   = models["lgb_q90"].predict(X_val)

    # Conformal calibration (split-conformal on validation)
    qhat = conformal_interval_adjustment(y_val, p10_val, p90_val, alpha=0.2)
    p10_cal = p10_raw - qhat
    p90_cal = p90_raw + qhat

    # ── Metrics ───────────────────────────────────────────────
    r_naive  = evaluate(y_test, naive_pred)
    r_hgb    = evaluate(y_test, hgb_pred)
    r_lgbm   = evaluate(y_test, p50_raw, p10_raw, p90_raw)
    r_lgbm_c = evaluate(y_test, p50_raw, p10_cal, p90_cal)

    print("\n── Model Comparison (test set) ──")
    for label, r in [
        ("Naive",            r_naive),
        ("HGB",              r_hgb),
        ("LightGBM P50",     r_lgbm),
        ("LightGBM + Cal",   r_lgbm_c),
    ]:
        cov = r.get("coverage_p10_p90", float("nan"))
        print(f"  {label:<22} MAE={r['mae']:.2f}  RMSE={r['rmse']:.2f}  Coverage={cov:.3f}")

    # ── Figure 1 — always uses canonical full-run numbers ───────────────────
    print("\nGenerating Figure 1 (using canonical full-run metrics) …")
    make_figure1(
        mae_naive   = CANONICAL["mae_naive"],
        mae_hgb     = CANONICAL["mae_hgb"],
        mae_lgbm    = CANONICAL["mae_lgbm"],
        mae_lgbm_cal= CANONICAL["mae_lgbm_cal"],
        rmse_naive  = CANONICAL["rmse_naive"],
        rmse_hgb    = CANONICAL["rmse_hgb"],
        rmse_lgbm   = CANONICAL["rmse_lgbm"],
        coverage_raw = CANONICAL["coverage_raw"],
        coverage_cal = CANONICAL["coverage_cal"],
    )

    # ── Figure 2 ─────────────────────────────────────────────
    print("\nGenerating Figure 2 …")
    make_figure2(y_test, p50_raw)

    # ── Figure 3 ─────────────────────────────────────────────
    print("\nGenerating Figure 3 …")
    make_figure3(df, models, qhat)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()

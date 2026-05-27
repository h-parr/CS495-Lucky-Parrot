"""Quick evaluation of saved models on a large sample of the full dataset.

Loads persisted models from models/ and reports MAE/RMSE/Coverage so we can
get the real LightGBM numbers without re-running the full multi-hour training.

Usage:
    python eval_saved_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from features import FEATURE_COLS, build_features
from metrics import evaluate
from eta_model import add_eta_label, naive_speed_predict, trip_split, conformal_interval_adjustment

RANDOM_STATE = 42
SAMPLE_TRIPS = 2000   # large enough for stable metrics, fast enough to run


def main() -> None:
    csv_path = ROOT / "data" / "segmented_trips.csv"
    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df.columns = df.columns.str.strip()

    rng = np.random.default_rng(RANDOM_STATE)
    all_trips = df["trip_id"].unique()
    chosen = rng.choice(all_trips, size=min(SAMPLE_TRIPS, len(all_trips)), replace=False)
    df = df[df["trip_id"].isin(chosen)].reset_index(drop=True)
    print(f"  Sampled {len(chosen)} trips ({len(df):,} rows)")

    print("Building features ...")
    df = build_features(df)
    df = add_eta_label(df)
    df = df.dropna(subset=FEATURE_COLS + ["eta_remaining_s"]).reset_index(drop=True)
    df["eta_remaining_min"] = df["eta_remaining_s"] / 60.0

    print("Splitting (trip-level) ...")
    train_df, val_df, test_df = trip_split(df)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_test = test_df["eta_remaining_min"].values
    X_val  = val_df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y_val  = val_df["eta_remaining_min"].values
    print(f"  test={len(test_df):,}  val={len(val_df):,}")

    print("\nLoading saved models ...")
    hgb    = joblib.load(ROOT / "models" / "baseline_hgb.pkl")
    lgb_q10 = joblib.load(ROOT / "models" / "lgb_q10.pkl")
    lgb_q50 = joblib.load(ROOT / "models" / "lgb_q50.pkl")
    lgb_q90 = joblib.load(ROOT / "models" / "lgb_q90.pkl")

    naive_pred = naive_speed_predict(test_df)
    hgb_pred   = hgb.predict(X_test)
    p10 = lgb_q10.predict(X_test)
    p50 = lgb_q50.predict(X_test)
    p90 = lgb_q90.predict(X_test)

    p10_val = lgb_q10.predict(X_val)
    p90_val = lgb_q90.predict(X_val)

    qhat = conformal_interval_adjustment(y_val, p10_val, p90_val, alpha=0.2)
    p10_cal = p10 - qhat
    p90_cal = p90 + qhat

    r_naive  = evaluate(y_test, naive_pred)
    r_hgb    = evaluate(y_test, hgb_pred)
    r_lgbm   = evaluate(y_test, p50, p10, p90)
    r_lgbm_c = evaluate(y_test, p50, p10_cal, p90_cal)

    print("\n=== Model Comparison (saved models, sampled test set) ===")
    print(f"  {'Model':<28} {'MAE':>8}  {'RMSE':>9}  {'Coverage':>10}")
    print(f"  {'-'*60}")
    for label, r in [
        ("Naive speed",          r_naive),
        ("HistGradientBoosting", r_hgb),
        ("LightGBM P50",         r_lgbm),
        ("LightGBM + Calibration", r_lgbm_c),
    ]:
        cov = r.get("coverage_p10_p90", float("nan"))
        cov_str = f"{cov*100:.1f}%" if not np.isnan(cov) else "  n/a"
        print(f"  {label:<28} {r['mae']:>8.2f}  {r['rmse']:>9.2f}  {cov_str:>10}")

    print(f"\n  Conformal qhat (80% target): {qhat:.2f} min")
    imp_naive = (r_naive['mae'] - r_lgbm['mae']) / r_naive['mae'] * 100
    imp_hgb   = (r_hgb['mae']  - r_lgbm['mae']) / r_hgb['mae']   * 100
    print(f"  LightGBM improvement vs Naive: {imp_naive:.1f}%")
    print(f"  LightGBM improvement vs HGB:   {imp_hgb:.1f}%")


if __name__ == "__main__":
    main()

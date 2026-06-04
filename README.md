# CS495 - Real Time Truck ETA prediction

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-green)
![ETA](https://img.shields.io/badge/ETA-Quantile%20Regression-orange)
![Telematics](https://img.shields.io/badge/Telematics-Real%20Time-red)

# Project Description

Using telematics to build a real-time model and streaming service to predict estimated time of arrival (ETA) and uncertainty.

## Latest Performance Snapshot (June 3, 2026)

From the latest full training run on `data/segmented_trips.csv`:

- Naive speed: MAE 230.67, RMSE 1013.65
- HGB baseline: MAE 27.75, RMSE 77.65
- **LightGBM best variant: MAE 26.52, RMSE 80.37** ⭐ **Winner**
- LightGBM raw P50: MAE 28.61, RMSE 102.92
- Ensemble stacking (Ridge meta-model): MAE 29.53, RMSE 83.69
- Reported MAE improvement vs HGB baseline: 4.4% (LightGBM best)

Important: when you run training (`python src/eta_model.py`), the reported "LightGBM best variant" is selected automatically as the LightGBM variant with the lowest MAE (currently choosing between raw P50 and near-opt). The improvement vs HGB in console output and in this README follows that same definition.

### Why Ensemble Stacking Underperformed

Ensemble stacking (combining naive speed, HGB, LightGBM raw P50, and LightGBM near-opt via Ridge regression) achieved **29.53 MAE—6.4% worse than HGB baseline and 11.4% worse than LightGBM best**. This counterintuitive result reveals why model stacking isn't universally superior:

**Root Causes:**

1. **LightGBM near-opt is already an optimized ensemble**: The "best" LightGBM model incorporates:
   - Quantile-specific training (P10/P50/P90 heads)
   - A specialist regressor weighted heavily on near-arrival samples
   - Validated blending logic that selects between global and specialist predictions
   - When we stacked this with other models, we created conflicting signal paths

2. **Conflicting optimization objectives**: The base models optimize for incompatible goals:
   - Naive: Distance/speed physics (deterministic, no training)
   - HGB: Overall MAE using squared error loss
   - LightGBM raw: Quantile loss at the median (0.5 quantile)
   - LightGBM near-opt: Weighted MAE on near-arrival samples, already blended
   - Ridge meta-model couldn't find weights that simultaneously satisfy all objectives

3. **Validation-set overfitting in meta-training**: The Ridge meta-model trained on validation predictions that were already highly optimized for the validation set. This created a second-order optimization problem where the meta-model learned validation-specific quirks rather than robust blending logic.

**Lesson Learned**: Best-in-class single models (especially those with built-in specialist blending) can outperform naive ensemble approaches. Effective ensembles require truly diverse base models trained independently without internal blending. See `MODEL_PERFORMANCE_LOG.md` Run 6 for detailed metrics and analysis.

# Objectives

• ETA Model: Quantile regression (P10/P50/P90) to predict estimated time of arrival

• Streaming Inference: Real-time state per (VIN, TripId) with continuous updates at each telemetry point

• Backtesting: Replay historical trip streams, evaluate stability smoothing, and ablation study

# Load State Detection (Phase 2b)

The project now uses a heuristic load-state algorithm as the primary production path.

## Primary algorithm (used in training)

- Method: centered rolling mean over per-trip speed (window = 3 pings).
- Rule: if smoothed speed < 10 mph, classify ping as loaded (`1`); otherwise unloaded (`0`).
- Derived feature: `load_change_count` counts loaded/unloaded transitions within a trip.

How it works, step by step:
1. Sort pings by `trip_id` and `Timestamp`.
2. Smooth noisy speed with a rolling mean.
3. Convert each ping into a binary state using the 10 mph threshold.
4. Count state transitions per trip and attach both features to the model row set.

Why this method is primary:
- It is much faster on large datasets than full change-point optimization.
- It preserves useful regime-shift signal for ETA modeling.

## Optional comparison methods

- PELT and Binary Segmentation are still implemented in `src/load_state.py` for offline CPD comparison/analysis.
- The feature pipeline defaults to heuristic mode for runtime practicality.

# Dataset

The `data/sample_trucks_dataset.csv` file contains real-time telematics records used for model training and evaluation. Each row represents a single telemetry ping from a truck. The tracked variables are described below:

| Variable | Description |
|---|---|
| `VIN` | Unique identifier for each truck (Vehicle Identification Number / Truck ID) |
| `Timestamp` | Date and time of the telemetry ping, used to compute elapsed trip time and derive time-of-day features |
| `Latitude` | GPS latitude coordinate of the truck at the time of the ping |
| `Longitude` | GPS longitude coordinate of the truck at the time of the ping |
| `Speed` | Instantaneous speed of the truck (mph) at the time of the ping |
| `Weight_lbs` | Reported payload weight of the truck in pounds at the time of the ping |
| `Device_Type` | Type of telematics device installed on the truck (e.g., OEM) |
| `Source` | Manufacturer or data source of the telematics device (e.g., Volvo) |

# Tools / Technologies

Tools: Python, LightGBM/CatBoost, DBSCAN/HDBSCAN, H3/geohash spatial indexing, Scikit-learn, Streamlit

# Setup (Makefile Commands)

This project uses a cross-platform `Makefile` to streamline setup and common workflows on macOS/Linux and Windows (with GNU Make installed).

## 1) Create and Populate the Virtual Environment

```bash
make install
```

This will:
- create `.venv` using Python 3.13
- upgrade `pip`
- install packages from `requirements.txt`

For development tools (pytest/coverage/ruff):

```bash
make install-dev
```

## 2) Common Commands

```bash
make help      # list all available targets
make audit     # run data audit script
make segment   # run trip segmentation
make train     # run ETA model training
make test      # run pytest suite
make lint      # run ruff linter
make clean     # remove venv and Python cache files
make rebuild   # clean + install
```

## 3) Optional Maintenance Commands

```bash
make freeze    # write currently installed packages to requirements.txt
```

# Interactive Demo

A **Streamlit demo app** is included to visualize ETA predictions on sample truck trips:

```bash
streamlit run demo_app.py
```

## Features

- **🗺️ Animated Trip Map**: Watch the truck route with live position marker, detected stops, and planned route
- **📊 Speed & Weight Profiles**: Time-series charts showing speed changes, stops, and load variations
- **📍 ETA Confidence Intervals**: Quantile regression predictions (P10/P50/P90) showing 80% confidence band
- **🛑 Stop Detection**: Automatic detection and visualization of stops > 1 minute duration
- **📈 Real-time Metrics**: Distance traveled, remaining distance, average speed, load info
- **⏱️ Timeline Navigation**: Scrub through the trip to see how ETA predictions evolve

## Models Used

The demo uses pre-trained LightGBM quantile regressors:
- `lgb_q10.pkl`: 10th percentile (optimistic ETA)
- `lgb_q50.pkl`: Median / best estimate
- `lgb_q90.pkl`: 90th percentile (conservative ETA)

## Data Requirements

The demo requires:
- `data/demo_trips_subset.csv`: Small real-data subset for the Streamlit demo
- `models/*.pkl`: Pre-trained model artifacts from `make train`

If you only have the full dataset somewhere outside the repo, generate a demo-sized subset:

```bash
python src/create_demo_subset.py --input "C:/path/to/segmented_trips.csv"
```

This writes `data/demo_trips_subset.csv`, which is allowed by `.gitignore` and can be committed for demo use.

# Development Workflow

Use a branch-based workflow instead of committing code changes directly to `main`.

1. Branch from `develop` for each feature or fix.
2. Open a Pull Request into `develop` first.
3. Ensure CI passes (`ruff` + `pytest`).
4. Merge to `main` only when a feature set is stable/reviewed.

Suggested naming:

- `feature/trip-segmentation`
- `feature/load-state-cpd`
- `fix/data-audit-cli`

# Timeline

Minimal implementation roadmap (8–10 weeks)
Build historical trip table → derive trip features

Train ETA quantile regressor (P10/P50/P90)

Build streaming feature state + inference loop

Backtest on replay streams + stability smoothing

Package as a small library + demo UI

Attribution Requirement: Any academic, research, or commercial usage must cite the original repository and authors.

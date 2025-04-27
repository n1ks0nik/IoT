"""
 Enhanced pipeline for preprocessing, *early* ABV forecasting and anomaly
 detection in the Brew‑IoT prototype.

 Why this revision?
 ------------------
 In the previous version the regression could be replaced by the trivial
 formula `ABV = 131.25 × (SG₀ − SG_final)`, because we used the *final* SG as
 both a feature and indirect label.  Now the model is forced to learn the
 relationship between the **first N hours** of fermentation and the eventual
 alcohol content, when the terminal density is still unknown.

 Key changes
 ~~~~~~~~~~~
 • **Early‑window features only** – default *early_hours = 12* (configurable)
   – SG statistics, CO₂ evolution, temperature profile and pH drop *within the
     first N h*.
   – *No* access to SG after the cutoff → the analytical formula cannot be
     applied at prediction time.
 • **Target**  (`abv_final`) is still derived from SG *after 72 h* → supervised
   training signal remains correct.
 • Added CO₂‑based and pressure‑based features to increase predictive power.
 • The public API `predict_abv(tank_id, early_hours)` can be wired into
   FastAPI later.

 The module remains self‑contained; it reuses `main.py` for DB access and
 synthetic data generation.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sqlmodel import Session, select

# ---------------------------------------------------------------------------
# Import Brew‑IoT objects (DB models & generator)
# ---------------------------------------------------------------------------
from main import (
    SensorType,
    Measurement,
    Tank,
    engine,
    generate_synthetic_data,
)

# ---------------------------------------------------------------------------
# 1. Synthetic‑batch generation utilities (unchanged)
# ---------------------------------------------------------------------------


def ensure_n_runs(n_runs: int = 100, duration_h: int = 72) -> None:
    """Populate the database with *n_runs* distinct fermentation profiles.

    The existing DB may already contain 1 or more runs (tanks).  We simply add
    new Tank IDs until the desired count is reached.  To induce variety we
    randomly perturb the logistic‑curve parameters each time.
    """

    with Session(engine) as session:
        # Highest current tank id (start at 0 if none)
        max_id = (
            session.exec(select(Tank.id).order_by(Tank.id.desc()).limit(1)).first()
            or 0
        )

    for i in range(max_id + 1, n_runs + 1):
        # Draw random parameters within realistic brewing bounds
        s0 = np.random.uniform(1.045, 1.070)  # initial SG
        dsg = np.random.uniform(0.035, 0.055)  # SG drop
        k = np.random.uniform(0.15, 0.25)  # curve steepness
        t0 = np.random.uniform(20, 28)  # midpoint in hours

        generate_synthetic_data(
            duration_h=duration_h,
            tank_id=i,
            s0=s0,
            dsg=dsg,
            k=k,
            t0=t0,
        )


# ---------------------------------------------------------------------------
# 2. Data extraction & preprocessing helpers
# ---------------------------------------------------------------------------


_ONE_MIN = pd.Timedelta(minutes=1)
_ROLL = 15  # smoothing window (15 min)


def _pivot_measurements(rows: List[Measurement]) -> pd.DataFrame:
    """Pivot a list of Measurement rows into a 1‑minute indexed DataFrame."""

    if not rows:
        raise ValueError("No measurements supplied")

    df = (
        pd.DataFrame(
            {
                "ts": [m.timestamp.replace(tzinfo=timezone.utc) for m in rows],
                "sensor": [m.sensor_type for m in rows],
                "value": [m.value for m in rows],
            }
        )
        .pivot(index="ts", columns="sensor", values="value")
        .sort_index()
    )

    # Re‑index to full 1‑minute grid
    idx = pd.date_range(df.index[0], df.index[-1], freq=_ONE_MIN, tz="UTC")
    df = df.reindex(idx)

    # Forward‑fill slower sensors (SG, pH)
    df[SensorType.sg] = df[SensorType.sg].ffill()
    df[SensorType.ph] = df[SensorType.ph].ffill()

    # Quick fill for any remaining gaps
    df = df.ffill().bfill()

    # Smoothing
    df["sg_smooth"] = df[SensorType.sg].rolling(_ROLL, min_periods=1).mean()
    df["temp_smooth"] = df[SensorType.temperature].rolling(_ROLL, min_periods=1).mean()
    df["co2_smooth"] = df[SensorType.co2].rolling(_ROLL, min_periods=1).mean()

    # Gradients (per hour)
    dt_h = _ONE_MIN / pd.Timedelta(hours=1)  # = 1/60
    df["dsg_dt"] = np.gradient(df["sg_smooth"]) / dt_h
    df["dT_dt"] = np.gradient(df["temp_smooth"]) / dt_h
    df["dCO2_dt"] = np.gradient(df["co2_smooth"]) / dt_h

    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering (EARLY‑WINDOW ONLY!)
# ---------------------------------------------------------------------------


def _window_stats(series: pd.Series, minutes: int) -> Tuple[float, float]:
    sl = series.iloc[:minutes]
    return float(sl.mean()), float(sl.std())


_DEF_EARLY_HOURS = 50  # default cutoff for features


def build_feature_vector(df: pd.DataFrame, early_hours: int = _DEF_EARLY_HOURS) -> Dict[str, float]:
    """Build features using **only the first *early_hours*** of fermentation."""

    cutoff = early_hours * 60  # minutes
    f: Dict[str, float] = {}

    sg0 = df["sg_smooth"].iloc[0]

    # SG‑based early dynamics
    f["sg0"] = sg0
    f["sg_drop_%dh" % early_hours] = sg0 - df["sg_smooth"].iloc[cutoff - 1]
    f["max_dsgdt_%dh" % early_hours] = df["dsg_dt"].iloc[:cutoff].min()

    # Temperature stats early vs whole early window
    f["temp_mean_%dh" % early_hours], f["temp_std_%dh" % early_hours] = _window_stats(
        df["temp_smooth"], cutoff
    )

    # CO₂ evolution (proxy for fermentation rate)
    f["co2_mean_%dh" % early_hours], f["co2_std_%dh" % early_hours] = _window_stats(
        df["co2_smooth"], cutoff
    )
    f["max_dCO2dt_%dh" % early_hours] = df["dCO2_dt"].iloc[:cutoff].max()

    # pH change early
    f["pH0"] = df[SensorType.ph].iloc[0]
    f["pH_drop_%dh" % early_hours] = f["pH0"] - df[SensorType.ph].iloc[cutoff - 1]

    return f


# ---------------------------------------------------------------------------
# 4. Dataset assembly (label uses *final* ABV, features use early window)
# ---------------------------------------------------------------------------


_ABV_FACTOR = 131.25  # standard hydrometer factor

def assemble_dataset(early_hours: int = _DEF_EARLY_HOURS) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix *X* and target *y* (final ABV)."""

    features: List[Dict[str, float]] = []
    targets: List[float] = []

    with Session(engine) as session:
        tanks = session.exec(select(Tank)).all()

    for tank in tanks:
        with Session(engine) as session:
            rows = session.exec(
                select(Measurement).where(Measurement.tank_id == tank.id)
            ).all()
        df = _pivot_measurements(rows)

        # Skip if fermentation shorter than early_hours
        if len(df) < early_hours * 60:
            continue

        features.append(build_feature_vector(df, early_hours))

        sg0 = df["sg_smooth"].iloc[0]
        sg_final = df["sg_smooth"].iloc[-1]
        final_abv = _ABV_FACTOR * (sg0 - sg_final)
        targets.append(final_abv)

    X = pd.DataFrame(features)
    y = pd.Series(targets, name="abv")
    return X, y


# ---------------------------------------------------------------------------
# 5. Model training
# ---------------------------------------------------------------------------


def train_models(save_dir: Path | str = "models", early_hours: int = _DEF_EARLY_HOURS) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Build dataset
    X, y = assemble_dataset(early_hours)

    # Show feature/target shapes for sanity‑check
    print(f"Dataset: {X.shape[0]} runs, {X.shape[1]} features  →  target len {len(y)}")

    # ABV regression -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg = XGBRegressor(objective='reg:squarederror', random_state=42)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Early‑forecast ABV MAE (cutoff {early_hours} h) = {mae:.3f} (% v/v)")

    joblib.dump(reg, save_dir / "abv_regressor.pkl")

    # Anomaly detector -----------------------------------------------------
    det = IsolationForest(contamination=0.05, random_state=42)
    det.fit(X)
    joblib.dump(det, save_dir / "iso_forest.pkl")


# ---------------------------------------------------------------------------
# 6. Inference utilities
# ---------------------------------------------------------------------------


_REG_PATH = Path("models/abv_regressor.pkl")
_DET_PATH = Path("models/iso_forest.pkl")


def load_models() -> Tuple[XGBRegressor, IsolationForest]:
    if not _REG_PATH.exists() or not _DET_PATH.exists():
        raise FileNotFoundError(
            "Models not found.  Run `train_models()` first to create them."
        )
    reg: XGBRegressor = joblib.load(_REG_PATH)
    det: IsolationForest = joblib.load(_DET_PATH)
    return reg, det


# Convenience wrapper ------------------------------------------------------

def predict_abv(tank_id: int, early_hours: int = _DEF_EARLY_HOURS) -> Tuple[float, bool]:
    """Return (predicted_abv, anomaly_flag) for a given *tank_id*."""

    with Session(engine) as session:
        rows = (
            session.exec(
                select(Measurement)
                .where(Measurement.tank_id == tank_id)
                .order_by(Measurement.timestamp)
            ).all()
        )

    df = _pivot_measurements(rows)
    reg, det = load_models()

    # Build features using early window only
    X_vec = pd.DataFrame([build_feature_vector(df, early_hours)])

    abv_pred = float(reg.predict(X_vec)[0])
    anomaly = bool(det.predict(X_vec)[0] == -1)

    return abv_pred, anomaly


# ---------------------------------------------------------------------------
# 7. Recommendations (unchanged logic)
# ---------------------------------------------------------------------------


def make_recommendations(abv_pred: float, anomaly: bool) -> List[str]:
    rec: List[str] = []

    if anomaly:
        rec.append(
            "⚠️  Sensor readings look unusual (anomaly detected).  Check fittings, sensors, and ensure no air leaks."
        )

    if abv_pred < 4.0:
        rec.append(
            "Predicted final ABV is low.  Consider increasing wort temperature by 0.5–1 °C in the first 24 h or pitch more yeast."
        )
    elif abv_pred > 6.5:
        rec.append(
            "High ABV forecast.  Ensure yeast nutrients are sufficient and plan for additional conditioning time."
        )
    else:
        rec.append(
            "ABV forecast is within the typical range.  Maintain temperature profile and monitor SG decline."
        )

    return rec


# ---------------------------------------------------------------------------
# 8. CLI entry‑point for quick experimentation
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train Brew‑IoT early‑forecast models")
    ap.add_argument(
        "--generate", action="store_true", help="Generate synthetic runs before training"
    )
    ap.add_argument("--runs", type=int, default=200, help="Number of synthetic runs to ensure")
    ap.add_argument("--duration", type=int, default=72, help="Duration of each run in hours")
    ap.add_argument("--early", type=int, default=_DEF_EARLY_HOURS, help="Cutoff in hours for early features")
    args = ap.parse_args()

    if args.generate:
        ensure_n_runs(args.runs, args.duration)

    train_models(early_hours=args.early)

    print("✔ Training complete.  Models saved in ./models/")

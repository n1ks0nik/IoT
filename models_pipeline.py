from __future__ import annotations

from pathlib import Path
from datetime import timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy.optimize import curve_fit
from sqlmodel import Session, select


from data_backend import (
    SQLModel,
    SensorType,
    Measurement,
    Tank,
    engine,
    generate_synthetic_data,
)

# ---------------------------------------------------------------------------
# 1. Synthetic‑batch generation utilities
# ---------------------------------------------------------------------------


def ensure_n_runs(
    n_runs: int = 100,
    duration_h: int = 72,
    noise_scale: float = 1.0,
) -> None:
    """Populate the database with *n_runs* fermentation profiles.

    Parameters
    ----------
    n_runs : int, default 100
        How many distinct tanks should be present in the DB in total.
    duration_h : int, default 72
        Length of each synthetic run (hours).
    noise_scale : float, default 1.0
        Multiplier applied to the stochastic terms inside
        ``generate_synthetic_data()``.  Use values < 1 to inspect model
        behaviour on cleaner data.
    """

    SQLModel.metadata.create_all(engine, checkfirst=True)

    with Session(engine) as session:
        existing = set(session.exec(select(Tank.id)).all())
        for tid in range(1, n_runs + 1):
            if tid not in existing:
                session.add(Tank(id=tid, name=f"Simulated tank #{tid}"))
        session.commit()

    with Session(engine) as session:
        existing = set(session.exec(select(Tank.id)).all())
        for tid in range(1, n_runs + 1):
            if tid not in existing:
                session.add(Tank(id=tid, name=f"Simulated tank #{tid}"))
        session.commit()

    for i in range(1, n_runs + 1):
        # Draw random parameters within realistic brewing bounds
        s0 = np.random.uniform(1.045, 1.070)
        dsg = np.random.uniform(0.035, 0.055)
        k = np.random.uniform(0.15, 0.25)
        t0 = np.random.uniform(20, 28)

        generate_synthetic_data(
            duration_h=duration_h,
            tank_id=i,
            s0=s0,
            dsg=dsg,
            k=k,
            t0=t0,
            noise_scale=noise_scale,  # <‑‑ new
        )


# ---------------------------------------------------------------------------
# 2. Data extraction & preprocessing helpers
# ---------------------------------------------------------------------------

_ONE_MIN = pd.Timedelta(minutes=1)
_ROLL = 15  # smoothing window (15 min)


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

    # First derivatives (per hour)
    dt_h = _ONE_MIN / pd.Timedelta(hours=1)  # = 1/60
    df["dsg_dt"] = np.gradient(df["sg_smooth"]) / dt_h
    df["dT_dt"] = np.gradient(df["temp_smooth"]) / dt_h
    df["dCO2_dt"] = np.gradient(df["co2_smooth"]) / dt_h

    # Second derivatives ---------------------------------------------------
    df["d2sg_dt2"] = np.gradient(df["dsg_dt"]) / dt_h
    df["d2T_dt2"] = np.gradient(df["dT_dt"]) / dt_h

    return df


# ---------------------------------------------------------------------------
# 3. Feature engineering helpers
# ---------------------------------------------------------------------------

_DEF_EARLY_HOURS = 36  # default cutoff for features


def _window_stats(series: pd.Series, minutes: int) -> Tuple[float, float]:
    sl = series.iloc[:minutes]
    return float(sl.mean()), float(sl.std())


# -- logistic fit ----------------------------------------------------------

def _fit_logistic_early(times_h: np.ndarray, sg: np.ndarray) -> Tuple[float, float]:
    """Fit *k* and *t₀* of the logistic SG curve using early data only.

    Returns *(k, t0)*.  Any optimisation error results in ``(nan, nan)`` so the
    downstream model can learn to ignore missing fits if needed.
    """

    def logistic(t, s0, dsg, k, t0):
        return s0 - dsg / (1 + np.exp(-k * (t - t0)))

    try:
        p0 = [sg[0], 0.04, 0.2, times_h[len(times_h) // 2]]
        bounds = ([1.030, 0.02, 0.05, 0.0], [1.080, 0.08, 1.0, times_h[-1]])
        popt, _ = curve_fit(logistic, times_h, sg, p0=p0, bounds=bounds, maxfev=10000)
        return float(popt[2]), float(popt[3])  # k, t0
    except Exception:
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
# 4. Feature engineering (EARLY‑WINDOW ONLY)
# ---------------------------------------------------------------------------


def build_feature_vector(
    df: pd.DataFrame,
    early_hours: int = _DEF_EARLY_HOURS,
) -> Dict[str, float]:
    """Build engineered features from **only the first *early_hours*** hours."""

    cutoff = early_hours * 60  # minutes
    f: Dict[str, float] = {}

    # ‑‑ SG & its dynamics --------------------------------------------------
    sg0 = df["sg_smooth"].iloc[0]
    f["sg0"] = sg0
    f[f"sg_drop_{early_hours}h"] = sg0 - df["sg_smooth"].iloc[cutoff - 1]
    f[f"max_dsgdt_{early_hours}h"] = df["dsg_dt"].iloc[:cutoff].min()

    # Second derivative (curvature)
    f[f"min_d2sgdt2_{early_hours}h"] = df["d2sg_dt2"].iloc[:cutoff].min()

    # Logistic fit parameters k & t0
    times_h = np.arange(cutoff) / 60
    k_fit, t0_fit = _fit_logistic_early(times_h, df["sg_smooth"].iloc[:cutoff].values)
    f["k_fit"] = k_fit
    f["t0_fit_h"] = t0_fit

    # ‑‑ Temperature -------------------------------------------------------
    f[f"temp_mean_{early_hours}h"], f[f"temp_std_{early_hours}h"] = _window_stats(
        df["temp_smooth"], cutoff
    )
    f[f"temp_max_rate_{early_hours}h"] = df["dT_dt"].iloc[:cutoff].max()
    f[f"temp_min_rate_{early_hours}h"] = df["dT_dt"].iloc[:cutoff].min()
    f["temp_spike_count"] = (np.abs(df["dT_dt"].iloc[:cutoff]) > 0.3).sum()

    # ‑‑ CO₂ evolution -----------------------------------------------------
    f[f"co2_mean_{early_hours}h"], f[f"co2_std_{early_hours}h"] = _window_stats(
        df["co2_smooth"], cutoff
    )
    f[f"max_dCO2dt_{early_hours}h"] = df["dCO2_dt"].iloc[:cutoff].max()

    # Integrals over 12 h blocks
    for start in range(0, early_hours, 12):
        end = min(start + 12, early_hours)
        area = df["co2_smooth"].iloc[start * 60 : end * 60].sum() / 60  # ppm·h
        f[f"co2_int_{start}_{end}h"] = area

    # ‑‑ pH dynamics -------------------------------------------------------
    pH0 = df[SensorType.ph].iloc[0]
    pH_end = df[SensorType.ph].iloc[cutoff - 1]
    f["pH0"] = pH0
    f["pH_drop_{early_hours}h"] = pH0 - pH_end
    f[f"pH_slope_{early_hours}h"] = (pH_end - pH0) / early_hours
    f[f"pH_std_{early_hours}h"] = df[SensorType.ph].iloc[:cutoff].std()

    return f


# ---------------------------------------------------------------------------
# 5. Dataset assembly
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
# 6. Model training
# ---------------------------------------------------------------------------


def train_models(
    save_dir: Path | str = "models",
    early_hours: int = _DEF_EARLY_HOURS,
    cv_splits: int = 5,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Build dataset
    X, y = assemble_dataset(early_hours)
    print(f"Dataset: {X.shape[0]} runs, {X.shape[1]} features → target len {len(y)}")

    # ─── ABV regression (K‑fold CV + early stopping) ──────────────────────
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    reg_params = dict(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,  # L1
        reg_lambda=1.0,  # L2
        early_stopping_rounds=50
    )

    fold_mae: List[float] = []
    best_iters: List[int] = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        reg = XGBRegressor(**reg_params)
        reg.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_val_pred = reg.predict(X_val, iteration_range=(0, reg.best_iteration + 1))
        mae = mean_absolute_error(y_val, y_val_pred)
        fold_mae.append(mae)
        best_iters.append(reg.best_iteration)

        print(
            f"  fold {fold}: best_iter={reg.best_iteration:4d} ⋅ MAE={mae:.3f} (% v/v)"
        )

    print(
        f"CV {cv_splits}‑fold ABV MAE = {np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f} (% v/v)"
    )

    # Refit on full data using the averaged best number of trees ------------
    optimal_n_estimators = max(1, int(np.mean(best_iters)))
    print(f"Retraining on full dataset (n_estimators={optimal_n_estimators}) …")

    reg_params['n_estimators'] = optimal_n_estimators
    reg_params.pop("early_stopping_rounds", None)

    final_reg = XGBRegressor(
        **reg_params
    )
    final_reg.fit(X, y)
    joblib.dump(final_reg, save_dir / "abv_regressor.pkl")

    # ─── Anomaly detector --------------------------------------------------
    det = IsolationForest(contamination=0.05, random_state=42)
    det.fit(X)
    joblib.dump(det, save_dir / "iso_forest.pkl")

    print("✔ Models trained & saved in", save_dir)


# ---------------------------------------------------------------------------
# 7. Inference utilities (unchanged)
# ---------------------------------------------------------------------------

_REG_PATH = Path("models/abv_regressor.pkl")
_DET_PATH = Path("models/iso_forest.pkl")


def load_models() -> Tuple[XGBRegressor, IsolationForest]:
    if not _REG_PATH.exists() or not _DET_PATH.exists():
        raise FileNotFoundError(
            "Models not found. Run `train_models()` first to create them."
        )
    reg: XGBRegressor = joblib.load(_REG_PATH)
    det: IsolationForest = joblib.load(_DET_PATH)
    return reg, det


# Convenience wrapper ------------------------------------------------------

def predict_abv(
    tank_id: int,
    early_hours: int = _DEF_EARLY_HOURS,
) -> Tuple[float, bool]:
    """Return *(predicted_abv, anomaly_flag)* for a given *tank_id*."""

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

    X_vec = pd.DataFrame([build_feature_vector(df, early_hours)])

    abv_pred = float(
        reg.predict(X_vec, iteration_range=(0, getattr(reg, "best_iteration", None)))
    )
    anomaly = bool(det.predict(X_vec)[0] == -1)

    return abv_pred, anomaly


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Train Brew‑IoT early‑forecast models (improved version)"
    )
    ap.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic runs before training",
    )
    ap.add_argument("--runs", type=int, default=50, help="Number of synthetic runs")
    ap.add_argument("--duration", type=int, default=72, help="Duration of each run, h")
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=0.3,
        help="Scale factor for noise when generating synthetics (σ ← σ·scale)",
    )
    ap.add_argument(
        "--early", type=int, default=_DEF_EARLY_HOURS, help="Early window in hours"
    )
    args = ap.parse_args()

    if args.generate:
        ensure_n_runs(args.runs, args.duration, args.noise_scale)

    train_models(early_hours=args.early)

    print("✔ Training complete. Models saved in ./models/")
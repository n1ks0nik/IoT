from sqlmodel import SQLModel, Field, create_engine, Session, select
from enum import Enum
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd

DATABASE_URL = "sqlite:///fermentation1.db"
engine = create_engine(DATABASE_URL, echo=False)


class SensorType(str, Enum):
    temperature = "temperature"
    sg = "sg"
    ph = "ph"
    co2 = "co2"
    pressure = "pressure"
    level = "level"


class Measurement(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(index=True)
    tank_id: int = Field(index=True)
    sensor_type: SensorType = Field(index=True)
    value: float


class Tank(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str


# ---------------------------------------------------------------------------
# Synthetic‑data generator
# ---------------------------------------------------------------------------

def logistic(t_h: np.ndarray,
            *,
            s0: float = 1.050,
            dsg: float = 0.035,
            k: float = 0.2,
            t0: float = 24) -> np.ndarray:
    """Logistic curve for specific gravity."""
    return s0 - dsg / (1 + np.exp(-k * (t_h - t0)))


def generate_synthetic_data(
    *,
    duration_h: int = 72,
    tank_id: int = 1,
    s0: float = 1.050,
    dsg: float = 0.035,
    k: float = 0.2,
    t0: float = 24,
    noise_scale: float = 1.0,
) -> None:
    """Populate SQLite with a single *duration_h*‑hour fermentation run.

    The *noise_scale* multiplier lets you dial random noise up or down: 0 ⇒ noiseless;
    1 ⇒ default; 2 ⇒ twice as noisy, etc.
    """
    start = datetime.now(tz=timezone.utc)
    total_min = duration_h * 60

    # -- 1‑minute grid ---------------------------------------------------
    idx_1min = pd.date_range(start, periods=total_min, freq="1min", tz=timezone.utc)
    minutes = np.arange(total_min)
    hours = minutes / 60

    # Temperature --------------------------------------------------------
    T0, dT = 18, 0.5
    step = (hours > 36).astype(float)  # cooling after 36 h
    temperature = T0 + dT * step + np.random.normal(0, 0.1 * noise_scale, total_min)

    # SG (5‑minute) ------------------------------------------------------
    idx_5min = pd.date_range(start, periods=total_min // 5, freq="5min", tz=timezone.utc)
    t_h_5 = np.arange(len(idx_5min)) * 5 / 60  # hours
    sg = logistic(t_h_5, s0=s0, dsg=dsg, k=k, t0=t0) + np.random.normal(
        0, 0.0005 * noise_scale, len(idx_5min)
    )

    # pH (hourly) --------------------------------------------------------
    idx_1h = pd.date_range(start, periods=duration_h, freq="1h", tz=timezone.utc)
    t_h_1h = np.arange(duration_h)
    pH0, alpha = 5.2, 0.01
    pH = pH0 - alpha * t_h_1h + np.random.normal(0, 0.02 * noise_scale, duration_h)
    # rare ±0.1 step every 12 h
    for h in range(12, duration_h, 12):
        pH[h:] += np.random.choice([-0.1, 0.1])

    # CO₂ & pressure (1‑minute) -----------------------------------------
    sg_1min = logistic(hours, s0=s0, dsg=dsg, k=k, t0=t0)
    dsg_dt = np.gradient(sg_1min) / (1 / 60)  # per hour → per minute
    COEFF_CO2 = 100  # ppm·h
    co2 = COEFF_CO2 * np.abs(dsg_dt) + np.random.normal(0, 5 * noise_scale, total_min)

    R = 0.08314  # bar·L / (mol·K)
    V = 1000  # fictive volume in L
    pressure = R * (temperature + 273.15) * co2 / V + np.random.normal(
        0, 0.02 * noise_scale, total_min
    )

    # Level (1‑minute) ---------------------------------------------------
    L0, beta = 1000, 1.5  # litres, L/h
    level = L0 - beta * hours + np.random.normal(0, 0.5 * noise_scale, total_min)

    # -------------------------------------------------------------------
    # Bulk‑insert into DB
    # -------------------------------------------------------------------

    def insert_array(ts_index: pd.DatetimeIndex, values: np.ndarray, stype: SensorType):
        with Session(engine) as session:
            session.add_all(
                [
                    Measurement(
                        timestamp=ts_index[i].to_pydatetime(),
                        tank_id=tank_id,
                        sensor_type=stype,
                        value=float(values[i]),
                    )
                    for i in range(len(ts_index))
                ]
            )
            session.commit()

    insert_array(idx_1min, temperature, SensorType.temperature)
    insert_array(idx_5min, sg, SensorType.sg)
    insert_array(idx_1h, pH, SensorType.ph)
    insert_array(idx_1min, co2, SensorType.co2)
    insert_array(idx_1min, pressure, SensorType.pressure)
    insert_array(idx_1min, level, SensorType.level)
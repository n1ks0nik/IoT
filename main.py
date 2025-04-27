from fastapi import FastAPI, Query, Depends
from sqlmodel import SQLModel, Field, create_engine, Session, select
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Database models & engine
# ---------------------------------------------------------------------------

DATABASE_URL = "sqlite:///fermentation.db"
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

def logistic(t_h, s0=1.050, dsg=0.04, k=0.2, t0=24):
    """Logistic curve for specific gravity."""
    return s0 - dsg / (1 + np.exp(-k * (t_h - t0)))


def generate_synthetic_data(duration_h: int = 72,
                            tank_id: int = 1,
                            *, s0: float = 1.050,
                            dsg: float = 0.04,
                            k: float = 0.2,
                            t0: int = 24,):
    """Populate SQLite with a single 72‑hour fermentation run."""

    start = datetime.now(tz=timezone.utc)
    total_min = duration_h * 60

    # -- 1‑minute grid -------------------------------------------------------
    idx_1min = pd.date_range(start, periods=total_min, freq="1min", tz=timezone.utc)
    minutes = np.arange(total_min)
    hours = minutes / 60

    # Temperature -----------------------------------------------------------
    T0, dT = 18, 0.5
    step = (hours > 36).astype(float)  # охлаждение после 36 ч
    temperature = T0 + dT * step + np.random.normal(0, 0.1, total_min)

    # SG (5‑minute) ----------------------------------------------------------
    idx_5min = pd.date_range(start, periods=total_min // 5, freq="5min", tz=timezone.utc)
    t_h_5 = np.arange(len(idx_5min)) * 5 / 60  # часы
    sg = logistic(t_h_5, s0=s0, dsg=dsg, k=k, t0=t0) + np.random.normal(0, 0.0005, len(idx_5min))

    # pH (hourly) ------------------------------------------------------------
    idx_1h = pd.date_range(start, periods=duration_h, freq="1h", tz=timezone.utc)
    t_h_1h = np.arange(duration_h)
    pH0, alpha = 5.2, 0.01
    pH = pH0 - alpha * t_h_1h + np.random.normal(0, 0.02, duration_h)
    # редкие скачки ±0.1 каждые 12 ч
    for h in range(12, duration_h, 12):
        pH[h:] += np.random.choice([-0.1, 0.1])

    # CO2 & pressure (1‑minute) --------------------------------------------
    sg_1min = logistic(hours, s0=s0, dsg=dsg, k=k, t0=t0)
    dsg_dt = np.gradient(sg_1min) / (1 / 60)  # per hour → per minute
    COEFF_CO2 = 100  # ppm·h
    co2 = COEFF_CO2 * np.abs(dsg_dt) + np.random.normal(0, 5, total_min)

    R = 0.08314  # bar·L / (mol·K)
    V = 1000  # L (фиктивный объем)
    pressure = R * (temperature + 273.15) * co2 / V + np.random.normal(0, 0.02, total_min)

    # Level (1‑minute) -------------------------------------------------------
    L0, beta = 1000, 1.5  # литров, л/ч
    level = L0 - beta * hours + np.random.normal(0, 0.5, total_min)

    # -----------------------------------------------------------------------
    # Bulk‑insert into DB
    # -----------------------------------------------------------------------

    def insert_array(ts_index: pd.DatetimeIndex, values: np.ndarray, stype: SensorType):
        with Session(engine) as session:
            session.add_all([
                Measurement(timestamp=ts_index[i].to_pydatetime(), tank_id=tank_id, sensor_type=stype, value=float(values[i]))
                for i in range(len(ts_index))
            ])
            session.commit()

    insert_array(idx_1min, temperature, SensorType.temperature)
    insert_array(idx_5min, sg, SensorType.sg)
    insert_array(idx_1h, pH, SensorType.ph)
    insert_array(idx_1min, co2, SensorType.co2)
    insert_array(idx_1min, pressure, SensorType.pressure)
    insert_array(idx_1min, level, SensorType.level)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Brew‑IoT prototype")


@app.on_event("startup")
def _on_startup():
    SQLModel.metadata.create_all(engine)

    # Создадим танк, если его нет
    with Session(engine) as session:
        tank = session.exec(select(Tank).where(Tank.id == 1)).first()
        if tank is None:
            session.add(Tank(id=1, name="Pilot tank #1"))
            session.commit()

    # Заполним БД единожды
    with Session(engine) as session:
        any_row = session.exec(select(Measurement)).first()
    if any_row is None:
        generate_synthetic_data()


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/data", response_model=List[Measurement])
def get_data(
    sensor_type: SensorType,
    from_ts: Optional[datetime] = Query(None),
    to_ts: Optional[datetime] = Query(None),
    limit: int = Query(1000, le=10000),
):
    """Return raw measurements filtered by sensor type & time range."""
    with Session(engine) as session:
        stmt = select(Measurement).where(Measurement.sensor_type == sensor_type)
        if from_ts:
            stmt = stmt.where(Measurement.timestamp >= from_ts)
        if to_ts:
            stmt = stmt.where(Measurement.timestamp <= to_ts)
        stmt = stmt.order_by(Measurement.timestamp.desc()).limit(limit)
        rows = session.exec(stmt).all()
        return list(reversed(rows))  # вернуть по возрастанию времени


@app.get("/")
def root():
    return {
        "message": "Brew‑IoT prototype is running. Hit /docs for the OpenAPI UI."
    }

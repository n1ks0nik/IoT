# main.py ─── только добавленные / изменённые строки
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles          # NEW
import asyncio                                        # NEW

from data_backend import (
    engine, SQLModel, Session, select,
    Tank, Measurement, SensorType,
    generate_synthetic_data,
)
from models_pipeline import ensure_n_runs, predict_abv

from typing import List, Dict

app = FastAPI(title="Brew-IoT prototype")

# отдаём фронтенд-страницу из папки static/
app.mount("/static", StaticFiles(directory="static"), name="static")   # NEW

last_gen_start: datetime = datetime.now(timezone.utc)
temp_msg_count: int    = 0

# ───────────────────────── 1. на старте заполняем БД и запускаем «реальное время»
@app.on_event("startup")
async def _on_startup() -> None:
    SQLModel.metadata.create_all(engine)

    # гарантируем, что есть хотя бы три танка со свежими данными
    #ensure_n_runs(n_runs=3, duration_h=72, noise_scale=0.5)

    # фоновый генератор – каждую минуту докидываем новые точки
    async def realtime_feed() -> None:
        global last_gen_start, temp_msg_count
        while True:
            last_gen_start = datetime.now(timezone.utc)
            temp_msg_count = 0
            generate_synthetic_data(duration_h=1, noise_scale=0.4)
            await asyncio.sleep(60)

    asyncio.create_task(realtime_feed())      # запускаем корутину

# ───────────────────────── 2. данные по сенсорам (как раньше)
@app.get("/api/data", response_model=list[Measurement])
def get_data(
    sensor_type: SensorType,
    tank_id: int = Query(1, ge=1),
    from_ts: datetime | None = Query(None),
    to_ts:   datetime | None = Query(None),
    limit: int = Query(1000, le=10000),
):
    with Session(engine) as session:
        stmt = (
            select(Measurement)
            .where(Measurement.tank_id == tank_id)
            .where(Measurement.sensor_type == sensor_type)
        )
        if from_ts:
            stmt = stmt.where(Measurement.timestamp >= from_ts)
        if to_ts:
            stmt = stmt.where(Measurement.timestamp <= to_ts)
        stmt = stmt.order_by(Measurement.timestamp.desc()).limit(limit)
        rows = session.exec(stmt).all()
    return list(reversed(rows))

# ───────────────────────── 3. прогноз ABV + детектор аномалий
@app.get("/api/abv")
def get_abv(tank_id: int = Query(1, ge=1)):
    abv, anomaly = predict_abv(tank_id=tank_id)
    return {"tank_id": tank_id, "abv": abv, "anomaly": anomaly}

# ───────────────────────── корневой роут отдаёт HTML
@app.get("/", include_in_schema=False)
def root():
    return {"msg": "Откройте /static/index.html для дашборда"}


@app.get("/api/tanks")
def get_tanks() -> List[Dict]:
    with Session(engine) as session:
        tanks = session.exec(select(Tank)).all()
    return [{"id": t.id, "name": t.name} for t in tanks]



from recommend import recommend_actions
import models_pipeline

@app.get("/api/recommend")
def get_recommendations(tank_id: int = 1, target: float = 4.6):
    abv, _ = predict_abv(tank_id=tank_id)
    # последние 36 ч измерений
    rows = Session(engine).exec(
        select(Measurement)
        .where(Measurement.tank_id == tank_id)
        .order_by(Measurement.timestamp)
    ).all()
    df = models_pipeline._pivot_measurements(rows)
    tips = recommend_actions(abv, target, df)
    return {"tank_id": tank_id, "pred_abv": abv, "target": target, "tips": tips}


from pydantic import BaseModel
from datetime import datetime, timezone
from data_backend import Session, engine, Measurement, SensorType

class SensorPayload(BaseModel):
    tank_id: int
    sensor_type: SensorType
    value: float
    timestamp: datetime | None = None

@app.post("/api/temp")
async def ingest_measurement(payload: SensorPayload):
    global last_gen_start, temp_msg_count
    ts = last_gen_start + timedelta(minutes=temp_msg_count)
    temp_msg_count = (temp_msg_count + 1) % 60
    with Session(engine) as session:
        m = Measurement(
            timestamp=ts,
            tank_id=payload.tank_id,
            sensor_type=payload.sensor_type,
            value=payload.value,
        )
        session.add(m)
        session.commit()
    return {"status": "ok"}
from fastapi import FastAPI, Query
from datetime import datetime
from typing import Optional, List

from data_backend import engine, SQLModel, Session, select, Tank, Measurement, SensorType, generate_synthetic_data

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

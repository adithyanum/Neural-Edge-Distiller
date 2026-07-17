from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import redis
import json
import psycopg2
from config import settings


app = FastAPI(title="Neural Edge Distiller — Control Plane")

redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)

def get_db_connection():
    return psycopg2.connect(
        host=settings.postgres_host,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password
    )


@app.get("/health")
def health():
    return {"status": "ok"}


class ExperimentCreate(BaseModel):
    name: str
    description: str


@app.post("/experiments")
def create_experiment(experiment: ExperimentCreate):
    experiment_id = str(uuid.uuid4())

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO experiments (id, name, description, status) VALUES (%s, %s, %s, %s)",
        (experiment_id, experiment.name, experiment.description, "queued")
    )
    conn.commit()
    cur.close()
    conn.close()

    job = {
        "id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "status": "queued"
    }

    redis_client.lpush("training_jobs", json.dumps(job))

    return job

@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, description, status, created_at, completed_at FROM experiments WHERE id = %s",
        (experiment_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "status": row[3],
        "created_at": row[4],
        "completed_at": row[5]
    }
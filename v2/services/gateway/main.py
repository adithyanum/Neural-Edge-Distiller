from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import redis
import json
import psycopg2

app = FastAPI(title="Neural Edge Distiller — Control Plane")

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        dbname="neural_edge",
        user="nova",
        password="devpassword"
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
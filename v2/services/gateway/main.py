from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import redis
import json

app = FastAPI(title="Neural Edge Distiller — Control Plane")

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)


@app.get("/health")
def health():
    return {"status": "ok"}


class ExperimentCreate(BaseModel):
    name: str
    description: str


@app.post("/experiments")
def create_experiment(experiment: ExperimentCreate):
    experiment_id = str(uuid.uuid4())

    job = {
        "id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "status": "queued"
    }

    redis_client.lpush("training_jobs", json.dumps(job))

    return job
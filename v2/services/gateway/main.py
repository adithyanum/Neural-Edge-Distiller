from fastapi import FastAPI
from pydantic import BaseModel
import uuid

app = FastAPI(title="Neural Edge Distiller — Control Plane")

@app.get("/health")
def health():
    return {"status": "ok"}


class ExperimentCreate(BaseModel):
    name: str
    description: str


@app.post("/experiments")
def create_experiment(experiment: ExperimentCreate):
    experiment_id = str(uuid.uuid4())
    return {
        "id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "status": "created"
    }
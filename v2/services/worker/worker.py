import redis
import json
import time
import ray
import psycopg2
import mlflow
from services.training import TrainingService
from services.status import ExperimentStatus
from config import settings

ray.init()

redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)


def get_db_connection():
    return psycopg2.connect(
        host=settings.postgres_host,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password
    )


def update_status(experiment_id, status, mark_completed=False):
    conn = get_db_connection()
    cur = conn.cursor()
    if mark_completed:
        cur.execute(
            "UPDATE experiments SET status = %s, completed_at = NOW() WHERE id = %s",
            (status, experiment_id)
        )
    else:
        cur.execute(
            "UPDATE experiments SET status = %s WHERE id = %s",
            (status, experiment_id)
        )
    conn.commit()
    cur.close()
    conn.close()

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

@ray.remote
def run_training_job(job):
    print(f"[TRAINING STARTED] {job['name']} ({job['id']})")
    update_status(job["id"], ExperimentStatus.TRAINING)

    training_service = TrainingService() 

    try:
        result = training_service.train(job)

        with mlflow.start_run(run_name=job["name"]):
            mlflow.log_param("experiment_id", job["id"])
            mlflow.log_metric("final_loss", result["final_loss"] or 0)
            if result.get("adapter_path"):
                mlflow.log_artifact(result["adapter_path"])

        update_status(job["id"], ExperimentStatus.COMPLETED, mark_completed=True)
        print(f"[TRAINING COMPLETE] {job['name']} ({job['id']})")
        return result

    except Exception as e:
        print(f"[TRAINING FAILED] {job['name']} ({job['id']}) — {e}")
        update_status(job["id"], ExperimentStatus.FAILED, mark_completed=True)
        return None


print("Worker started. Watching queue: training_jobs")

while True:
    _, raw_job = redis_client.brpop("training_jobs")
    job = json.loads(raw_job)
    print(f"[PICKED UP] {job['name']}")

    future = run_training_job.remote(job)
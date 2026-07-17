import redis
import json
import time
import ray
import psycopg2
from services.training import TrainingService
from services.status import ExperimentStatus
from config import settings

ray.init()

redis_client = redis.Redis(host=settings.redis_host, port=settings.redis_port, decode_responses=True)
training_service = TrainingService()


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


@ray.remote
def run_training_job(job):
    print(f"[TRAINING STARTED] {job['name']} ({job['id']})")
    update_status(job["id"], ExperimentStatus.TRAINING)

    try:
        result = training_service.train(job)
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
    ray.get(future)
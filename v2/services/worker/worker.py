import redis
import json
import time
import ray
import psycopg2

ray.init()

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)

def get_db_connection():
    return psycopg2.connect(
        host="postgres",
        dbname="neural_edge",
        user="nova",
        password="devpassword"
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
    update_status(job["id"], "training")

    time.sleep(5)  # placeholder — real training goes here later

    update_status(job["id"], "completed", mark_completed=True)
    print(f"[TRAINING COMPLETE] {job['name']} ({job['id']})")
    return job["id"]



print("Worker started. Watching queue: training_jobs")

while True:
    _, raw_job = redis_client.brpop("training_jobs")
    job = json.loads(raw_job)
    print(f"[PICKED UP] {job['name']}")

    future = run_training_job.remote(job)
    ray.get(future)
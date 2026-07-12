import redis
import json
import time
import ray

ray.init()

redis_client = redis.Redis(host="redis", port=6379, decode_responses=True)


@ray.remote
def run_training_job(job):
    print(f"[TRAINING STARTED] {job['name']} ({job['id']})")
    time.sleep(5)  # placeholder — real training goes here later
    print(f"[TRAINING COMPLETE] {job['name']} ({job['id']})")
    return job["id"]


print("Worker started. Watching queue: training_jobs")

while True:
    _, raw_job = redis_client.brpop("training_jobs")
    job = json.loads(raw_job)
    print(f"[PICKED UP] {job['name']}")

    future = run_training_job.remote(job)
    ray.get(future)
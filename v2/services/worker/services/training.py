import os
from config import settings

os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
os.environ["KAGGLE_KEY"] = settings.kaggle_key

import time
from services.training_backend import TrainingBackend
from services.kaggle_backend import KaggleBackend


class TrainingService:
    def __init__(self, backend: TrainingBackend = None):
        self.backend = backend or KaggleBackend(username=settings.kaggle_username)

    def train(self, job):
        print(f"[TrainingService] Submitting {job['name']} to backend")
        external_job_id = self.backend.submit(job)
        print(f"[TrainingService] Submitted — external id: {external_job_id}")

        while True:
            status = self.backend.status(external_job_id)
            print(f"[TrainingService] Status for {job['name']}: {status}")

            if status == "complete":
                return self.backend.download(external_job_id)
            elif status == "error":
                raise RuntimeError(f"Kaggle job {external_job_id} failed")

            time.sleep(15)
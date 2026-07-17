import time


class TrainingService:
    def train(self, job):
        print(f"[TrainingService] Starting training for {job['name']}")
        time.sleep(5)  # placeholder — real training (Kaggle) goes here later
        print(f"[TrainingService] Finished training for {job['name']}")
        return {"final_loss": None, "adapter_path": None}
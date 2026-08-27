import os
import json
import shutil
import tempfile
from kaggle.api.kaggle_api_extended import KaggleApi
from services.training_backend import TrainingBackend


# Path to the real training script, shipped alongside this file.
TRAIN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "kaggle_scripts", "train.py")

# Slug of the Kaggle Dataset containing v2/datasets/final/all.jsonl.
# Upload once via `kaggle datasets create` / update via `kaggle datasets version`
# whenever the final dataset changes - not re-uploaded on every submit().
DATASET_SLUG = os.environ.get("DISTILLER_DATASET_SLUG", "novaadi01/neural-edge-distiller-v2")

# The Kaggle API has no way to attach interactively-configured Secrets to a
# kernel pushed via the API (kernel_secrets is not a real metadata field -
# see https://github.com/Kaggle/kaggle-api/issues/582). Workaround: the HF
# token lives as the sole file in a small private Kaggle Dataset, attached
# like any other dataset_source, and train.py reads it from the mounted path.
HF_TOKEN_DATASET_SLUG = os.environ.get("HF_TOKEN_DATASET_SLUG", "novaadi01/hf-secrets")


class KaggleBackend(TrainingBackend):
    def __init__(self, username):
        self.api = KaggleApi()
        self.api.authenticate()
        self.username = username

    def submit(self, job) -> str:
        kernel_slug = f"distiller-train-{job['id'][:8]}"
        work_dir = tempfile.mkdtemp()

        shutil.copy(TRAIN_SCRIPT_PATH, os.path.join(work_dir, "train.py"))

        metadata = {
            "id": f"{self.username}/{kernel_slug}",
            "title": kernel_slug,
            "code_file": "train.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [DATASET_SLUG, HF_TOKEN_DATASET_SLUG],
        }

        ###debug


        print("\nChecking Kaggle datasets...")

        for dataset in [DATASET_SLUG, HF_TOKEN_DATASET_SLUG]:
            try:
                print(f"Checking: {dataset}")
                result = self.api.dataset_view(dataset)
                print(f"  FOUND: {result.ref}")
            except Exception as e:
                print(f"  FAILED: {dataset}")
                print(f"  ERROR: {type(e).__name__}: {e}")

        print("\n" + "=" * 60)
        print("KAGGLE DEBUG: KERNEL SUBMISSION")
        print("=" * 60)

        print(f"Kernel ID: {metadata['id']}")
        print(f"GPU enabled: {metadata['enable_gpu']}")
        print(f"Internet enabled: {metadata['enable_internet']}")

        print("\nDataset sources:")
        for dataset in metadata["dataset_sources"]:
            print(f"  -> {dataset}")

        print("\nMetadata:")
        print(json.dumps(metadata, indent=2))

        print(f"\nWork directory: {work_dir}")
        print(f"Files in work directory: {os.listdir(work_dir)}")

        print("=" * 60 + "\n")


        with open(os.path.join(work_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f)

        self.api.kernels_push(work_dir)
        return f"{self.username}/{kernel_slug}"

    def status(self, external_job_id: str) -> str:
        result = self.api.kernels_status(external_job_id)
        return result.status.name.lower()  # 'queued' | 'running' | 'complete' | 'error'

    def download(self, external_job_id: str) -> dict:
        output_dir = tempfile.mkdtemp()
        self.api.kernels_output(external_job_id, path=output_dir)

        with open(os.path.join(output_dir, "metrics.json")) as f:
            metrics = json.load(f)

        return {
            "final_loss": metrics.get("final_loss"),
            "adapter_path": os.path.join(output_dir, "adapter_output"),
        }
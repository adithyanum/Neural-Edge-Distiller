import os
import json
import shutil
import tempfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from azure.storage.blob import BlobServiceClient
from services.training_backend import TrainingBackend
from config import settings


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

# Adapters are pushed to Azure Blob Storage after every successful run, since
# serving will also live on Azure (single-cloud - avoids cross-cloud egress
# costs and a second set of credentials just for artifact storage). Requires
# AZURE_STORAGE_CONNECTION_STRING to be set in the worker's environment
# (Azure Portal -> Storage Account -> Access keys -> Connection string).
AZURE_BLOB_CONTAINER = os.environ.get("AZURE_BLOB_CONTAINER", "distiller-adapters")


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

        adapter_path = os.path.join(output_dir, "adapter_output")

        blob_prefix = self._upload_adapter_to_blob(adapter_path, external_job_id)

        return {
            "final_loss": metrics.get("final_loss"),
            "adapter_path": adapter_path,
            "adapter_blob_prefix": blob_prefix,
        }

    def _upload_adapter_to_blob(self, adapter_path: str, external_job_id: str) -> str | None:
        """
        Push the adapter to Azure Blob Storage so it's durably stored and
        accessible outside this worker's local disk/tempdir. Keyed by the
        Kaggle external_job_id so every run's adapter is kept, not
        overwritten - lets you compare adapters across runs later.

        Failure here doesn't fail the whole training job - a missing/mis-
        configured connection string shouldn't turn a successful training
        run into a reported failure. It's logged clearly instead so it's
        obviously visible in the worker logs, not silently swallowed.
        """
        conn_str = settings.azure_storage_connection_string
        if not conn_str:
            print("[KaggleBackend] azure_storage_connection_string not set - "
                  "skipping adapter upload, adapter only exists in local tempdir.")
            return None

        try:
            service_client = BlobServiceClient.from_connection_string(conn_str)
            container_client = service_client.get_container_client(AZURE_BLOB_CONTAINER)
            if not container_client.exists():
                container_client.create_container()

            slug = external_job_id.split("/")[-1]  # e.g. "distiller-train-8dc401a0"
            blob_prefix = f"adapters/{slug}"

            adapter_dir = Path(adapter_path)
            for f in adapter_dir.rglob("*"):
                if not f.is_file():
                    continue
                relative = f.relative_to(adapter_dir)
                blob_path = f"{blob_prefix}/{relative}"
                with open(f, "rb") as data:
                    container_client.upload_blob(name=blob_path, data=data, overwrite=True)

            print(f"[KaggleBackend] Adapter uploaded to Azure Blob: "
                  f"{AZURE_BLOB_CONTAINER}/{blob_prefix}")
            return blob_prefix

        except Exception as e:
            print(f"[KaggleBackend] Azure Blob upload failed (training result "
                  f"still valid, adapter just isn't backed up): {type(e).__name__}: {e}")
            return None
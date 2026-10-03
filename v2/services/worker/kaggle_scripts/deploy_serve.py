"""
Pushes app.py to Kaggle as a long-running script kernel and prints where to
watch the logs. Separate from KaggleBackend (services/worker/services/) on
purpose - that class's lifecycle assumes submit -> poll until complete/error
-> download results. This kernel never reaches "complete" while serving is
active, so it doesn't fit that state machine. This is a manual, one-off
deploy action, not part of the automated training pipeline.

Usage:
    python deploy_serve.py
"""

import json
import os
import tempfile
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "novaadi01")
APP_SCRIPT_PATH = Path(__file__).parent / "app.py"

# Same private-dataset secrets pattern as training. Add azure_conn.txt to
# this dataset (alongside the existing hf_token.txt) before running this.
SECRETS_DATASET_SLUG = f"{KAGGLE_USERNAME}/hf-secrets"


def main():
    api = KaggleApi()
    api.authenticate()

    kernel_slug = "distiller-serve"
    work_dir = tempfile.mkdtemp()
    shutil.copy(APP_SCRIPT_PATH, os.path.join(work_dir, "app.py"))

    metadata = {
        "id": f"{KAGGLE_USERNAME}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": "app.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [SECRETS_DATASET_SLUG],
    }
    with open(os.path.join(work_dir, "kernel-metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Pushing {kernel_slug} to Kaggle...")
    api.kernels_push(work_dir)

    print(f"\nPushed. Watch it start up here:")
    print(f"  https://www.kaggle.com/code/{KAGGLE_USERNAME}/{kernel_slug}")
    print(f"\nOnce running, click the Logs tab - the public URL will appear "
          f"there once vLLM + the tunnel are both up (takes a few minutes).")
    print(f"\nTo stop it later: go to the kernel page and click Stop, or it "
          f"will stop automatically at Kaggle's max session runtime.")


if __name__ == "__main__":
    main()
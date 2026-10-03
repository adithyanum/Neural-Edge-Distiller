"""
Neural Edge Distiller V2 - serving script (Kaggle + pycloudflared).

Same pattern as the spec-decode project: vLLM (instead of a custom FastAPI
wrapper - vLLM already serves an OpenAI-compatible FastAPI app on port 8000
out of the box) + pycloudflared for the tunnel, no auth token needed.

Runs as a Kaggle SCRIPT kernel (pushed via kernels_push, same mechanism as
train.py). Blocks indefinitely to keep the server + tunnel alive until the
kernel hits its max session runtime or is stopped manually.

Requires the same secrets dataset as train.py, plus one more file added to
it: azure_conn.txt (your Azure Storage connection string, plaintext).
"""

import os
import glob
import subprocess
import sys
import time
from pathlib import Path

print("Installing vLLM, pycloudflared, and Azure Blob Storage support...")
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "vllm", "pycloudflared", "azure-storage-blob",
    ],
    check=True,
)

# Kaggle may preinstall torchaudio for a different CUDA build. Transformers
# imports it while vLLM starts, even though this text-only server does not use audio.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

from pycloudflared import try_cloudflare

# ============================================================
# Load secrets (same private-dataset pattern as train.py)
# ============================================================

def load_secret(filename):
    paths = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not paths:
        raise RuntimeError(f"{filename} not found under /kaggle/input/ - is the secrets dataset attached?")
    return Path(paths[0]).read_text().strip()

HF_TOKEN = load_secret("hf_token.txt")
AZURE_CONN_STR = load_secret("azure_conn.txt")
os.environ["HF_TOKEN"] = HF_TOKEN

# ============================================================
# Pull adapter from Azure Blob
# ============================================================

from azure.storage.blob import BlobServiceClient

CONTAINER_NAME = "distiller-adapters"
BLOB_PREFIX = "adapters/distiller-train-3b93f081"
ADAPTER_DIR = Path("/kaggle/working/adapter")
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading adapter from Azure Blob...")
service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR, retry_total=6)
container_client = service_client.get_container_client(CONTAINER_NAME)
blobs = list(container_client.list_blobs(name_starts_with=BLOB_PREFIX))
for blob in blobs:
    relative_name = blob.name[len(BLOB_PREFIX):].lstrip("/")
    local_path = ADAPTER_DIR / relative_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob_client = container_client.get_blob_client(blob.name)
    with open(local_path, "wb") as f:
        f.write(blob_client.download_blob(timeout=600).readall())
    print(f"  -> {local_path}")
print("Adapter ready.\n")

# ============================================================
# launch vLLM in the background
# ============================================================

print("\nStarting vLLM server (takes a couple minutes to load the model)...")
vllm_process = subprocess.Popen(
    [
        "vllm", "serve", "meta-llama/Llama-3.2-3B-Instruct",
        "--enable-lora",
        "--lora-modules", f"distiller-v2={ADAPTER_DIR}",
        "--max-lora-rank", "16",
        "--dtype", "float16",
        "--max-model-len", "32768",
        "--port", "8000",
        "--host", "0.0.0.0",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

server_ready = False
for line in vllm_process.stdout:
    print("[vllm]", line, end="")
    if "Application startup complete." in line:
        server_ready = True
        print("\nvLLM is ready.\n")
        break

if not server_ready:
    exit_code = vllm_process.wait()
    raise RuntimeError(f"vLLM exited before becoming ready (exit code {exit_code})")

# ============================================================
# Open the tunnel - same call as the spec-decode setup, no auth token needed
# ============================================================

print("Opening Cloudflare tunnel...")
tunnel = try_cloudflare(port=8000)
public_url = tunnel.tunnel if hasattr(tunnel, "tunnel") else str(tunnel)

print("\n" + "=" * 60)
print(f"PUBLIC URL: {public_url}")
print("=" * 60)
print(f"\nTest with:\n  curl {public_url}/v1/completions -H 'Content-Type: application/json' "
      f"-d '{{\"model\": \"distiller-v2\", \"prompt\": \"Say hello\", \"max_tokens\": 50}}'\n")
print("NOTE: Cloudflare's free quick-tunnel has a 100s proxy timeout (same "
      "gotcha as the spec-decode project). Longer generations - long "
      "max_tokens or slow-loading requests - can exceed that and come back "
      "as an HTML error page instead of JSON. Check Content-Type before "
      "calling .json() on the client side, same fix as before.")

print("\nServer running. This kernel stays alive until Kaggle's max session "
      "runtime or manual stop.")
vllm_process.wait()
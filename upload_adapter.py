"""
One-off: upload the current local adapter to Azure Blob Storage.

Usage:
    pip install azure-storage-blob
    export AZURE_STORAGE_CONNECTION_STRING="<from Azure Portal - Storage Account -> Access keys>"
    python upload_adapter.py --adapter-path v2/models/adapter_output --blob-prefix adapters/llama-3.2-3b-lora-v1

After this, the adapter is at:
    https://<account>.blob.core.windows.net/<container>/<blob-prefix>/<file>
(private by default - see the container access-level note at the bottom for
making it publicly downloadable if that's actually what "anyone could access
it" means for you, vs. just "durably stored and reachable via the pipeline".)
"""

import argparse
import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


CONTAINER_NAME = "distiller-adapters"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--blob-prefix", required=True, help="e.g. adapters/llama-3.2-3b-lora-v1")
    args = parser.parse_args()

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING not set. Get it from the Azure "
            "Portal: Storage Account -> Access keys -> Connection string."
        )

    # retry_total: automatic retries on transient network failures (the
    # default client already retries some errors, but not aggressively
    # enough for larger files on slower connections - this raises the ceiling).
    service_client = BlobServiceClient.from_connection_string(
        conn_str, retry_total=6
    )
    container_client = service_client.get_container_client(CONTAINER_NAME)
    if not container_client.exists():
        print(f"Creating container: {CONTAINER_NAME}")
        container_client.create_container()

    adapter_dir = Path(args.adapter_path)
    files = list(adapter_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    print(f"Uploading {len(files)} files from {adapter_dir} to "
          f"{CONTAINER_NAME}/{args.blob_prefix}/ ...")

    failed = []
    for f in files:
        relative = f.relative_to(adapter_dir)
        blob_path = f"{args.blob_prefix}/{relative}"
        size_mb = f.stat().st_size / (1024 * 1024)
        try:
            with open(f, "rb") as data:
                # timeout is per-chunk connection timeout in seconds, not a
                # hard cap on total upload time - 600s per chunk is generous
                # headroom for a slow connection on a large adapter file.
                container_client.upload_blob(
                    name=blob_path, data=data, overwrite=True, timeout=600
                )
            print(f"  -> {blob_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: {blob_path} ({size_mb:.1f} MB) - {type(e).__name__}: {e}")
            failed.append(blob_path)

    if failed:
        print(f"\n{len(failed)} file(s) failed to upload: {failed}")
        print("Re-run the same command - overwrite=True means it's safe to "
              "retry, already-uploaded files just get re-uploaded (or you "
              "can comment those out if bandwidth is a concern).")
    else:
        print("\nDone, all files uploaded successfully.")
    print(f"Container: {CONTAINER_NAME}")
    print(f"Prefix: {args.blob_prefix}")
    print(
        "\nBy default this container is private (auth required). If you "
        "specifically want public/anonymous read access - e.g. for sharing a "
        "direct download link without credentials - set the container's "
        "access level to 'blob' in the Azure Portal (Container -> Change "
        "access level), or pass public_access='blob' to create_container() "
        "above. Keeping it private is the safer default; only change this "
        "if you have a real reason to want anonymous public access."
    )


if __name__ == "__main__":
    main()
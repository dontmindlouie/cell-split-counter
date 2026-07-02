"""Cloud entrypoint: download video from Blob, run pipeline, upload output."""

import os
import subprocess
import sys
from pathlib import Path

from azure.storage.blob import BlobServiceClient

STORAGE_CONN = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
INPUT_CONTAINER = os.environ.get("INPUT_CONTAINER", "input")
OUTPUT_CONTAINER = os.environ.get("OUTPUT_CONTAINER", "output")
VIDEO_BLOB = os.environ["VIDEO_BLOB"]
FRAME_STEP = os.environ.get("FRAME_STEP", "1")
TRACKER = os.environ.get("TRACKER", "trackastra")
END_FRAME = os.environ.get("END_FRAME")

client = BlobServiceClient.from_connection_string(STORAGE_CONN)

video_path = Path("/tmp") / VIDEO_BLOB
output_dir = Path("/tmp/output")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {VIDEO_BLOB} from blob storage...")
with open(video_path, "wb") as f:
    f.write(client.get_blob_client(INPUT_CONTAINER, VIDEO_BLOB).download_blob().readall())
print(f"Downloaded {video_path.stat().st_size / 1e6:.1f} MB")

cmd = [
    sys.executable, "main.py",
    str(video_path),
    "--frame-step", FRAME_STEP,
    "--tracker", TRACKER,
    "--output-dir", str(output_dir),
    "--frame-dir", str(output_dir / "frames"),
    "--debug-crops",
    "--classify-divisions",
]
if END_FRAME:
    cmd += ["--end-frame", END_FRAME]

result = subprocess.run(cmd, check=False)

print("Uploading output...")
out_container = client.get_container_client(OUTPUT_CONTAINER)
run_prefix = Path(VIDEO_BLOB).stem
for f in output_dir.rglob("*"):
    if f.is_file():
        blob_name = f"{run_prefix}/{f.relative_to(output_dir)}"
        with open(f, "rb") as data:
            out_container.upload_blob(blob_name, data, overwrite=True)
        print(f"  uploaded {blob_name}")

sys.exit(result.returncode)

"""Build and push the cloud pipeline image via ACR Tasks.

`az acr build` does NOT respect .dockerignore for its local context-packing step
(that file only applies later, inside the Docker build itself) -- it only applies a
small hardcoded set of default exclusions (.git, .venv, etc). Running it straight
from the repo root tars up the entire data/ directory (tens of GB and growing) before
upload, which looks hung for 20+ minutes with zero network activity rather than
erroring. This script sidesteps that by copying only the files the Dockerfile
actually needs into a clean temp directory and building from there.

Usage:
  python scripts/build_cloud_image.py <image-tag> [--registry NAME] [--resource-group NAME]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_FILES = ["Dockerfile", "requirements.txt", "main.py", "cloud_run.py"]
BUILD_DIRS = ["src"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag", help="image tag, e.g. v6")
    parser.add_argument("--registry", default="cellanalysisacr")
    parser.add_argument("--resource-group", default="cell-analysis")
    parser.add_argument("--image-name", default="cell-analysis")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="cell-analysis-build-") as tmp:
        ctx = Path(tmp)
        for name in BUILD_FILES:
            shutil.copy2(REPO_ROOT / name, ctx / name)
        for name in BUILD_DIRS:
            shutil.copytree(
                REPO_ROOT / name, ctx / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        size = sum(f.stat().st_size for f in ctx.rglob("*") if f.is_file())
        print(f"Build context: {ctx} ({size / 1024:.1f} KB, {sum(1 for _ in ctx.rglob('*') if _.is_file())} files)")

        image = f"{args.image_name}:{args.tag}"
        cmd = [
            "az", "acr", "build",
            "--registry", args.registry,
            "--resource-group", args.resource_group,
            "--image", image,
            str(ctx),
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

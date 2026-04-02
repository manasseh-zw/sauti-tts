#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Modal launcher for Sauti TTS data prep + training.

Usage:
    # Train with existing prepared dataset
    modal run scripts/modal_train.py --config configs/finetune_single_gpu.yaml

    # Prepare dataset then train (first run)
    modal run scripts/modal_train.py --config configs/finetune_single_gpu.yaml --prepare-data

This script persists data/checkpoints/outputs in Modal Volumes.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    import modal
except ImportError as exc:
    raise ImportError(
        "Modal SDK is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


APP_NAME = "sauti-tts"
PYTHON_VERSION = "3.11"
DEFAULT_GPU = "A100:2"

DATA_VOL_NAME = "sauti-tts-data"
CKPT_VOL_NAME = "sauti-tts-ckpts"
OUTPUT_VOL_NAME = "sauti-tts-outputs"

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_DATA_DIR = f"{REMOTE_PROJECT_ROOT}/data"
REMOTE_CKPT_DIR = f"{REMOTE_PROJECT_ROOT}/ckpts"
REMOTE_OUTPUT_DIR = f"{REMOTE_PROJECT_ROOT}/outputs"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path=REMOTE_PROJECT_ROOT)
)

volumes = {
    REMOTE_DATA_DIR: modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True),
    REMOTE_CKPT_DIR: modal.Volume.from_name(CKPT_VOL_NAME, create_if_missing=True),
    REMOTE_OUTPUT_DIR: modal.Volume.from_name(OUTPUT_VOL_NAME, create_if_missing=True),
}


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 24,
    volumes=volumes,
)
def run_pipeline(
    config: str = "configs/finetune_single_gpu.yaml",
    prepare_data: bool = False,
    sample_rate: int = 24000,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> None:
    """Run optional data preparation and then training on Modal."""
    os.chdir(REMOTE_PROJECT_ROOT)

    Path(REMOTE_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_CKPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if prepare_data:
        _run(
            [
                "python",
                "scripts/prepare_waxal_data.py",
                "--output_dir",
                "data/waxal_swahili",
                "--sample_rate",
                str(sample_rate),
                "--min_duration",
                str(min_duration),
                "--max_duration",
                str(max_duration),
            ]
        )

    _run(["python", "scripts/train.py", "--config", config])


@app.local_entrypoint()
def main(
    config: str = "configs/finetune_single_gpu.yaml",
    prepare_data: bool = False,
    sample_rate: int = 24000,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
):
    run_pipeline.remote(
        config=config,
        prepare_data=prepare_data,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
    )

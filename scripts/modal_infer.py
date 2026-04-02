#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Modal inference launcher for Sauti TTS.

Runs a single generation job against checkpoints and dataset assets stored in
Modal volumes, then saves the generated WAV into the outputs volume.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from pathlib import Path

try:
    import modal
except ImportError as exc:
    raise ImportError(
        "Modal SDK is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


APP_NAME = "sauti-tts-infer"
PYTHON_VERSION = "3.11"
GPU_TYPE = "A100"
DEFAULT_TIMEOUT_SECONDS = 60 * 60

F5_TTS_REPO = "https://github.com/SWivid/F5-TTS.git"
F5_TTS_REF = "main"
F5_TTS_DIR = "/opt/F5-TTS"

DATA_VOL_NAME = "sauti-tts-data"
CKPT_VOL_NAME = "sauti-tts-ckpts"
OUTPUT_VOL_NAME = "sauti-tts-outputs"

REMOTE_PROJECT_ROOT = "/root/project"
REMOTE_DATA_DIR = f"{REMOTE_PROJECT_ROOT}/data"
REMOTE_CKPT_DIR = f"{REMOTE_PROJECT_ROOT}/ckpts"
REMOTE_OUTPUT_DIR = f"{REMOTE_PROJECT_ROOT}/outputs"
REMOTE_HF_HOME = f"{REMOTE_DATA_DIR}/.hf"
REMOTE_TORCH_HOME = f"{REMOTE_DATA_DIR}/.torch"

DEFAULT_TEXT = "Habari, huu ni mfano wa sauti iliyotengenezwa na modeli yetu ya Kiswahili."

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
ckpt_volume = modal.Volume.from_name(CKPT_VOL_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOL_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        f"git clone --depth 1 --branch {F5_TTS_REF} {F5_TTS_REPO} {F5_TTS_DIR}",
        f"python -m pip install -e {F5_TTS_DIR}",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_PROJECT_ROOT,
        ignore=[
            ".venv",
            "data",
            "ckpts",
            "outputs",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "notebooks",
        ],
    )
)

volumes = {
    REMOTE_DATA_DIR: data_volume,
    REMOTE_CKPT_DIR: ckpt_volume,
    REMOTE_OUTPUT_DIR: output_volume,
}


def _commit_volumes() -> None:
    data_volume.commit()
    ckpt_volume.commit()
    output_volume.commit()


def _configure_runtime_environment() -> None:
    os.environ["HF_HOME"] = REMOTE_HF_HOME
    os.environ["HF_DATASETS_CACHE"] = f"{REMOTE_HF_HOME}/datasets"
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{REMOTE_HF_HOME}/hub"
    os.environ["TORCH_HOME"] = REMOTE_TORCH_HOME

    Path(REMOTE_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_CKPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_HF_HOME).mkdir(parents=True, exist_ok=True)
    Path(REMOTE_TORCH_HOME).mkdir(parents=True, exist_ok=True)


def _dataset_dir(dataset_name: str) -> Path:
    return Path(REMOTE_DATA_DIR) / dataset_name


def _checkpoint_path(checkpoint_subdir: str, checkpoint_name: str) -> Path:
    return Path(REMOTE_CKPT_DIR) / checkpoint_subdir / checkpoint_name


def _pick_reference_sample(
    dataset_name: str,
    split: str,
    index: int,
) -> tuple[str, str, str]:
    metadata_path = _dataset_dir(dataset_name) / f"{split}_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Reference metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))

    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    if index < 0 or index >= len(rows):
        raise IndexError(f"Reference index {index} is out of range for {split}")

    row = rows[index]
    return row["audio_path"], row["text"], row["id"]


def _copy_output_to_named_file(temp_output: Path, output_name: str) -> Path:
    output_path = Path(REMOTE_OUTPUT_DIR) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(temp_output, output_path)
    return output_path


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    volumes=volumes,
)
def generate_sample(
    text: str = DEFAULT_TEXT,
    dataset_name: str = "waxal_swahili",
    checkpoint_subdir: str = "sauti_tts_multi",
    checkpoint_name: str = "model_last.pt",
    reference_split: str = "train",
    reference_index: int = 0,
    output_name: str = "generated/test_sample.wav",
    steps: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    seed: int = 42,
    long_text: bool = False,
    max_chars: int = 180,
    cross_fade_sec: float = 0.25,
) -> dict[str, str]:
    os.chdir(REMOTE_PROJECT_ROOT)
    if REMOTE_PROJECT_ROOT not in sys.path:
        sys.path.insert(0, REMOTE_PROJECT_ROOT)
    _configure_runtime_environment()

    from scripts.inference import SautiInference

    dataset_dir = _dataset_dir(dataset_name)
    checkpoint_path = _checkpoint_path(checkpoint_subdir, checkpoint_name)
    vocab_path = dataset_dir / "vocab.txt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    ref_audio_path, ref_text, ref_id = _pick_reference_sample(
        dataset_name=dataset_name,
        split=reference_split,
        index=reference_index,
    )

    temp_output = Path(REMOTE_OUTPUT_DIR) / "_tmp_generated.wav"

    engine = SautiInference(
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(vocab_path),
    )
    effective_steps = max(steps, 48) if long_text else steps

    generation_kwargs = {
        "text": text,
        "ref_audio_path": ref_audio_path,
        "ref_text": ref_text,
        "output_path": str(temp_output),
        "nfe_steps": effective_steps,
        "cfg_strength": cfg_strength,
        "speed": speed,
        "seed": seed,
    }
    if long_text:
        engine.generate_long(
            max_chars_per_chunk=max_chars,
            cross_fade_sec=cross_fade_sec,
            **generation_kwargs,
        )
    else:
        engine.generate(**generation_kwargs)

    saved_output = _copy_output_to_named_file(temp_output, output_name)
    _commit_volumes()

    return {
        "checkpoint_path": str(checkpoint_path),
        "reference_audio": ref_audio_path,
        "reference_text": ref_text,
        "reference_id": ref_id,
        "output_path": str(saved_output),
        "text": text,
    }


@app.local_entrypoint()
def main(
    text: str = DEFAULT_TEXT,
    dataset_name: str = "waxal_swahili",
    checkpoint_subdir: str = "sauti_tts_multi",
    checkpoint_name: str = "model_last.pt",
    reference_split: str = "train",
    reference_index: int = 0,
    output_name: str = "generated/test_sample.wav",
    steps: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    seed: int = 42,
    long_text: bool = False,
    max_chars: int = 180,
    cross_fade_sec: float = 0.25,
) -> None:
    result = generate_sample.remote(
        text=text,
        dataset_name=dataset_name,
        checkpoint_subdir=checkpoint_subdir,
        checkpoint_name=checkpoint_name,
        reference_split=reference_split,
        reference_index=reference_index,
        output_name=output_name,
        steps=steps,
        cfg_strength=cfg_strength,
        speed=speed,
        seed=seed,
        long_text=long_text,
        max_chars=max_chars,
        cross_fade_sec=cross_fade_sec,
    )
    print(result, flush=True)

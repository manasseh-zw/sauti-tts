#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Modal launcher for single-speaker F5-TTS inference.

This wrapper reuses scripts.inference.SautiInference while preserving the
repo's shared Modal contract and defaulting to reference-audio selection from
the prepared single-speaker dataset cache.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError as exc:
    raise ImportError(
        "Modal SDK is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


APP_NAME = "sna-f5-tts-single-speaker-infer"
PYTHON_VERSION = "3.10"
GPU_TYPE = "L40S"
INFER_TIMEOUT_SECONDS = 60 * 60 * 2

DATA_VOLUME_NAME = "sna-data-vol"
MODEL_VOLUME_NAME = "sna-model-vol"

F5_TTS_REPO = "https://github.com/SWivid/F5-TTS.git"
F5_TTS_REF = "1.1.10"
F5_TTS_DIR = "/opt/F5-TTS"

REMOTE_WORKSPACE = "/workspace"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_MODEL_ROOT = Path("/model/sna-f5-tts/speakers")
PREP_SUBDIR = Path("f5_prepared/default")

DEFAULT_SPEAKER_ID = 1
DEFAULT_SPEAKER_GENDER = "Female"
DEFAULT_RUN_NAME = "default"
DEFAULT_CHECKPOINT_SUBPATH = "checkpoints/model_last.pt"
DEFAULT_SAMPLES_SUBDIR = "default"

DEFAULT_SENTENCES = [
    "Mangwanani, wakadini zvako nhasi?",
    "Tiri kuyedza modhi itsva yekutaura muShona ine inzwi rimwe chete.",
    "Kana izvi zvikafamba zvakanaka, tinobva tawedzera data uye toenzanisa mhando yezwi.",
]

IMAGE_IGNORE = [
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".DS_Store",
    "exp",
    "wandb",
    "data",
    "ckpts",
    "outputs",
    "*.pt",
    "*.ckpt",
    "*.hdf5",
    "*.log",
]

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

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
        f"git clone {F5_TTS_REPO} {F5_TTS_DIR}",
        f"cd {F5_TTS_DIR} && git checkout {F5_TTS_REF}",
        f"python -m pip install -e {F5_TTS_DIR}",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_WORKSPACE,
        ignore=IMAGE_IGNORE,
    )
)


def _speaker_dir_name(speaker_id: int, speaker_gender: str) -> str:
    return f"{speaker_id}_{speaker_gender}_full"


def _speaker_root(speaker_id: int, speaker_gender: str) -> Path:
    return REMOTE_DATA_ROOT / "speakers" / _speaker_dir_name(speaker_id, speaker_gender)


def _prepared_dataset_dir(speaker_id: int, speaker_gender: str) -> Path:
    return _speaker_root(speaker_id, speaker_gender) / PREP_SUBDIR


def _run_dir(speaker_id: int, speaker_gender: str, run_name: str) -> Path:
    return REMOTE_MODEL_ROOT / _speaker_dir_name(speaker_id, speaker_gender) / run_name


def _configure_runtime_environment() -> None:
    hf_home = REMOTE_DATA_ROOT / ".hf"
    torch_home = REMOTE_DATA_ROOT / ".torch"

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TORCH_HOME"] = str(torch_home)

    for path in (
        REMOTE_DATA_ROOT,
        REMOTE_MODEL_ROOT,
        hf_home,
        hf_home / "datasets",
        hf_home / "hub",
        torch_home,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _clean_text(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def _resolve_remote_text_file(text_file: str) -> Path:
    candidate = Path(text_file)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    workspace_candidate = Path(REMOTE_WORKSPACE) / text_file
    if workspace_candidate.exists():
        return workspace_candidate

    raise FileNotFoundError(f"Text file not found in remote environment: {text_file}")


def _load_texts(text: str, text_file: str) -> list[str]:
    if text_file:
        path = _resolve_remote_text_file(text_file)
        lines = [_clean_text(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]

    if text:
        return [_clean_text(text)]

    return DEFAULT_SENTENCES


def _load_prepared_rows(prep_dir: Path) -> list[dict[str, Any]]:
    metadata_path = prep_dir / "metadata.csv"
    duration_path = prep_dir / "duration.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Prepared metadata not found: {metadata_path}")
    if not duration_path.exists():
        raise FileNotFoundError(f"Prepared duration.json not found: {duration_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="|"))

    durations = json.loads(duration_path.read_text(encoding="utf-8")).get("duration", [])
    prepared_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        full_audio_path = (prep_dir / row["audio_file"]).resolve()
        if not full_audio_path.exists():
            continue
        duration = float(durations[index]) if index < len(durations) else None
        prepared_rows.append(
            {
                "index": index,
                "audio_path": str(full_audio_path),
                "audio_file": row["audio_file"],
                "text": _clean_text(row.get("text", "")),
                "duration": duration,
                "basename": full_audio_path.name,
            }
        )
    if not prepared_rows:
        raise RuntimeError(f"No valid prepared rows found in {metadata_path}")
    return prepared_rows


def _pick_reference_sample(prepared_rows: list[dict[str, Any]]) -> dict[str, Any]:
    preferred = [
        row
        for row in prepared_rows
        if row["text"] and row["duration"] is not None and 3.0 <= row["duration"] <= 15.0
    ]
    if preferred:
        return preferred[0]
    return prepared_rows[0]


def _resolve_checkpoint(
    *,
    run_dir: Path,
    checkpoint_path: str,
    checkpoint_subpath: str,
) -> Path:
    if checkpoint_path:
        candidate = Path(checkpoint_path)
        if candidate.is_absolute():
            resolved = candidate
        else:
            run_relative = run_dir / checkpoint_path
            resolved = run_relative if run_relative.exists() else candidate
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved

    resolved = run_dir / checkpoint_subpath
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved


def _resolve_reference_override(
    *,
    ref_audio: str,
    ref_text: str,
    speaker_root: Path,
    prepared_rows: list[dict[str, Any]],
) -> tuple[str, str, str]:
    if not ref_audio:
        chosen = _pick_reference_sample(prepared_rows)
        return chosen["audio_path"], chosen["text"], "prepared_dataset"

    candidate = Path(ref_audio)
    if candidate.is_absolute():
        resolved = candidate
    else:
        speaker_audio_candidate = speaker_root / "audio" / ref_audio
        if speaker_audio_candidate.exists():
            resolved = speaker_audio_candidate
        else:
            resolved = candidate

    if not resolved.exists():
        raise FileNotFoundError(f"Reference audio not found: {resolved}")

    effective_ref_text = _clean_text(ref_text)
    if not effective_ref_text:
        basename = resolved.name
        matched = next((row for row in prepared_rows if row["basename"] == basename), None)
        effective_ref_text = matched["text"] if matched else ""

    return str(resolved), effective_ref_text, "explicit_override"


def _write_manifest(sample_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = sample_dir / "manifest.csv"
    fieldnames = [
        "speaker_id",
        "speaker_gender",
        "run_name",
        "sample_index",
        "file_name",
        "text",
        "checkpoint_path",
        "ref_audio_path",
        "ref_text",
        "reference_source",
        "steps",
        "cfg_strength",
        "speed",
        "seed",
        "long_text",
        "output_path",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=INFER_TIMEOUT_SECONDS,
    volumes={"/data": data_volume, "/model": model_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def run_f5_single_speaker_infer(
    speaker_id: int = DEFAULT_SPEAKER_ID,
    speaker_gender: str = DEFAULT_SPEAKER_GENDER,
    run_name: str = DEFAULT_RUN_NAME,
    checkpoint_subpath: str = DEFAULT_CHECKPOINT_SUBPATH,
    checkpoint_path: str = "",
    text: str = "",
    text_file: str = "",
    ref_audio: str = "",
    ref_text: str = "",
    samples_subdir: str = DEFAULT_SAMPLES_SUBDIR,
    steps: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    seed: int = 42,
    long_text: bool = False,
    max_chars: int = 180,
    cross_fade_sec: float = 0.25,
) -> dict[str, Any]:
    os.chdir(REMOTE_WORKSPACE)
    if REMOTE_WORKSPACE not in sys.path:
        sys.path.insert(0, REMOTE_WORKSPACE)

    _configure_runtime_environment()

    speaker_root = _speaker_root(speaker_id, speaker_gender)
    prep_dir = _prepared_dataset_dir(speaker_id, speaker_gender)
    run_dir = _run_dir(speaker_id, speaker_gender, run_name)
    sample_dir = run_dir / "samples" / samples_subdir
    sample_dir.mkdir(parents=True, exist_ok=True)

    prepared_rows = _load_prepared_rows(prep_dir)
    resolved_checkpoint = _resolve_checkpoint(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_subpath=checkpoint_subpath,
    )
    vocab_path = prep_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Prepared vocab not found: {vocab_path}")

    effective_ref_audio, effective_ref_text, reference_source = _resolve_reference_override(
        ref_audio=ref_audio,
        ref_text=ref_text,
        speaker_root=speaker_root,
        prepared_rows=prepared_rows,
    )

    import scripts.inference as inference_module

    inference_module.normalize_swahili_text = _clean_text
    engine = inference_module.SautiInference(
        checkpoint_path=str(resolved_checkpoint),
        vocab_path=str(vocab_path),
    )

    texts = _load_texts(text=text, text_file=text_file)
    manifest_rows: list[dict[str, Any]] = []

    for index, item in enumerate(texts, start=1):
        cleaned_text = _clean_text(item)
        file_name = f"sample_{index:03d}.wav"
        output_path = sample_dir / file_name

        generation_kwargs = {
            "text": cleaned_text,
            "ref_audio_path": effective_ref_audio,
            "ref_text": effective_ref_text,
            "output_path": str(output_path),
            "nfe_steps": steps,
            "cfg_strength": cfg_strength,
            "speed": speed,
            "seed": seed + index - 1,
        }
        if long_text:
            engine.generate_long(
                max_chars_per_chunk=max_chars,
                cross_fade_sec=cross_fade_sec,
                **generation_kwargs,
            )
        else:
            engine.generate(**generation_kwargs)

        manifest_rows.append(
            {
                "speaker_id": speaker_id,
                "speaker_gender": speaker_gender,
                "run_name": run_name,
                "sample_index": index,
                "file_name": file_name,
                "text": cleaned_text,
                "checkpoint_path": str(resolved_checkpoint),
                "ref_audio_path": effective_ref_audio,
                "ref_text": effective_ref_text,
                "reference_source": reference_source,
                "steps": steps,
                "cfg_strength": cfg_strength,
                "speed": speed,
                "seed": seed + index - 1,
                "long_text": long_text,
                "output_path": str(output_path),
            }
        )

    _write_manifest(sample_dir, manifest_rows)
    model_volume.commit()

    return {
        "speaker_id": speaker_id,
        "speaker_gender": speaker_gender,
        "run_name": run_name,
        "checkpoint_path": str(resolved_checkpoint),
        "reference_audio_path": effective_ref_audio,
        "reference_source": reference_source,
        "sample_dir": str(sample_dir),
        "num_samples": len(manifest_rows),
    }


@app.local_entrypoint()
def main(
    speaker_id: int = DEFAULT_SPEAKER_ID,
    speaker_gender: str = DEFAULT_SPEAKER_GENDER,
    run_name: str = DEFAULT_RUN_NAME,
    checkpoint_subpath: str = DEFAULT_CHECKPOINT_SUBPATH,
    checkpoint_path: str = "",
    text: str = "",
    text_file: str = "",
    ref_audio: str = "",
    ref_text: str = "",
    samples_subdir: str = DEFAULT_SAMPLES_SUBDIR,
    steps: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    seed: int = 42,
    long_text: bool = False,
    max_chars: int = 180,
    cross_fade_sec: float = 0.25,
):
    run_f5_single_speaker_infer.remote(
        speaker_id=speaker_id,
        speaker_gender=speaker_gender,
        run_name=run_name,
        checkpoint_subpath=checkpoint_subpath,
        checkpoint_path=checkpoint_path,
        text=text,
        text_file=text_file,
        ref_audio=ref_audio,
        ref_text=ref_text,
        samples_subdir=samples_subdir,
        steps=steps,
        cfg_strength=cfg_strength,
        speed=speed,
        seed=seed,
        long_text=long_text,
        max_chars=max_chars,
        cross_fade_sec=cross_fade_sec,
    )

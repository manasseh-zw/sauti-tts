#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Modal launcher for single-speaker F5-TTS fine-tuning.

This wrapper keeps the repo's shared Modal contract:
- data volume mounted at /data from sna-data-vol
- model volume mounted at /model from sna-model-vol
- dotenv secrets injected into the remote function

It converts a speaker export into the F5-TTS training format, writes a
run-local config, and then launches the existing scripts/train.py entrypoint.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError as exc:
    raise ImportError(
        "Modal SDK is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


APP_NAME = "sna-f5-tts-single-speaker"
PYTHON_VERSION = "3.10"
GPU_TYPE = "A10G"
TRAIN_TIMEOUT_SECONDS = 60 * 60 * 24

DATA_VOLUME_NAME = "sna-data-vol"
MODEL_VOLUME_NAME = "sna-model-vol"

F5_TTS_REPO = "https://github.com/SWivid/F5-TTS.git"
# Pinned for reproducible Modal builds. Update deliberately if upstream changes
# are required for future training runs.
F5_TTS_REF = "1.1.10"
F5_TTS_DIR = "/opt/F5-TTS"

REMOTE_WORKSPACE = "/workspace"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_MODEL_ROOT = Path("/model/sna-f5-tts/speakers")
PREP_SUBDIR = Path("f5_prepared/default")

DEFAULT_SPEAKER_ID = 1
DEFAULT_SPEAKER_GENDER = "Female"
DEFAULT_RUN_NAME = "default"
DEFAULT_WANDB_PROJECT = "sna-f5-tts"
DEFAULT_BASE_CONFIG = "configs/finetune_lora.yaml"
DEFAULT_CUSTOM_DATASET_SUBDIR = ""
DEFAULT_CUSTOM_METADATA_FILENAME = "metadata.csv"
DEFAULT_CUSTOM_AUDIO_SUBDIR = ""
DEFAULT_CUSTOM_AUDIO_COLUMN = "filename"
DEFAULT_CUSTOM_TEXT_COLUMN = "text"
DEFAULT_CUSTOM_DURATION_COLUMN = ""
DEFAULT_BASE_MODEL_RUN = ""
DEFAULT_BASE_CHECKPOINT_SUBPATH = "checkpoints/model_last.pt"
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.0
DEFAULT_SAMPLE_EVERY_UPDATES = 500
DEFAULT_SAMPLE_EVERY_EPOCHS = 2
DEFAULT_SAMPLE_STEPS = 32
MIN_F5_DURATION_SEC = 0.3
MAX_F5_DURATION_SEC = 30.0
SAMPLE_RATE = 24000
HOP_LENGTH = 256

DEFAULT_SAMPLE_SENTENCES = [
    "Mangwanani. Ndamuka zvakanaka nhasi, ndirikugadzirira kuenda kusvondo.",
    "Mhoro shamwari yangu. Wakafamba sei nezuro, wakasvika zvakanaka here",
    "Waswera sei, ndanga ndichifona kuti ndibvunze kuti tinogona kusangana here nhasi.",
]

PRETRAINED_VOCAB_FILENAMES = (
    "F5TTS_v1_Base/vocab.txt",
    "data/Emilia_ZH_EN_pinyin/vocab.txt",
)

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _speaker_dir_name(speaker_id: int, speaker_gender: str) -> str:
    return f"{speaker_id}_{speaker_gender}_full"


def _speaker_root(speaker_id: int, speaker_gender: str) -> Path:
    return REMOTE_DATA_ROOT / "speakers" / _speaker_dir_name(speaker_id, speaker_gender)


def _dataset_root(
    speaker_id: int,
    speaker_gender: str,
    custom_dataset_subdir: str,
) -> Path:
    if custom_dataset_subdir.strip():
        return REMOTE_DATA_ROOT / custom_dataset_subdir.strip("/ ")
    return _speaker_root(speaker_id, speaker_gender)


def _prepared_dataset_dir(
    speaker_id: int,
    speaker_gender: str,
    custom_dataset_subdir: str,
) -> Path:
    return _dataset_root(speaker_id, speaker_gender, custom_dataset_subdir) / PREP_SUBDIR


def _run_dir(speaker_id: int, speaker_gender: str, run_name: str) -> Path:
    return REMOTE_MODEL_ROOT / _speaker_dir_name(speaker_id, speaker_gender) / run_name


def _resolve_base_checkpoint_path(
    *,
    base_model_run: str,
    base_checkpoint_subpath: str,
    base_checkpoint_path: str,
) -> Path | None:
    if base_checkpoint_path.strip():
        return Path(base_checkpoint_path.strip())
    if base_model_run.strip():
        return (
            Path("/model/sna-f5-tts/shona-base")
            / base_model_run.strip()
            / base_checkpoint_subpath.strip()
        )
    return None


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


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _log_prep_summary(summary: dict[str, Any]) -> None:
    print(
        "\n".join(
            [
                "Prepared dataset summary:",
                f"  rows_written={summary.get('rows_written', '?')}",
                f"  too_long_duration_rows={summary.get('too_long_duration_rows', 0)}",
                f"  too_short_duration_rows={summary.get('too_short_duration_rows', 0)}",
                f"  empty_transcript_rows={summary.get('empty_transcript_rows', 0)}",
                f"  missing_audio_rows={summary.get('missing_audio_rows', 0)}",
                f"  blank_filename_rows={summary.get('blank_filename_rows', 0)}",
                f"  prepared_dataset_dir={summary.get('prepared_dataset_dir', '')}",
            ]
        ),
        flush=True,
    )


def _clean_transcript(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split())


def _ensure_vocab_file(prep_dir: Path) -> Path:
    vocab_path = prep_dir / "vocab.txt"
    if vocab_path.exists():
        return vocab_path

    from huggingface_hub import hf_hub_download

    local_candidates = [
        Path(F5_TTS_DIR) / "data" / "Emilia_ZH_EN_pinyin" / "vocab.txt",
        Path(F5_TTS_DIR) / "ckpts" / "F5TTS_v1_Base" / "vocab.txt",
        Path(F5_TTS_DIR) / "F5TTS_v1_Base" / "vocab.txt",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            shutil.copy2(candidate, vocab_path)
            return vocab_path

    last_error: Exception | None = None
    for filename in PRETRAINED_VOCAB_FILENAMES:
        try:
            downloaded = hf_hub_download(
                repo_id="SWivid/F5-TTS",
                filename=filename,
            )
            shutil.copy2(downloaded, vocab_path)
            return vocab_path
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not locate pretrained F5 vocab.txt in the cloned repo or on Hugging Face."
    ) from last_error


def _resolve_audio_path(dataset_root: Path, audio_root: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path).strip())
    if candidate.is_absolute() and candidate.exists():
        return candidate

    rooted = dataset_root / candidate
    if rooted.exists():
        return rooted

    audio_rooted = audio_root / candidate
    if audio_rooted.exists():
        return audio_rooted

    basename = audio_root / candidate.name
    if basename.exists():
        return basename

    return audio_rooted


def _validate_dataset_layout(
    *,
    speaker_id: int,
    speaker_gender: str,
    custom_dataset_subdir: str,
    metadata_filename: str,
    custom_audio_subdir: str,
) -> tuple[Path, Path, Path]:
    dataset_root = _dataset_root(speaker_id, speaker_gender, custom_dataset_subdir)
    metadata_path = dataset_root / metadata_filename
    audio_root = dataset_root / custom_audio_subdir.strip("/ ") if custom_audio_subdir.strip() else dataset_root

    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    if not audio_root.exists():
        raise FileNotFoundError(f"Dataset audio directory not found: {audio_root}")

    return dataset_root, metadata_path, audio_root


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
                "text": _clean_transcript(row.get("text", "")),
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


def _prepare_dataset(
    speaker_id: int,
    speaker_gender: str,
    custom_dataset_subdir: str,
    metadata_filename: str,
    custom_audio_subdir: str,
    custom_audio_column: str,
    custom_text_column: str,
    custom_duration_column: str,
    force_rebuild_prep: bool,
) -> dict[str, Any]:
    from sauti_tts.data import prepare_f5tts_format

    dataset_root, metadata_path, audio_root = _validate_dataset_layout(
        speaker_id=speaker_id,
        speaker_gender=speaker_gender,
        custom_dataset_subdir=custom_dataset_subdir,
        metadata_filename=metadata_filename,
        custom_audio_subdir=custom_audio_subdir,
    )
    prep_dir = _prepared_dataset_dir(speaker_id, speaker_gender, custom_dataset_subdir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = prep_dir / "metadata.csv"
    prep_summary_path = prep_dir / "prep_summary.json"

    if force_rebuild_prep:
        for stale_name in ("metadata.csv", "raw.arrow", "duration.json", "vocab.txt", "prep_summary.json"):
            stale_path = prep_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()

    audio_column = custom_audio_column.strip() or "file_name"
    text_column = custom_text_column.strip() or "transcription"
    duration_column = custom_duration_column.strip()

    required_columns = {
        audio_column,
        text_column,
    }
    if duration_column:
        required_columns.add(duration_column)

    kept_rows: list[dict[str, str]] = []
    missing_audio = 0
    empty_transcript = 0
    blank_filename = 0
    too_short_duration = 0
    too_long_duration = 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            raise ValueError(
                f"Speaker export metadata is missing required columns: {missing_columns}"
            )

        for row in reader:
            file_name = str(row.get(audio_column, "")).strip()
            if not file_name:
                blank_filename += 1
                continue

            transcript = _clean_transcript(row.get(text_column, ""))
            if not transcript:
                empty_transcript += 1
                continue

            duration_value = row.get(duration_column, "") if duration_column else ""
            if duration_value not in (None, ""):
                try:
                    duration_sec = float(duration_value)
                    if duration_sec < MIN_F5_DURATION_SEC:
                        too_short_duration += 1
                        continue
                    if duration_sec > MAX_F5_DURATION_SEC:
                        too_long_duration += 1
                        continue
                except ValueError:
                    pass

            audio_path = _resolve_audio_path(dataset_root, audio_root, file_name)
            if not audio_path.exists():
                missing_audio += 1
                continue

            kept_rows.append(
                {
                    "audio_file": os.path.relpath(audio_path, prep_dir),
                    "text": transcript,
                    "file_name": file_name,
                    "duration": str(duration_value).strip(),
                }
            )

    if not kept_rows:
        raise RuntimeError(
            f"No valid clips were found in {metadata_path}. "
            "Check transcript content and audio filenames."
        )

    with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "duration"])
        for row in kept_rows:
            writer.writerow([row["audio_file"], row["text"], row["duration"]])

    vocab_path = _ensure_vocab_file(prep_dir)
    prep_vocab_path = prep_dir / "vocab.txt"
    vocab_arg = str(vocab_path)
    try:
        if Path(vocab_path).resolve() == prep_vocab_path.resolve():
            vocab_arg = None
    except FileNotFoundError:
        pass

    prepare_f5tts_format(
        metadata_csv=str(metadata_csv),
        output_dir=str(prep_dir),
        vocab_path=vocab_arg,
    )

    summary = {
        "speaker_id": speaker_id,
        "speaker_gender": speaker_gender,
        "dataset_root": str(dataset_root),
        "prepared_dataset_dir": str(prep_dir),
        "metadata_source": str(metadata_path),
        "audio_root": str(audio_root),
        "audio_column": audio_column,
        "text_column": text_column,
        "duration_column": duration_column or None,
        "rows_written": len(kept_rows),
        "empty_transcript_rows": empty_transcript,
        "missing_audio_rows": missing_audio,
        "blank_filename_rows": blank_filename,
        "too_short_duration_rows": too_short_duration,
        "too_long_duration_rows": too_long_duration,
        "forced_rebuild": force_rebuild_prep,
        "created_at": _utc_now(),
        "sample_rows": kept_rows[:5],
    }
    _write_json(prep_summary_path, summary)
    return summary


def _validate_prepared_dataset(prep_dir: Path) -> dict[str, Any]:
    required = [
        prep_dir / "metadata.csv",
        prep_dir / "raw.arrow",
        prep_dir / "duration.json",
        prep_dir / "vocab.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared dataset is incomplete. Missing: " + ", ".join(missing)
        )

    prep_summary_path = prep_dir / "prep_summary.json"
    if prep_summary_path.exists():
        return json.loads(prep_summary_path.read_text(encoding="utf-8"))

    return {
        "prepared_dataset_dir": str(prep_dir),
        "validated_at": _utc_now(),
    }


def _prepared_dataset_exists(prep_dir: Path) -> bool:
    required = [
        prep_dir / "metadata.csv",
        prep_dir / "raw.arrow",
        prep_dir / "duration.json",
        prep_dir / "vocab.txt",
        prep_dir / "prep_summary.json",
    ]
    return all(path.exists() for path in required)


def _recommended_batch_frames(prep_dir: Path, base_frames: int) -> int:
    duration_path = prep_dir / "duration.json"
    if not duration_path.exists():
        return base_frames

    payload = json.loads(duration_path.read_text(encoding="utf-8"))
    durations = payload.get("duration", [])
    valid_durations = [
        float(duration)
        for duration in durations
        if MIN_F5_DURATION_SEC <= float(duration) <= MAX_F5_DURATION_SEC
    ]
    if not valid_durations:
        return base_frames

    max_duration = max(valid_durations)
    required_frames = math.ceil(max_duration * SAMPLE_RATE / HOP_LENGTH)
    return max(base_frames, required_frames)


def _estimate_updates_per_epoch(
    prep_dir: Path,
    batch_size_per_gpu: int,
    max_samples: int,
) -> int:
    duration_path = prep_dir / "duration.json"
    if not duration_path.exists():
        return 0

    payload = json.loads(duration_path.read_text(encoding="utf-8"))
    durations = payload.get("duration", [])
    frame_lengths = sorted(
        (
            math.ceil(float(duration) * SAMPLE_RATE / HOP_LENGTH)
            for duration in durations
            if MIN_F5_DURATION_SEC <= float(duration) <= MAX_F5_DURATION_SEC
        ),
        reverse=True,
    )
    if not frame_lengths:
        return 0

    batches = 0
    current_frames = 0
    current_count = 0
    for frame_length in frame_lengths:
        if current_count > 0 and (
            current_frames + frame_length > batch_size_per_gpu
            or current_count + 1 > max_samples
        ):
            batches += 1
            current_frames = 0
            current_count = 0

        current_frames += frame_length
        current_count += 1

    if current_count > 0:
        batches += 1

    return batches


def _write_generated_config(
    *,
    speaker_id: int,
    speaker_gender: str,
    run_name: str,
    prep_dir: Path,
    run_dir: Path,
    wandb_project: str,
    wandb_run_name: str | None,
    base_config: str,
    base_checkpoint_path: Path | None,
    epochs: int,
    learning_rate: float,
    sample_every_epochs: int,
    save_per_updates: int,
) -> tuple[Path, Path, str, str, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = Path(REMOTE_WORKSPACE) / base_config
    config = _load_yaml(base_config_path)

    speaker_dir = _speaker_dir_name(speaker_id, speaker_gender)
    effective_wandb_run_name = wandb_run_name or f"{speaker_dir}-{run_name}"
    logger_type = "wandb" if os.getenv("WANDB_API_KEY") else "tensorboard"

    config["exp_name"] = f"sna_f5_tts_{speaker_dir}_{run_name}"
    config["output_dir"] = str(checkpoints_dir)

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["name"] = speaker_dir
    dataset_cfg["dir"] = str(prep_dir)
    dataset_cfg["metadata"] = "metadata.csv"
    dataset_cfg["sample_rate"] = 24000

    model_cfg = config.setdefault("model", {})
    model_cfg["vocab_path"] = str(prep_dir / "vocab.txt")
    model_cfg["use_lora"] = True
    if base_checkpoint_path is not None:
        model_cfg["pretrained_ckpt"] = str(base_checkpoint_path)

    training_cfg = config.setdefault("training", {})
    training_cfg["epochs"] = epochs
    if learning_rate > 0:
        training_cfg["learning_rate"] = learning_rate
    training_cfg["batch_size_per_gpu"] = _recommended_batch_frames(
        prep_dir,
        int(training_cfg.get("batch_size_per_gpu", 800)),
    )
    max_samples = int(training_cfg.get("max_samples", 32))
    estimated_updates_per_epoch = _estimate_updates_per_epoch(
        prep_dir=prep_dir,
        batch_size_per_gpu=int(training_cfg["batch_size_per_gpu"]),
        max_samples=max_samples,
    )

    logging_cfg = config.setdefault("logging", {})
    logging_cfg["type"] = logger_type
    logging_cfg["wandb_project"] = wandb_project
    logging_cfg["wandb_run_name"] = effective_wandb_run_name

    checkpoint_cfg = config.setdefault("checkpoint", {})
    effective_save_per_updates = save_per_updates
    if sample_every_epochs > 0 and estimated_updates_per_epoch > 0:
        effective_save_per_updates = max(
            estimated_updates_per_epoch * sample_every_epochs,
            1,
        )

    checkpoint_cfg["save_per_updates"] = effective_save_per_updates
    checkpoint_cfg["last_per_steps"] = max(
        int(checkpoint_cfg.get("last_per_steps", effective_save_per_updates)),
        effective_save_per_updates,
    )

    generated_config_path = run_dir / "config.generated.yaml"
    _dump_yaml(generated_config_path, config)
    return (
        generated_config_path,
        checkpoints_dir,
        logger_type,
        effective_wandb_run_name,
        estimated_updates_per_epoch,
    )


def _finish_wandb_if_active() -> None:
    try:
        import wandb

        if getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception:
        pass


def _commit_volumes() -> None:
    data_volume.commit()
    model_volume.commit()


def _parse_checkpoint_step(path: Path) -> int | None:
    stem = path.stem
    if stem == "model_last":
        return None
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else None


def _iter_saved_checkpoints(checkpoints_dir: Path) -> list[Path]:
    candidates = list(checkpoints_dir.glob("model_[0-9]*.pt"))
    candidates.extend(checkpoints_dir.glob("model_[0-9]*.safetensors"))
    return sorted(
        candidates,
        key=lambda path: (_parse_checkpoint_step(path) or -1, path.name),
    )


def _wait_for_stable_file(path: Path, max_wait_seconds: int = 120) -> None:
    deadline = time.time() + max_wait_seconds
    previous_size = -1
    while time.time() < deadline:
        if not path.exists():
            time.sleep(2)
            continue
        current_size = path.stat().st_size
        if current_size > 0 and current_size == previous_size:
            return
        previous_size = current_size
        time.sleep(2)
    raise TimeoutError(f"Checkpoint file did not stabilize in time: {path}")


def _write_sample_manifest(sample_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = sample_dir / "manifest.csv"
    fieldnames = [
        "speaker_id",
        "speaker_gender",
        "run_name",
        "checkpoint_label",
        "checkpoint_path",
        "sentence_index",
        "file_name",
        "text",
        "reference_audio_path",
        "reference_text",
        "steps",
        "cfg_strength",
        "speed",
        "seed",
        "output_path",
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=60 * 60 * 2,
    volumes={"/data": data_volume, "/model": model_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def generate_checkpoint_samples(
    speaker_id: int,
    speaker_gender: str,
    run_name: str,
    custom_dataset_subdir: str,
    checkpoint_path: str,
    checkpoint_label: str,
    wandb_project: str,
    wandb_group: str,
    sample_sentences: list[str],
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    os.chdir(REMOTE_WORKSPACE)
    if REMOTE_WORKSPACE not in sys.path:
        sys.path.insert(0, REMOTE_WORKSPACE)

    _configure_runtime_environment()

    import scripts.inference as inference_module

    inference_module.normalize_swahili_text = _clean_transcript

    prep_dir = _prepared_dataset_dir(
        speaker_id,
        speaker_gender,
        custom_dataset_subdir,
    )
    run_dir = _run_dir(speaker_id, speaker_gender, run_name)
    sample_dir = run_dir / "checkpoint_samples" / checkpoint_label
    sample_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint = Path(checkpoint_path)
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found for sampling: {resolved_checkpoint}")

    prepared_rows = _load_prepared_rows(prep_dir)
    reference = _pick_reference_sample(prepared_rows)
    vocab_path = prep_dir / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Prepared vocab not found: {vocab_path}")

    engine = inference_module.SautiInference(
        checkpoint_path=str(resolved_checkpoint),
        vocab_path=str(vocab_path),
    )

    manifest_rows: list[dict[str, Any]] = []
    audio_logs: dict[str, Any] = {}

    for index, sentence in enumerate(sample_sentences, start=1):
        cleaned_sentence = _clean_transcript(sentence)
        file_name = f"sample_{index:02d}.wav"
        output_path = sample_dir / file_name
        effective_seed = seed + index - 1

        engine.generate(
            text=cleaned_sentence,
            ref_audio_path=reference["audio_path"],
            ref_text=reference["text"],
            output_path=str(output_path),
            nfe_steps=sample_steps,
            cfg_strength=cfg_strength,
            speed=speed,
            seed=effective_seed,
        )

        manifest_rows.append(
            {
                "speaker_id": speaker_id,
                "speaker_gender": speaker_gender,
                "run_name": run_name,
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(resolved_checkpoint),
                "sentence_index": index,
                "file_name": file_name,
                "text": cleaned_sentence,
                "reference_audio_path": reference["audio_path"],
                "reference_text": reference["text"],
                "steps": sample_steps,
                "cfg_strength": cfg_strength,
                "speed": speed,
                "seed": effective_seed,
                "output_path": str(output_path),
            }
        )

        try:
            import wandb

            audio_logs[f"samples/sentence_{index:02d}"] = wandb.Audio(
                str(output_path),
                caption=f"{checkpoint_label}: {cleaned_sentence}",
                sample_rate=24000,
            )
        except Exception:
            pass

    _write_sample_manifest(sample_dir, manifest_rows)

    if os.getenv("WANDB_API_KEY"):
        try:
            import wandb

            run = wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=f"{wandb_group}-{checkpoint_label}",
                job_type="checkpoint-samples",
                config={
                    "speaker_id": speaker_id,
                    "speaker_gender": speaker_gender,
                    "run_name": run_name,
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_path": str(resolved_checkpoint),
                    "reference_audio_path": reference["audio_path"],
                    "reference_text": reference["text"],
                    "sample_steps": sample_steps,
                    "cfg_strength": cfg_strength,
                    "speed": speed,
                },
                reinit=True,
            )
            if audio_logs:
                wandb.log(audio_logs)
            wandb.finish()
        except Exception:
            pass

    model_volume.commit()
    return {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(resolved_checkpoint),
        "sample_dir": str(sample_dir),
        "num_samples": len(manifest_rows),
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=TRAIN_TIMEOUT_SECONDS,
    volumes={"/data": data_volume, "/model": model_volume},
    secrets=[modal.Secret.from_dotenv()],
)
def run_f5_single_speaker(
    speaker_id: int = DEFAULT_SPEAKER_ID,
    speaker_gender: str = DEFAULT_SPEAKER_GENDER,
    run_name: str = DEFAULT_RUN_NAME,
    prepare_data: bool = True,
    force_rebuild_prep: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_run_name: str = "",
    base_config: str = DEFAULT_BASE_CONFIG,
    custom_dataset_subdir: str = DEFAULT_CUSTOM_DATASET_SUBDIR,
    metadata_filename: str = DEFAULT_CUSTOM_METADATA_FILENAME,
    custom_audio_subdir: str = DEFAULT_CUSTOM_AUDIO_SUBDIR,
    custom_audio_column: str = DEFAULT_CUSTOM_AUDIO_COLUMN,
    custom_text_column: str = DEFAULT_CUSTOM_TEXT_COLUMN,
    custom_duration_column: str = DEFAULT_CUSTOM_DURATION_COLUMN,
    base_model_run: str = DEFAULT_BASE_MODEL_RUN,
    base_checkpoint_subpath: str = DEFAULT_BASE_CHECKPOINT_SUBPATH,
    base_checkpoint_path: str = "",
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    sample_every_epochs: int = DEFAULT_SAMPLE_EVERY_EPOCHS,
    sample_every_updates: int = DEFAULT_SAMPLE_EVERY_UPDATES,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
) -> dict[str, Any]:
    os.chdir(REMOTE_WORKSPACE)
    if REMOTE_WORKSPACE not in sys.path:
        sys.path.insert(0, REMOTE_WORKSPACE)

    _configure_runtime_environment()

    prep_dir = _prepared_dataset_dir(speaker_id, speaker_gender, custom_dataset_subdir)
    run_dir = _run_dir(speaker_id, speaker_gender, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"

    summary: dict[str, Any] = {
        "speaker_id": speaker_id,
        "speaker_gender": speaker_gender,
        "speaker_dir": _speaker_dir_name(speaker_id, speaker_gender),
        "run_name": run_name,
        "prepare_data": prepare_data,
        "force_rebuild_prep": force_rebuild_prep,
        "custom_dataset_subdir": custom_dataset_subdir or None,
        "metadata_filename": metadata_filename,
        "custom_audio_subdir": custom_audio_subdir or None,
        "custom_audio_column": custom_audio_column,
        "custom_text_column": custom_text_column,
        "custom_duration_column": custom_duration_column or None,
        "prepared_dataset_dir": str(prep_dir),
        "run_dir": str(run_dir),
        "base_config": base_config,
        "base_model_run": base_model_run or None,
        "base_checkpoint_subpath": base_checkpoint_subpath,
        "requested_base_checkpoint_path": base_checkpoint_path or None,
        "wandb_project": wandb_project,
        "requested_wandb_run_name": wandb_run_name or None,
        "epochs": epochs,
        "learning_rate": learning_rate if learning_rate > 0 else None,
        "gpu_type": GPU_TYPE,
        "sample_every_epochs": sample_every_epochs,
        "sample_every_updates": sample_every_updates,
        "sample_steps": sample_steps,
        "sample_sentences": DEFAULT_SAMPLE_SENTENCES,
        "started_at": _utc_now(),
        "status": "running",
    }
    sample_calls: list[Any] = []
    seen_checkpoint_steps: set[int] = set()

    try:
        if force_rebuild_prep:
            summary["prep_summary"] = _prepare_dataset(
                speaker_id=speaker_id,
                speaker_gender=speaker_gender,
                custom_dataset_subdir=custom_dataset_subdir,
                metadata_filename=metadata_filename,
                custom_audio_subdir=custom_audio_subdir,
                custom_audio_column=custom_audio_column,
                custom_text_column=custom_text_column,
                custom_duration_column=custom_duration_column,
                force_rebuild_prep=force_rebuild_prep,
            )
            summary["reused_prepared_dataset"] = False
        elif prepare_data and not _prepared_dataset_exists(prep_dir):
            summary["prep_summary"] = _prepare_dataset(
                speaker_id=speaker_id,
                speaker_gender=speaker_gender,
                custom_dataset_subdir=custom_dataset_subdir,
                metadata_filename=metadata_filename,
                custom_audio_subdir=custom_audio_subdir,
                custom_audio_column=custom_audio_column,
                custom_text_column=custom_text_column,
                custom_duration_column=custom_duration_column,
                force_rebuild_prep=False,
            )
            summary["reused_prepared_dataset"] = False
        else:
            summary["prep_summary"] = _validate_prepared_dataset(prep_dir)
            summary["reused_prepared_dataset"] = True
        _log_prep_summary(summary["prep_summary"])

        resolved_base_checkpoint = _resolve_base_checkpoint_path(
            base_model_run=base_model_run,
            base_checkpoint_subpath=base_checkpoint_subpath,
            base_checkpoint_path=base_checkpoint_path,
        )
        if resolved_base_checkpoint is not None:
            if not resolved_base_checkpoint.exists():
                raise FileNotFoundError(
                    f"Base checkpoint not found: {resolved_base_checkpoint}"
                )
            summary["resolved_base_checkpoint_path"] = str(resolved_base_checkpoint)
        else:
            summary["resolved_base_checkpoint_path"] = None

        (
            generated_config_path,
            checkpoints_dir,
            logger_type,
            effective_run_name,
            estimated_updates_per_epoch,
        ) = (
            _write_generated_config(
                speaker_id=speaker_id,
                speaker_gender=speaker_gender,
                run_name=run_name,
                prep_dir=prep_dir,
                run_dir=run_dir,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name or None,
                base_config=base_config,
                base_checkpoint_path=resolved_base_checkpoint,
                epochs=epochs,
                learning_rate=learning_rate,
                sample_every_epochs=sample_every_epochs,
                save_per_updates=sample_every_updates,
            )
        )
        summary["generated_config"] = str(generated_config_path)
        summary["checkpoints_dir"] = str(checkpoints_dir)
        summary["logger_type"] = logger_type
        summary["effective_wandb_run_name"] = effective_run_name
        summary["estimated_updates_per_epoch"] = estimated_updates_per_epoch
        if sample_every_epochs > 0 and estimated_updates_per_epoch > 0:
            summary["effective_sample_every_updates"] = (
                estimated_updates_per_epoch * sample_every_epochs
            )
        else:
            summary["effective_sample_every_updates"] = sample_every_updates

        cmd = [sys.executable, "scripts/train.py", "--config", str(generated_config_path)]
        summary["train_command"] = cmd
        process = subprocess.Popen(cmd, cwd=REMOTE_WORKSPACE)

        while True:
            for checkpoint_path in _iter_saved_checkpoints(checkpoints_dir):
                step = _parse_checkpoint_step(checkpoint_path)
                if step is None or step in seen_checkpoint_steps:
                    continue
                _wait_for_stable_file(checkpoint_path)
                model_volume.commit()
                seen_checkpoint_steps.add(step)
                sample_calls.append(
                    generate_checkpoint_samples.spawn(
                        speaker_id=speaker_id,
                        speaker_gender=speaker_gender,
                        run_name=run_name,
                        custom_dataset_subdir=custom_dataset_subdir,
                        checkpoint_path=str(checkpoint_path),
                        checkpoint_label=f"step_{step:07d}",
                        wandb_project=wandb_project,
                        wandb_group=effective_run_name,
                        sample_sentences=DEFAULT_SAMPLE_SENTENCES,
                        sample_steps=sample_steps,
                    )
                )

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
                break
            time.sleep(15)

        final_checkpoint = checkpoints_dir / "model_last.pt"
        if final_checkpoint.exists():
            _wait_for_stable_file(final_checkpoint)
            model_volume.commit()
            sample_calls.append(
                generate_checkpoint_samples.spawn(
                    speaker_id=speaker_id,
                    speaker_gender=speaker_gender,
                    run_name=run_name,
                    custom_dataset_subdir=custom_dataset_subdir,
                    checkpoint_path=str(final_checkpoint),
                    checkpoint_label="final",
                    wandb_project=wandb_project,
                    wandb_group=effective_run_name,
                    sample_sentences=DEFAULT_SAMPLE_SENTENCES,
                    sample_steps=sample_steps,
                )
            )

        sample_results = []
        sample_errors = []
        for call in sample_calls:
            try:
                sample_results.append(call.get())
            except Exception as sample_exc:
                sample_errors.append(str(sample_exc))
        summary["checkpoint_sample_jobs"] = sample_results
        if sample_errors:
            summary["checkpoint_sample_errors"] = sample_errors

        summary["status"] = "completed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["finished_at"] = _utc_now()
        _write_json(summary_path, summary)
        _finish_wandb_if_active()
        _commit_volumes()

    return summary


@app.local_entrypoint()
def main(
    speaker_id: int = DEFAULT_SPEAKER_ID,
    speaker_gender: str = DEFAULT_SPEAKER_GENDER,
    run_name: str = DEFAULT_RUN_NAME,
    prepare_data: bool = True,
    force_rebuild_prep: bool = False,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_run_name: str = "",
    base_config: str = DEFAULT_BASE_CONFIG,
    custom_dataset_subdir: str = DEFAULT_CUSTOM_DATASET_SUBDIR,
    metadata_filename: str = DEFAULT_CUSTOM_METADATA_FILENAME,
    custom_audio_subdir: str = DEFAULT_CUSTOM_AUDIO_SUBDIR,
    custom_audio_column: str = DEFAULT_CUSTOM_AUDIO_COLUMN,
    custom_text_column: str = DEFAULT_CUSTOM_TEXT_COLUMN,
    custom_duration_column: str = DEFAULT_CUSTOM_DURATION_COLUMN,
    base_model_run: str = DEFAULT_BASE_MODEL_RUN,
    base_checkpoint_subpath: str = DEFAULT_BASE_CHECKPOINT_SUBPATH,
    base_checkpoint_path: str = "",
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    sample_every_epochs: int = DEFAULT_SAMPLE_EVERY_EPOCHS,
    sample_every_updates: int = DEFAULT_SAMPLE_EVERY_UPDATES,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
):
    run_f5_single_speaker.remote(
        speaker_id=speaker_id,
        speaker_gender=speaker_gender,
        run_name=run_name,
        prepare_data=prepare_data,
        force_rebuild_prep=force_rebuild_prep,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        base_config=base_config,
        custom_dataset_subdir=custom_dataset_subdir,
        metadata_filename=metadata_filename,
        custom_audio_subdir=custom_audio_subdir,
        custom_audio_column=custom_audio_column,
        custom_text_column=custom_text_column,
        custom_duration_column=custom_duration_column,
        base_model_run=base_model_run,
        base_checkpoint_subpath=base_checkpoint_subpath,
        base_checkpoint_path=base_checkpoint_path,
        epochs=epochs,
        learning_rate=learning_rate,
        sample_every_epochs=sample_every_epochs,
        sample_every_updates=sample_every_updates,
        sample_steps=sample_steps,
    )

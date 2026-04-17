"""
Modal entrypoint to publish a trained F5-TTS Shona model from the model volume
to the Hugging Face Hub.

This uploader targets the single-speaker fine-tune layout created by
`scripts/run_f5_single_speaker.py`:

  /model/sna-f5-tts/speakers/{speaker_id}_{speaker_gender}_full/{run_name}

It uploads a cleaned public artifact set:
- `model.pt`
- `vocab.txt`
- `README.md`
- optional `samples/`
- optional research metadata under `research/`

Example:
  uv run modal run -d scripts/upload_f5_tts_to_hf.py \
    --speaker-id 999 \
    --speaker-gender Custom \
    --run-name shona_f5_hq_voice_from_base_v1 \
    --hf-username manassehzw \
    --repo-name f5-tts-shona-voice
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


MODEL_VOLUME_NAME = "sna-model-vol"
DATA_VOLUME_NAME = "sna-data-vol"

DEFAULT_SPEAKER_ID = 999
DEFAULT_SPEAKER_GENDER = "Custom"
DEFAULT_RUN_NAME = "shona_f5_hq_voice_from_base_v1"
HF_TOKEN_ENV_VAR_DEFAULT = "HF_TOKEN"
DEFAULT_CUSTOM_DATASET_SUBDIR = ""

DEFAULT_REPO_NAME = "f5-tts-shona-voice"
DEFAULT_BASE_MODEL_ID = "SWivid/F5-TTS"
DEFAULT_DATASET_REPO_ID = ""
DEFAULT_LANGUAGE = "sna"
DEFAULT_LICENSE = "cc-by-nc-4.0"
DEFAULT_PRETTY_NAME = "Shona F5-TTS Voice"
DEFAULT_TRAINING_DESCRIPTION = (
    "Fine-tuned from the F5-TTS base model on curated Shona speech data, "
    "including a broader Shona adaptation stage followed by speaker-focused adaptation."
)
DEFAULT_AUTHOR = "Manasseh Changachirere"
DEFAULT_AUTHOR_AFFILIATION = "Harare Institute of Technology"
DEFAULT_AUTHOR_URL = "https://www.manasseh.dev/"


def _speaker_dir_name(speaker_id: int, speaker_gender: str) -> str:
    return f"{speaker_id}_{speaker_gender}_full"


def run_dir_for(
    speaker_id: int,
    speaker_gender: str,
    run_name: str,
    model_root: Path = Path("/model"),
) -> Path:
    return model_root / "sna-f5-tts" / "speakers" / _speaker_dir_name(speaker_id, speaker_gender) / run_name


def prep_dir_for(
    speaker_id: int,
    speaker_gender: str,
    custom_dataset_subdir: str = "",
    data_root: Path = Path("/data"),
) -> Path:
    if custom_dataset_subdir.strip():
        return data_root / custom_dataset_subdir.strip("/ ") / "f5_prepared" / "default"
    return data_root / "speakers" / _speaker_dir_name(speaker_id, speaker_gender) / "f5_prepared" / "default"


def _coalesce_repo_id(hf_username: str, repo_name: str) -> str:
    repo_name = repo_name.strip()
    if "/" in repo_name:
        return repo_name
    if not hf_username.strip():
        raise ValueError("hf_username is required when repo_name does not include an owner.")
    return f"{hf_username.strip()}/{repo_name}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id}"


def _export_safetensors(checkpoint_path: Path) -> Path:
    import torch
    from safetensors.torch import save_file

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    tensor_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
    }

    export_dir = Path(tempfile.mkdtemp(prefix="f5tts_export_"))
    export_path = export_dir / "model.safetensors"
    save_file(tensor_dict, str(export_path))
    return export_path


def _format_hours(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_model_card(
    *,
    repo_id: str,
    pretty_name: str,
    base_model_id: str,
    dataset_repo_id: str,
    language: str,
    license_id: str,
    author_name: str,
    author_affiliation: str,
    author_url: str,
    training_description: str,
    summary: dict[str, Any],
    prep_summary: dict[str, Any],
) -> str:
    rows_written = prep_summary.get("rows_written", "N/A")
    total_duration_hours = prep_summary.get("total_duration_hours")
    if total_duration_hours is None:
        total_duration_hours = prep_summary.get("rows_written")
    duration_text = _format_hours(total_duration_hours)
    dataset_ref_text = (
        f"[`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id})"
        if dataset_repo_id.strip()
        else "private / local research corpus"
    )

    yaml_block = f"""---
language:
- {language}
license: {license_id}
tags:
- text-to-speech
- tts
- shona
- voice-cloning
- f5-tts
- audio
base_model:
- {base_model_id}
library_name: none
---
"""

    dataset_line = (
        f"- **Training dataset reference:** [`{dataset_repo_id}`](https://huggingface.co/datasets/{dataset_repo_id})"
        if dataset_repo_id.strip()
        else "- **Training dataset reference:** private / local research corpus"
    )

    body = f"""
# {pretty_name}

{pretty_name} is a Shona (`{language}`) text-to-speech model built on top of
[`{base_model_id}`](https://huggingface.co/{base_model_id}). It is intended for
research on natural-sounding Shona speech synthesis and voice adaptation.

## Model Details

- **Author:** [{author_name} ({author_affiliation})]({author_url})
- **Base model:** [`{base_model_id}`](https://huggingface.co/{base_model_id})
{dataset_line}
- **Language:** Shona
- **Model family:** F5-TTS
- **Approx. prepared training duration:** {duration_text} hours
- **Prepared training rows:** {rows_written}

## Overview

{training_description}

## Files

- `model.safetensors`: inference-oriented weight export
- `model.pt`: full compatibility checkpoint
- `vocab.txt`: tokenizer vocabulary used by the model
- `samples/`: optional example generations, when included
- `research/`: optional run metadata for reproducibility

## Intended Use

This model is intended for:

- Shona TTS research
- speech technology prototyping
- comparative evaluation against other open TTS systems
- voice adaptation experiments built on Shona speech synthesis

It is not positioned as a production-hardened commercial speech API.

## Compatibility

This repository does **not** follow the standard `transformers` text-to-speech
layout. It is intended for the F5-TTS / `sna-f5-tts` inference stack.

For inference you will generally want:

1. the upstream F5-TTS package installed
2. the matching inference code path for this checkpoint format
3. a short reference audio clip and transcript for voice-cloning style synthesis

## Example Usage

```python
from scripts.inference import SautiInference

checkpoint_path = "model.safetensors"
vocab_path = "vocab.txt"

engine = SautiInference(
    checkpoint_path=checkpoint_path,
    vocab_path=vocab_path,
)

engine.generate(
    text="Mangwanani. Ndamuka zvakanaka nhasi.",
    ref_audio_path="reference.wav",
    ref_text="Mangwanani. Ndamuka zvakanaka nhasi.",
    output_path="generated.wav",
    nfe_steps=24,
    cfg_strength=2.0,
    speed=1.0,
)
```

## Inference Notes

- This model is best used with a short, clean reference clip.
- Long text is typically handled by sentence chunking rather than true streaming.
- Faster inference is possible with lower `nfe_steps`, at the cost of some quality.

## Training Provenance

- **Base model:** [`{base_model_id}`](https://huggingface.co/{base_model_id})
- **Training data reference:** {dataset_ref_text}
- **Training setup:** F5-TTS fine-tuning with Shona adaptation and speaker-focused refinement
- **Research metadata:** available under `research/` when included

## Limitations

- This is a research checkpoint and may be sensitive to prompt/reference mismatch.
- Long-form synthesis is typically handled by chunking, not native token-level streaming.
- Voice identity and pronunciation quality can vary depending on the reference audio used.

## Citation

If you use this model, please also credit the upstream F5-TTS project:

- [`{base_model_id}`](https://huggingface.co/{base_model_id})
- [F5-TTS GitHub](https://github.com/SWivid/F5-TTS)
"""

    return yaml_block + body


app = modal.App("sna-f5-tts-upload")
model_vol = modal.Volume.from_name(MODEL_VOLUME_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME)

image = modal.Image.debian_slim(python_version="3.10").uv_pip_install(
    "huggingface_hub",
    "numpy",
    "torch",
    "safetensors",
)


@app.function(
    image=image,
    cpu=2.0,
    memory=8192,
    timeout=60 * 60,
    volumes={"/model": model_vol, "/data": data_vol},
    secrets=[modal.Secret.from_dotenv(__file__)],
)
def upload_f5_tts_model(
    speaker_id: int = DEFAULT_SPEAKER_ID,
    speaker_gender: str = DEFAULT_SPEAKER_GENDER,
    run_name: str = DEFAULT_RUN_NAME,
    custom_dataset_subdir: str = DEFAULT_CUSTOM_DATASET_SUBDIR,
    hf_username: str = "",
    repo_name: str = DEFAULT_REPO_NAME,
    private_repo: bool = False,
    replace_existing: bool = True,
    include_research_artifacts: bool = True,
    hf_token_env_var: str = HF_TOKEN_ENV_VAR_DEFAULT,
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    pretty_name: str = DEFAULT_PRETTY_NAME,
    training_description: str = DEFAULT_TRAINING_DESCRIPTION,
    language: str = DEFAULT_LANGUAGE,
    license_id: str = DEFAULT_LICENSE,
    author_name: str = DEFAULT_AUTHOR,
    author_affiliation: str = DEFAULT_AUTHOR_AFFILIATION,
    author_url: str = DEFAULT_AUTHOR_URL,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    hf_token = os.environ.get(hf_token_env_var, "").strip()
    if not hf_token:
        raise ValueError(f"Missing {hf_token_env_var} in environment.")

    repo_id = _coalesce_repo_id(hf_username=hf_username, repo_name=repo_name)

    run_dir = run_dir_for(speaker_id=speaker_id, speaker_gender=speaker_gender, run_name=run_name)
    prep_dir = prep_dir_for(
        speaker_id=speaker_id,
        speaker_gender=speaker_gender,
        custom_dataset_subdir=custom_dataset_subdir,
    )

    checkpoint_path = run_dir / "checkpoints" / "model_last.pt"
    generated_config_path = run_dir / "config.generated.yaml"
    summary_path = run_dir / "summary.json"
    prep_summary_path = prep_dir / "prep_summary.json"
    vocab_path = prep_dir / "vocab.txt"
    upload_audit_path = run_dir / "upload_audit.json"
    final_sample_dir = run_dir / "checkpoint_samples" / "final"
    sample_manifest_path = final_sample_dir / "manifest.csv"

    required_paths = {
        "checkpoint": checkpoint_path,
        "generated_config": generated_config_path,
        "summary": summary_path,
        "prep_summary": prep_summary_path,
        "vocab": vocab_path,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    summary = _read_json(summary_path)
    prep_summary = _read_json(prep_summary_path)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private_repo, exist_ok=True)

    if replace_existing:
        existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        for path_in_repo in existing_files:
            api.delete_file(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Remove stale file before republishing {pretty_name}",
            )

    print("=" * 72)
    print("SNA F5-TTS - UPLOAD MODEL")
    print("=" * 72)
    print(f"Speaker:        {speaker_id}_{speaker_gender}_full")
    print(f"Run name:       {run_name}")
    print(f"Repo:           {repo_id}")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Prep dir:       {prep_dir}")
    if custom_dataset_subdir.strip():
        print(f"Custom dataset: {custom_dataset_subdir}")
    print(f"Base model:     {base_model_id}")
    print(f"Dataset ref:    {dataset_repo_id or 'private/local corpus'}")
    print(f"Replace repo:   {replace_existing}")

    print("Exporting safetensors...")
    safetensors_path = _export_safetensors(checkpoint_path)

    print("Uploading model checkpoint...")
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo="model.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(safetensors_path),
        path_in_repo="model.safetensors",
        repo_id=repo_id,
        repo_type="model",
    )

    print("Uploading vocab...")
    api.upload_file(
        path_or_fileobj=str(vocab_path),
        path_in_repo="vocab.txt",
        repo_id=repo_id,
        repo_type="model",
    )

    if include_research_artifacts:
        print("Uploading research metadata...")
        api.upload_file(
            path_or_fileobj=str(generated_config_path),
            path_in_repo="research/train_config.yaml",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="research/summary.json",
            repo_id=repo_id,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(prep_summary_path),
            path_in_repo="research/prep_summary.json",
            repo_id=repo_id,
            repo_type="model",
        )

    if sample_manifest_path.exists():
        print("Uploading final sample manifest...")
        api.upload_file(
            path_or_fileobj=str(sample_manifest_path),
            path_in_repo="samples/final_manifest.csv",
            repo_id=repo_id,
            repo_type="model",
        )

        sample_wavs = sorted(final_sample_dir.glob("*.wav"))
        for wav_path in sample_wavs:
            api.upload_file(
                path_or_fileobj=str(wav_path),
                path_in_repo=f"samples/{wav_path.name}",
                repo_id=repo_id,
                repo_type="model",
            )

    readme = _build_model_card(
        repo_id=repo_id,
        pretty_name=pretty_name,
        base_model_id=base_model_id,
        dataset_repo_id=dataset_repo_id,
        language=language,
        license_id=license_id,
        author_name=author_name,
        author_affiliation=author_affiliation,
        author_url=author_url,
        training_description=training_description,
        summary=summary,
        prep_summary=prep_summary,
    )

    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=io.BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    upload_audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "speaker_id": speaker_id,
        "speaker_gender": speaker_gender,
        "run_name": run_name,
        "custom_dataset_subdir": custom_dataset_subdir or None,
        "repo_id": repo_id,
        "repo_url": _repo_url(repo_id),
        "checkpoint_path": str(checkpoint_path),
        "safetensors_path": str(safetensors_path),
        "generated_config_path": str(generated_config_path),
        "summary_path": str(summary_path),
        "prep_summary_path": str(prep_summary_path),
        "vocab_path": str(vocab_path),
        "dataset_repo_id": dataset_repo_id,
        "base_model_id": base_model_id,
        "private_repo": private_repo,
        "replace_existing": replace_existing,
        "include_research_artifacts": include_research_artifacts,
        "readme_uploaded": True,
        "training_metadata_uploaded": include_research_artifacts,
        "sample_manifest_uploaded": sample_manifest_path.exists(),
    }
    upload_audit_path.write_text(
        json.dumps(upload_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_vol.commit()

    print("=" * 72)
    print("UPLOAD COMPLETE")
    print(f"Model repo: {_repo_url(repo_id)}")
    print(f"Audit:      {upload_audit_path}")
    print("=" * 72)

    return upload_audit


@app.local_entrypoint()
def main(
    speaker_id: int = DEFAULT_SPEAKER_ID,
    speaker_gender: str = DEFAULT_SPEAKER_GENDER,
    run_name: str = DEFAULT_RUN_NAME,
    custom_dataset_subdir: str = DEFAULT_CUSTOM_DATASET_SUBDIR,
    hf_username: str = "",
    repo_name: str = DEFAULT_REPO_NAME,
    private_repo: bool = False,
    replace_existing: bool = True,
    include_research_artifacts: bool = True,
    hf_token_env_var: str = HF_TOKEN_ENV_VAR_DEFAULT,
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    pretty_name: str = DEFAULT_PRETTY_NAME,
    training_description: str = DEFAULT_TRAINING_DESCRIPTION,
    language: str = DEFAULT_LANGUAGE,
    license_id: str = DEFAULT_LICENSE,
    author_name: str = DEFAULT_AUTHOR,
    author_affiliation: str = DEFAULT_AUTHOR_AFFILIATION,
    author_url: str = DEFAULT_AUTHOR_URL,
):
    payload = upload_f5_tts_model.remote(
        speaker_id=speaker_id,
        speaker_gender=speaker_gender,
        run_name=run_name,
        custom_dataset_subdir=custom_dataset_subdir,
        hf_username=hf_username,
        repo_name=repo_name,
        private_repo=private_repo,
        replace_existing=replace_existing,
        include_research_artifacts=include_research_artifacts,
        hf_token_env_var=hf_token_env_var,
        dataset_repo_id=dataset_repo_id,
        base_model_id=base_model_id,
        pretty_name=pretty_name,
        training_description=training_description,
        language=language,
        license_id=license_id,
        author_name=author_name,
        author_affiliation=author_affiliation,
        author_url=author_url,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

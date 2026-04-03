# Sauti TTS

![Lab](https://img.shields.io/badge/MsingiAI-AI%20Research%20Lab-0f172a?style=flat-square)
![Language](https://img.shields.io/badge/Language-Swahili-1d4ed8?style=flat-square)
![Base Model](https://img.shields.io/badge/Base-F5--TTS%20v1-15803d?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active%20Research-b45309?style=flat-square)

Swahili text-to-speech research developed by **MsingiAI**.

**Sauti TTS** is a Swahili TTS research project built on top of
[F5-TTS v1](https://github.com/SWivid/F5-TTS) and trained on the
[Google WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) `swa_tts`
dataset. The repository packages the full workflow needed to reproduce the
system: dataset preparation, fine-tuning, inference, evaluation, and Modal
training jobs.

The goal is straightforward: build strong speech systems for African languages
with code that is practical, reproducible, and easy to extend.

> "Sauti" means "voice" in Swahili.

**Developed by MsingiAI**

MsingiAI is an AI Research Lab focused on African language technology, speech
systems, and production-ready machine learning workflows.

**Repository Note**

Internal Python package names, checkpoint directories, and command paths still
use `sauti_tts` for compatibility with the current codebase.

## Why This Repo Exists

- Reproduce a Swahili TTS system from a strong open base model.
- Provide a clean research stack for data preparation, training, inference,
  and evaluation.
- Support both full fine-tuning and lower-VRAM LoRA adaptation.
- Make cloud training practical with Modal-based jobs and persistent volumes.

## System Overview

| Component | Choice | Why it was chosen |
|-----------|--------|-------------------|
| Base model | F5-TTS v1 Base | Strong flow-matching TTS model with DiT backbone and voice cloning support |
| Dataset | Google WaxalNLP `swa_tts` | High-quality Swahili speech data with aligned text |
| Vocoder | Vocos | Integrated high-fidelity neural vocoder used by F5-TTS |
| Fine-tuning modes | Full fine-tune and LoRA | Supports both best-quality and budget-GPU workflows |
| Deployment path | Local scripts and Modal jobs | Keeps experimentation and remote training in one repo |

## Research Highlights

- Swahili-first text normalization and data preparation.
- F5-TTS compatible dataset export with `metadata.csv`, `raw.arrow`, and
  `duration.json`.
- Full fine-tuning pipeline for 24 GB+ GPUs.
- LoRA option for lower-memory experiments.
- Inference cleanup passes for quiet pauses and short end-click artifacts.
- Evaluation hooks for intelligibility, speaker similarity, and MOS-style
  analysis.

## Quick Start

### 1. Install

```bash
# Clone this repo
git clone <your-repo-url> sauti-tts
cd sauti-tts

# Install dependencies
pip install -r requirements.txt

# Install F5-TTS from source (required for training)
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS && pip install -e . && cd ..
```

### 2. Prepare Data

```bash
python scripts/prepare_waxal_data.py \
    --output_dir data/waxal_swahili \
    --sample_rate 24000 \
    --min_duration 1.0 \
    --max_duration 30.0
```

### 3. Fine-Tune

```bash
# Single GPU fine-tuning (RTX 3090 / 4090, 24 GB)
python scripts/train.py --config configs/finetune_single_gpu.yaml

# Multi-GPU fine-tuning
accelerate launch scripts/train.py --config configs/finetune_multi_gpu.yaml

# LoRA fine-tuning (about 12 GB VRAM)
python scripts/train.py --config configs/finetune_lora.yaml
```

### 4. Train on Modal

```bash
# Authenticate once
modal setup

# Train with an existing prepared dataset
modal run scripts/modal_train.py --config configs/finetune_single_gpu.yaml

# Prepare data and train in one run
modal run scripts/modal_train.py \
    --config configs/finetune_single_gpu.yaml \
    --prepare-data
```

Modal volumes are used automatically:

- `sauti-tts-data` -> `data/`
- `sauti-tts-ckpts` -> `ckpts/`
- `sauti-tts-outputs` -> `outputs/`

### 5. Run Inference

```bash
# Generate speech from text
python scripts/inference.py \
    --text "Habari, karibu kwenye Sauti TTS" \
    --ref_audio data/reference/sample.wav \
    --output outputs/generated.wav

# Batch inference from file
python scripts/inference.py \
    --text_file data/test_sentences.txt \
    --ref_audio data/reference/sample.wav \
    --output_dir outputs/batch/
```

### 6. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint ckpts/sauti_tts/model_best.pt \
    --test_set data/waxal_swahili/test \
    --output_dir outputs/eval/
```

## Training Recipes

### Full Fine-Tune

- Base checkpoint: `SWivid/F5-TTS`
- Model type: `F5TTS_v1_Base`
- Sample rate: `24000`
- Default single-GPU learning rate: `1e-5`
- Default batch size: `1600` frames per GPU
- Warmup: `500` updates
- Precision: `fp16`
- EMA: enabled

### Multi-GPU Fine-Tune

- Output directory: `ckpts/sauti_tts_multi`
- Learning rate: `2e-5`
- Batch size: `2000` frames per GPU
- Precision: `bf16`
- Logging: Weights and Biases

### LoRA Fine-Tune

- Output directory: `ckpts/sauti_tts_lora`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Target modules: `to_q`, `to_k`, `to_v`, `to_out.0`
- EMA: disabled

## Repository Layout

```text
sauti-tts/
|-- configs/
|   |-- finetune_single_gpu.yaml
|   |-- finetune_multi_gpu.yaml
|   |-- finetune_lora.yaml
|   `-- inference.yaml
|-- scripts/
|   |-- prepare_waxal_data.py
|   |-- train.py
|   |-- inference.py
|   |-- evaluate.py
|   `-- modal_train.py
|-- sauti_tts/
|   |-- __init__.py
|   |-- data.py
|   |-- model.py
|   |-- trainer.py
|   |-- utils.py
|   `-- metrics.py
|-- notebooks/
|   `-- sauti_tts_demo.ipynb
|-- data/
|-- ckpts/
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Hardware Guidance

| Setup | GPU | VRAM | Batch Size | Training Time |
|-------|-----|------|------------|---------------|
| Full fine-tune | RTX 3090 | 24 GB | 1600 frames | about 24-48h |
| Full fine-tune | RTX 4090 | 24 GB | 4000 frames | about 16-32h |
| Full fine-tune | A100 or H100 | 40-80 GB | 12000 frames | about 8-16h |
| LoRA fine-tune | RTX 3060 | 12 GB | 800 frames | about 24-48h |

## Dataset Notes

The WaxalNLP Swahili TTS subset includes:

- Studio-quality speech recordings.
- Text aligned to audio utterances.
- Male and female speakers.
- Phonetically useful coverage for Swahili synthesis.
- Licensing that should be reviewed carefully before public model release.

## Open Source Scope

- Repository code: released under the MIT License. See `LICENSE`.
- Third-party attributions: documented in `THIRD_PARTY_NOTICES.md`.
- Checkpoint release guidance: documented in `MODEL_CARD.md`.
- Base model weights: follow the upstream F5-TTS license.
- Dataset usage: follow the WaxalNLP dataset license and provider-specific
  terms.

If you plan to publish checkpoints, document the upstream model license and
dataset obligations clearly in the release notes.

## Citation

If you use Sauti TTS in research, please cite:

```bibtex
@misc{sauti_tts_2026,
  title={Sauti TTS: Swahili Text-to-Speech via F5-TTS Fine-tuning on WaxalNLP},
  author={MsingiAI},
  year={2026},
  url={https://github.com/your-repo/sauti-tts}
}

@article{chen2024f5tts,
  title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author={Chen, Yushen and others},
  journal={arXiv preprint arXiv:2410.06885},
  year={2024}
}

@misc{waxal2026,
  title={WAXAL: A Multilingual African Speech Dataset},
  author={Google Research and collaborators},
  year={2026},
  url={https://huggingface.co/datasets/google/WaxalNLP}
}
```

## Acknowledgements

- [F5-TTS](https://github.com/SWivid/F5-TTS) for the base architecture and
  training stack.
- [Google WaxalNLP](https://huggingface.co/datasets/google/WaxalNLP) for the
  Swahili TTS dataset.

## Release Files

This repository now includes the core release files expected for a public
research project:

- `LICENSE` for the repository code.
- `THIRD_PARTY_NOTICES.md` for upstream model and dataset attribution.
- `MODEL_CARD.md` for checkpoint-style release notes and usage guidance.

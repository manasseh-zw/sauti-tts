# Sauti TTS Model Card

This file is the model-card style release note for any published **Sauti TTS**
checkpoint from **MsingiAI**. If you release multiple checkpoints, duplicate
this file per release and fill in the checkpoint-specific details.

## Model Details

- Model name: Sauti TTS
- Developer: MsingiAI
- Primary language: Swahili
- Base model family: F5-TTS v1 Base
- Vocoder path: Vocos through the F5-TTS stack
- Repository: `sauti-tts`

## Training Data

- Primary dataset family: Google WaxalNLP
- Expected subset: `swa_tts`
- Data preparation in this repo includes resampling, trimming, normalization,
  and F5-TTS-compatible metadata export

## Intended Use

- Swahili text-to-speech research
- Voice-cloning experiments using a short reference clip
- Benchmarking fine-tuning workflows for African language speech systems

## Out-of-Scope Use

- Impersonation, fraud, or deception
- Biometric identity claims
- Safety-critical decision systems
- Any release that ignores upstream model or dataset license obligations

## Training Setup Summary

- Base checkpoint: `SWivid/F5-TTS`
- Typical sample rate: `24000`
- Fine-tuning modes supported: full fine-tune and LoRA
- Training scripts: `scripts/train.py`, `scripts/modal_train.py`
- Data preparation script: `scripts/prepare_waxal_data.py`

## Evaluation Notes

This repository includes evaluation utilities for:

- intelligibility-style checks
- speaker similarity measurements
- MOS-style analysis workflows

Any published checkpoint should document the exact evaluation set, metrics,
reference audio conditions, and known failure cases used for that release.

## Known Limitations

- Output quality depends heavily on reference audio quality and transcript
  quality.
- Very short pauses and waveform edges may require cleanup during inference.
- Performance outside Swahili has not been established in this repository.
- A checkpoint release may inherit restrictions from upstream model and dataset
  licenses.

## Release Guidance

For each public checkpoint release, document:

- checkpoint name and date
- exact training config
- exact data subset used
- intended use and out-of-scope use
- evaluation summary
- known limitations
- upstream license references

## Licensing and Attribution

- Repository code: MIT. See `LICENSE`.
- Upstream model and dataset attributions: see `THIRD_PARTY_NOTICES.md`.
- Derived checkpoint distribution must respect upstream model and dataset
  licensing terms in addition to the repository code license.

## Current Repository Status

This repository does not bundle checkpoints in version control. When a
checkpoint is released, update this file with the checkpoint-specific metadata
and publish it alongside the weights.

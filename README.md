# Shona F5-TTS

Shona text-to-speech research and training workflows built on top of
[F5-TTS](https://github.com/SWivid/F5-TTS).

This repository is now focused on the Shona adaptation path we actually used:

- multi-speaker Shona base adaptation on Modal
- single-speaker Shona voice adaptation on Modal
- local and Modal inference wrappers
- Hugging Face model publishing

Internal package names still use `sauti_tts` for compatibility with the code
already in the repo.

## Active Scripts

- [scripts/run_f5_shona_base.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/run_f5_shona_base.py): multi-speaker Shona base training on Modal
- [scripts/run_f5_single_speaker.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/run_f5_single_speaker.py): single-speaker adaptation and refinement on Modal
- [scripts/inference_f5_single_speaker.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/inference_f5_single_speaker.py): Modal inference wrapper for saved checkpoints
- [scripts/local_infer_hf_f5.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/local_infer_hf_f5.py): local inference from a published Hugging Face model
- [scripts/upload_f5_tts_to_hf.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/upload_f5_tts_to_hf.py): publish a trained model to Hugging Face
- [scripts/train.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/train.py): shared F5 training entrypoint used by the Modal wrappers
- [scripts/inference.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/inference.py): shared inference engine and long-text chunking

## Repo Layout

```text
sna-f5-tts/
|-- configs/
|   |-- finetune_lora.yaml
|   |-- finetune_multi_gpu.yaml
|   |-- finetune_single_gpu.yaml
|   `-- inference.yaml
|-- sauti_tts/
|   |-- data.py
|   |-- model.py
|   |-- trainer.py
|   `-- utils.py
|-- scripts/
|   |-- run_f5_shona_base.py
|   |-- run_f5_single_speaker.py
|   |-- inference_f5_single_speaker.py
|   |-- local_infer_hf_f5.py
|   |-- upload_f5_tts_to_hf.py
|   |-- train.py
|   `-- inference.py
|-- requirements.txt
`-- README.md
```

## Install

```bash
pip install -r requirements.txt

git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS && pip install -e . && cd ..
```

## Modal Training

### 1. Shona Base Adaptation

```bash
modal run scripts/run_f5_shona_base.py \
  --dataset-subdir wav_normalised \
  --metadata-filename metadata_normalized.csv \
  --run-name shona_base_v1 \
  --prepare-data \
  --wandb-project sna-f5-tts \
  --epochs 3
```

### 2. Single-Speaker Adaptation

```bash
modal run scripts/run_f5_single_speaker.py \
  --speaker-id 1 \
  --speaker-gender Female \
  --run-name shona_f5_stage2_v1 \
  --no-prepare-data \
  --wandb-project sna-f5-tts \
  --base-model-run shona_base_v1 \
  --epochs 5
```

## Local Inference From Hugging Face

This repo uses F5-style reference-audio inference, so local generation needs a
reference clip and its transcript.

```bash
python scripts/local_infer_hf_f5.py \
  --repo-id manassehzw/sna-f5-tts \
  --ref-audio /absolute/path/to/reference.wav \
  --ref-text "Mangwanani. Ndamuka zvakanaka nhasi." \
  --text "Mhoro shamwari yangu. Wakafamba sei nezuro?" \
  --output-dir outputs/local_hf_test
```

The script downloads `model.pt` and `vocab.txt` from Hugging Face, generates a
WAV, and writes an `inference_summary.json` with elapsed time and RTF.

## Notes

- The repo is now Shona-focused and old Swahili-specific helper scripts have
  been removed.
- Public model publishing is handled through
  [scripts/upload_f5_tts_to_hf.py](/Users/manasseh/Projects/hit400/sna-f5-tts/scripts/upload_f5_tts_to_hf.py).
- Some internal class and package names still contain `sauti_tts`; that is
  compatibility debt, not an indication that this repo is still Swahili-first.

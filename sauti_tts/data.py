"""
Sauti TTS — Data pipeline for WaxalNLP Swahili TTS dataset.
Handles downloading, preprocessing, and creating F5-TTS compatible data format.
"""

import os
import csv
import json
import io
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter
import pyarrow.parquet as pq

from sauti_tts.utils import (
    normalize_audio,
    trim_silence,
    normalize_swahili_text,
    get_audio_duration,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Single audio-text pair."""
    id: str
    audio_path: str
    text: str
    duration: float
    speaker_id: str = ""
    gender: str = ""


@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_samples: int = 0
    total_duration_hours: float = 0.0
    avg_duration_sec: float = 0.0
    min_duration_sec: float = float("inf")
    max_duration_sec: float = 0.0
    num_speakers: int = 0
    speakers: Dict[str, float] = field(default_factory=dict)
    gender_distribution: Dict[str, int] = field(default_factory=dict)
    duration_histogram: List[int] = field(default_factory=lambda: [0] * 30)


class WaxalSwahiliDataset:
    """
    Download and preprocess the WaxalNLP Swahili TTS dataset
    into F5-TTS compatible format.
    """

    HF_DATASET_NAME = "google/WaxalNLP"
    HF_CONFIG_NAME = "swa_tts"

    def __init__(
        self,
        output_dir: str = "data/waxal_swahili",
        sample_rate: int = 24000,
        min_duration: float = 1.0,
        max_duration: float = 30.0,
        normalize: bool = True,
        trim: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.normalize = normalize
        self.trim = trim

        # Create output directories
        self.wavs_dir = self.output_dir / "wavs"
        self.wavs_dir.mkdir(parents=True, exist_ok=True)

    def download_and_prepare(self) -> DatasetStats:
        """
        Full pipeline: download from HuggingFace -> preprocess -> save in F5-TTS format.
        Returns dataset statistics.
        """
        logger.info("=" * 60)
        logger.info("Sauti TTS — WaxalNLP Swahili Data Preparation")
        logger.info("=" * 60)

        # Step 1: Download
        logger.info("\n[1/4] Downloading WaxalNLP Swahili TTS dataset...")
        raw_dataset = self._download_dataset()

        # Step 2: Process each split
        all_stats = DatasetStats()
        for split_name in ["train", "validation", "test"]:
            if split_name in raw_dataset:
                logger.info(f"\n[2/4] Processing '{split_name}' split...")
                split_data = raw_dataset[split_name]
                samples = self._process_split(split_data, split_name)

                logger.info(
                    f"  -> {len(samples)} samples after filtering "
                    f"(duration: {self.min_duration}-{self.max_duration}s)"
                )

                # Save metadata
                self._save_metadata(samples, split_name)
                self._update_stats(all_stats, samples)

        # Step 3: Create F5-TTS formatted metadata
        logger.info("\n[3/4] Creating F5-TTS compatible metadata...")
        self._create_f5tts_metadata()

        # Step 4: Save statistics
        logger.info("\n[4/4] Saving dataset statistics...")
        self._save_stats(all_stats)

        logger.info("\n" + "=" * 60)
        logger.info("Data preparation complete!")
        logger.info(f"  Total samples: {all_stats.total_samples}")
        logger.info(f"  Total duration: {all_stats.total_duration_hours:.2f} hours")
        logger.info(f"  Speakers: {all_stats.num_speakers}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 60)

        return all_stats

    def _download_dataset(self):
        """Download dataset parquet shards from HuggingFace."""
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_files = list_repo_files(self.HF_DATASET_NAME, repo_type="dataset")
        prefix = f"data/TTS/{self.HF_CONFIG_NAME.split('_')[0]}/"
        dataset = {"train": [], "validation": [], "test": []}

        for repo_path in repo_files:
            if not repo_path.startswith(prefix) or not repo_path.endswith(".parquet"):
                continue

            filename = os.path.basename(repo_path)
            split_name = next(
                (name for name in dataset if re.search(fr"{name}", filename)),
                None,
            )
            if split_name is None:
                continue

            local_path = hf_hub_download(
                repo_id=self.HF_DATASET_NAME,
                repo_type="dataset",
                filename=repo_path,
            )
            dataset[split_name].append(local_path)

        return {
            split_name: sorted(split_paths)
            for split_name, split_paths in dataset.items()
            if split_paths
        }

    def _process_split(
        self, split_data, split_name: str
    ) -> List[AudioSample]:
        """Process a single dataset split."""
        samples = []
        idx = 0

        if split_data and isinstance(split_data, list):
            for parquet_path in split_data:
                rows = pq.read_table(parquet_path).to_pylist()
                for item in tqdm(rows, desc=f"  {split_name}"):
                    try:
                        sample = self._process_single_item(item, split_name, idx)
                        if sample is not None:
                            samples.append(sample)
                    except Exception as e:
                        logger.warning(f"  Skipping {split_name}_{idx}: {e}")
                    finally:
                        idx += 1
            return samples

        for item in tqdm(split_data, desc=f"  {split_name}"):
            try:
                sample = self._process_single_item(item, split_name, idx)
                if sample is not None:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"  Skipping {split_name}_{idx}: {e}")
            finally:
                idx += 1

        return samples

    def _process_single_item(
        self, item: dict, split_name: str, idx: int
    ) -> Optional[AudioSample]:
        """Process a single audio-text item."""
        # Extract fields from WaxalNLP format
        audio_data = item.get("audio", {})
        text = item.get("text", "").strip()
        speaker_id = item.get("speaker_id", "unknown")
        gender = item.get("gender", "")
        sample_id = item.get("id", f"{split_name}_{idx}")

        if not text:
            return None

        # Load audio from the datasets cache path when decode=False is used.
        if isinstance(audio_data, dict) and audio_data.get("bytes") is not None:
            audio_array, orig_sr = sf.read(
                io.BytesIO(audio_data["bytes"]), dtype="float32"
            )
            if audio_array.ndim == 1:
                audio_array = audio_array[:, None]
            waveform = torch.from_numpy(audio_array.T)
        elif (
            isinstance(audio_data, dict)
            and audio_data.get("path")
            and os.path.exists(audio_data["path"])
        ):
            audio_array, orig_sr = sf.read(audio_data["path"], dtype="float32")
            if audio_array.ndim == 1:
                audio_array = audio_array[:, None]
            waveform = torch.from_numpy(audio_array.T)
        elif isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = np.array(audio_data["array"], dtype=np.float32)
            orig_sr = audio_data["sample_rate"]
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
        else:
            return None

        # Convert to mono before further processing.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            waveform = resampler(waveform)

        # Trim silence
        if self.trim:
            waveform = trim_silence(waveform, sr=self.sample_rate)

        # Check duration
        duration = waveform.shape[1] / self.sample_rate
        if duration < self.min_duration or duration > self.max_duration:
            return None

        # Normalize loudness
        if self.normalize:
            waveform = normalize_audio(waveform)

        # Normalize text
        normalized_text = normalize_swahili_text(text)

        # Save wav
        wav_filename = f"{sample_id}.wav"
        wav_path = self.wavs_dir / wav_filename
        sf.write(
            str(wav_path),
            waveform.squeeze(0).cpu().numpy(),
            self.sample_rate,
        )

        return AudioSample(
            id=sample_id,
            audio_path=str(wav_path),
            text=normalized_text,
            duration=duration,
            speaker_id=speaker_id,
            gender=gender,
        )

    def _save_metadata(self, samples: List[AudioSample], split_name: str) -> None:
        """Save metadata CSV for a split."""
        meta_path = self.output_dir / f"{split_name}_metadata.csv"
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["id", "audio_path", "text", "duration", "speaker_id", "gender"])
            for s in samples:
                writer.writerow([s.id, s.audio_path, s.text, f"{s.duration:.3f}", s.speaker_id, s.gender])
        logger.info(f"  Saved: {meta_path}")

    def _create_f5tts_metadata(self) -> None:
        """
        Create the unified metadata.csv in F5-TTS format:
        audio_file|text
        """
        f5_meta_path = self.output_dir / "metadata.csv"
        all_entries = []

        # Merge train + validation for training
        for split_file in ["train_metadata.csv", "validation_metadata.csv"]:
            split_path = self.output_dir / split_file
            if split_path.exists():
                with open(split_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="|")
                    for row in reader:
                        rel_path = os.path.relpath(row["audio_path"], str(self.output_dir))
                        all_entries.append((rel_path, row["text"]))

        with open(f5_meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow(["audio_file", "text"])
            for audio_file, text in all_entries:
                writer.writerow([audio_file, text])

        logger.info(f"  F5-TTS metadata: {f5_meta_path} ({len(all_entries)} entries)")

        # Also create test metadata separately
        test_path = self.output_dir / "test_metadata.csv"
        if test_path.exists():
            test_f5_path = self.output_dir / "metadata_test.csv"
            test_entries = []
            with open(test_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="|")
                for row in reader:
                    rel_path = os.path.relpath(row["audio_path"], str(self.output_dir))
                    test_entries.append((rel_path, row["text"]))
            with open(test_f5_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="|")
                writer.writerow(["audio_file", "text"])
                for audio_file, text in test_entries:
                    writer.writerow([audio_file, text])
            logger.info(f"  Test metadata: {test_f5_path} ({len(test_entries)} entries)")

    def _update_stats(self, stats: DatasetStats, samples: List[AudioSample]) -> None:
        """Update running statistics."""
        for s in samples:
            stats.total_samples += 1
            stats.total_duration_hours += s.duration / 3600
            stats.min_duration_sec = min(stats.min_duration_sec, s.duration)
            stats.max_duration_sec = max(stats.max_duration_sec, s.duration)

            # Speaker stats
            if s.speaker_id not in stats.speakers:
                stats.speakers[s.speaker_id] = 0.0
            stats.speakers[s.speaker_id] += s.duration / 3600

            # Gender stats
            g = s.gender or "unknown"
            stats.gender_distribution[g] = stats.gender_distribution.get(g, 0) + 1

            # Duration histogram (1-second bins)
            bin_idx = min(int(s.duration), 29)
            stats.duration_histogram[bin_idx] += 1

        if stats.total_samples > 0:
            stats.avg_duration_sec = (
                stats.total_duration_hours * 3600 / stats.total_samples
            )
        stats.num_speakers = len(stats.speakers)

    def _save_stats(self, stats: DatasetStats) -> None:
        """Save dataset statistics to JSON."""
        stats_dict = {
            "total_samples": stats.total_samples,
            "total_duration_hours": round(stats.total_duration_hours, 2),
            "avg_duration_sec": round(stats.avg_duration_sec, 2),
            "min_duration_sec": round(stats.min_duration_sec, 2),
            "max_duration_sec": round(stats.max_duration_sec, 2),
            "num_speakers": stats.num_speakers,
            "speakers": {k: round(v, 2) for k, v in stats.speakers.items()},
            "gender_distribution": stats.gender_distribution,
            "duration_histogram_1s_bins": stats.duration_histogram,
        }
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2)
        logger.info(f"  Stats: {stats_path}")


# =============================================================================
# F5-TTS Dataset Preparation Script
# =============================================================================


def prepare_f5tts_format(
    metadata_csv: str,
    output_dir: str,
    vocab_path: Optional[str] = None,
) -> str:
    """
    Convert our metadata CSV to F5-TTS training format.
    Copies vocab.txt from pretrained model if provided.
    Returns path to prepared dataset directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy vocab.txt from pretrained F5-TTS
    # This is CRITICAL — must reuse pretrained vocab for fine-tuning
    if vocab_path and os.path.exists(vocab_path):
        import shutil
        dest_vocab = output_path / "vocab.txt"
        shutil.copy2(vocab_path, dest_vocab)
        logger.info(f"Copied pretrained vocab: {dest_vocab}")
    else:
        logger.warning(
            "No vocab_path provided. Will attempt to download from "
            "F5-TTS pretrained model. For fine-tuning, always reuse "
            "the pretrained vocab.txt!"
        )

    # Build F5-TTS raw.arrow records with absolute audio paths.
    records = []
    durations = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            audio_file = row["audio_file"]
            full_path = os.path.abspath(
                os.path.join(os.path.dirname(metadata_csv), audio_file)
            )
            if os.path.exists(full_path):
                dur = get_audio_duration(full_path)
                dur = round(dur, 4)
                records.append(
                    {
                        "audio_path": full_path,
                        "text": row["text"],
                        "duration": dur,
                    }
                )
                durations.append(dur)

    raw_arrow_path = output_path / "raw.arrow"
    with ArrowWriter(path=str(raw_arrow_path)) as writer:
        for record in records:
            writer.write(record)
        writer.finalize()

    dur_path = output_path / "duration.json"
    with open(dur_path, "w") as f:
        json.dump({"duration": durations}, f, indent=2)

    logger.info(
        f"Prepared F5-TTS dataset at {output_path} "
        f"({len(records)} samples)"
    )
    return str(output_path)

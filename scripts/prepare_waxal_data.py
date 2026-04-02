#!/usr/bin/env python3
"""
Sauti TTS — WaxalNLP Swahili Dataset Preparation Script

Downloads the Google WaxalNLP Swahili TTS dataset from HuggingFace,
preprocesses audio (resample, normalize, trim), and creates F5-TTS
compatible metadata format.

Usage:
    python scripts/prepare_waxal_data.py \
        --output_dir data/waxal_swahili \
        --sample_rate 24000 \
        --min_duration 1.0 \
        --max_duration 30.0

Requirements:
    pip install datasets torchaudio pyloudnorm
"""

import os
import sys
import argparse
import logging
import json
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sauti_tts.utils import setup_logging
from sauti_tts.data import WaxalSwahiliDataset, prepare_f5tts_format

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WaxalNLP Swahili TTS dataset for F5-TTS fine-tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/waxal_swahili",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Target sample rate (F5-TTS uses 24000)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip loudness normalization",
    )
    parser.add_argument(
        "--no_trim",
        action="store_true",
        help="Skip silence trimming",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to pretrained vocab.txt (will try to download if not provided)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("  Sauti TTS — Data Preparation Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Output:      {args.output_dir}")
    logger.info(f"  Sample rate: {args.sample_rate} Hz")
    logger.info(f"  Duration:    {args.min_duration}-{args.max_duration}s")
    logger.info(f"  Normalize:   {not args.no_normalize}")
    logger.info(f"  Trim:        {not args.no_trim}")
    logger.info("=" * 60)

    # Step 1: Download and preprocess WaxalNLP
    dataset = WaxalSwahiliDataset(
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        normalize=not args.no_normalize,
        trim=not args.no_trim,
    )
    stats = dataset.download_and_prepare()

    # Step 2: Prepare F5-TTS format
    logger.info("\nPreparing F5-TTS format...")

    # Try to get vocab.txt
    vocab_path = args.vocab_path
    if vocab_path is None:
        # Try downloading from F5-TTS HuggingFace
        try:
            from huggingface_hub import hf_hub_download

            logger.info("Downloading pretrained vocab.txt from F5-TTS...")
            vocab_path = hf_hub_download(
                repo_id="SWivid/F5-TTS",
                filename="data/Emilia_ZH_EN_pinyin/vocab.txt",
                local_dir="ckpts",
            )
            logger.info(f"Downloaded vocab: {vocab_path}")
        except Exception as e:
            logger.warning(
                f"Could not download vocab.txt: {e}\n"
                "You can manually copy it from F5-TTS pretrained model.\n"
                "Fine-tuning REQUIRES the pretrained vocab.txt!"
            )

    # Copy vocab to dataset directory
    if vocab_path and os.path.exists(vocab_path):
        dest = os.path.join(args.output_dir, "vocab.txt")
        shutil.copy2(vocab_path, dest)
        logger.info(f"Vocab copied to: {dest}")

    # Create duration.json and ensure F5-TTS expected artifacts exist.
    metadata_csv = os.path.join(args.output_dir, "metadata.csv")
    if os.path.exists(metadata_csv):
        prepare_f5tts_format(
            metadata_csv=metadata_csv,
            output_dir=args.output_dir,
            vocab_path=vocab_path,
        )
    else:
        logger.warning(
            "metadata.csv not found after preprocessing; skipped F5-TTS format prep"
        )

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("  Preparation Complete!")
    logger.info("=" * 60)
    logger.info(f"  Samples: {stats.total_samples}")
    logger.info(f"  Duration: {stats.total_duration_hours:.2f} hours")
    logger.info(f"  Speakers: {stats.num_speakers}")
    logger.info(f"\n  Next step: Fine-tune with:")
    logger.info(f"    python scripts/train.py --config configs/finetune_single_gpu.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

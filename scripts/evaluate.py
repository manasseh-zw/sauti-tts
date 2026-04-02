#!/usr/bin/env python3
"""
Sauti TTS — Evaluation Script

Evaluates generated speech quality using multiple metrics:
- MOS estimation (UTMOS)
- Speaker similarity (ECAPA-TDNN cosine similarity)
- Intelligibility (Whisper CER/WER for Swahili)
- Signal quality (PESQ, STOI)

Usage:
    # Evaluate against test set
    python scripts/evaluate.py \
        --checkpoint ckpts/sauti_tts/model_last.pt \
        --test_set data/waxal_swahili/test_metadata.csv \
        --ref_audio data/reference/sample.wav \
        --output_dir outputs/eval/

    # Evaluate pre-generated audio
    python scripts/evaluate.py \
        --gen_dir outputs/batch/ \
        --ref_dir data/waxal_swahili/wavs/ \
        --test_set data/waxal_swahili/test_metadata.csv \
        --output_dir outputs/eval/
"""

import os
import sys
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sauti_tts.utils import setup_logging, load_audio
from sauti_tts.metrics import SautiEvaluator, TTSEvalResult, TTSEvalSummary

logger = logging.getLogger(__name__)


def load_test_metadata(metadata_path: str) -> List[dict]:
    """Load test set metadata."""
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            samples.append(row)
    return samples


def evaluate_from_checkpoint(
    checkpoint_path: str,
    test_metadata_path: str,
    ref_audio_path: str,
    output_dir: str,
    max_samples: int = 50,
    nfe_steps: int = 32,
    vocab_path: Optional[str] = None,
):
    """Generate + evaluate from a model checkpoint."""
    from scripts.inference import SautiInference

    logger.info("=" * 60)
    logger.info("  Sauti TTS — Generation + Evaluation")
    logger.info("=" * 60)

    # Load model
    engine = SautiInference(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
    )

    # Load test data
    test_samples = load_test_metadata(test_metadata_path)[:max_samples]
    logger.info(f"Evaluating {len(test_samples)} test samples...")

    # Setup output
    gen_dir = os.path.join(output_dir, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    # Generate and evaluate
    evaluator = SautiEvaluator()
    results = []

    for i, sample in enumerate(test_samples):
        sample_id = sample.get("id", f"test_{i}")
        text = sample.get("text", "")
        ref_path = sample.get("audio_path", "")

        logger.info(f"[{i+1}/{len(test_samples)}] {text[:50]}...")

        try:
            # Generate
            gen_path = os.path.join(gen_dir, f"{sample_id}.wav")
            gen_audio = engine.generate(
                text=text,
                ref_audio_path=ref_audio_path,
                output_path=gen_path,
                nfe_steps=nfe_steps,
            )

            # Load reference for comparison
            if os.path.exists(ref_path):
                ref_waveform, _ = load_audio(ref_path, target_sr=24000)
                ref_audio_np = ref_waveform.squeeze().numpy()
            else:
                ref_audio_np = np.zeros(1)

            # Evaluate
            result = evaluator.evaluate_sample(
                ref_audio=ref_audio_np,
                gen_audio=gen_audio,
                ref_text=text,
                sample_id=sample_id,
            )
            results.append(result)

            logger.info(
                f"  MOS: {result.mos_score:.2f} | "
                f"SpkSim: {result.speaker_similarity:.3f} | "
                f"CER: {result.cer:.3f}"
            )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Aggregate results
    summary = evaluator.evaluate_batch(results)
    evaluator.print_summary(summary)

    # Save results
    save_evaluation_results(summary, output_dir)

    return summary


def evaluate_pregenerated(
    gen_dir: str,
    ref_dir: str,
    test_metadata_path: str,
    output_dir: str,
    max_samples: int = 50,
):
    """Evaluate pre-generated audio files."""
    logger.info("=" * 60)
    logger.info("  Sauti TTS — Evaluation (Pre-generated)")
    logger.info("=" * 60)

    test_samples = load_test_metadata(test_metadata_path)[:max_samples]
    evaluator = SautiEvaluator()
    results = []

    for i, sample in enumerate(test_samples):
        sample_id = sample.get("id", f"test_{i}")
        text = sample.get("text", "")

        gen_path = os.path.join(gen_dir, f"{sample_id}.wav")
        ref_path = sample.get("audio_path", "")

        if not os.path.exists(gen_path):
            logger.warning(f"Generated audio not found: {gen_path}")
            continue

        try:
            # Load audios
            gen_waveform, _ = load_audio(gen_path, target_sr=24000)
            gen_audio_np = gen_waveform.squeeze().numpy()

            if os.path.exists(ref_path):
                ref_waveform, _ = load_audio(ref_path, target_sr=24000)
                ref_audio_np = ref_waveform.squeeze().numpy()
            else:
                ref_audio_np = np.zeros(1)

            # Evaluate
            result = evaluator.evaluate_sample(
                ref_audio=ref_audio_np,
                gen_audio=gen_audio_np,
                ref_text=text,
                sample_id=sample_id,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed for {sample_id}: {e}")
            continue

    summary = evaluator.evaluate_batch(results)
    evaluator.print_summary(summary)
    save_evaluation_results(summary, output_dir)

    return summary


def save_evaluation_results(summary: TTSEvalSummary, output_dir: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary JSON
    summary_dict = {
        "num_samples": summary.num_samples,
        "mean_mos": round(summary.mean_mos, 3),
        "std_mos": round(summary.std_mos, 3),
        "mean_speaker_similarity": round(summary.mean_speaker_sim, 3),
        "mean_cer": round(summary.mean_cer, 4),
        "mean_wer": round(summary.mean_wer, 4),
        "mean_pesq": round(summary.mean_pesq, 3),
        "mean_stoi": round(summary.mean_stoi, 3),
        "mean_duration_ratio": round(summary.mean_duration_ratio, 3),
    }
    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Save per-sample CSV
    csv_path = os.path.join(output_dir, "eval_details.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "ref_text", "gen_text",
            "mos", "speaker_sim", "cer", "wer",
            "pesq", "stoi", "duration_ratio",
        ])
        for r in summary.results:
            writer.writerow([
                r.sample_id, r.ref_text, r.gen_text,
                f"{r.mos_score:.3f}", f"{r.speaker_similarity:.3f}",
                f"{r.cer:.4f}", f"{r.wer:.4f}",
                f"{r.pesq_score:.3f}", f"{r.stoi_score:.3f}",
                f"{r.duration_ratio:.3f}",
            ])

    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sauti TTS Evaluation")

    parser.add_argument("--checkpoint", type=str, help="Model checkpoint path")
    parser.add_argument("--test_set", type=str, required=True, help="Test metadata CSV")
    parser.add_argument("--ref_audio", type=str, help="Reference audio for generation")
    parser.add_argument("--gen_dir", type=str, help="Pre-generated audio directory")
    parser.add_argument("--ref_dir", type=str, help="Reference audio directory")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--whisper_model", type=str, default="medium")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.gen_dir:
        # Evaluate pre-generated audio
        evaluate_pregenerated(
            gen_dir=args.gen_dir,
            ref_dir=args.ref_dir or "",
            test_metadata_path=args.test_set,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )
    elif args.checkpoint:
        # Generate + evaluate
        evaluate_from_checkpoint(
            checkpoint_path=args.checkpoint,
            test_metadata_path=args.test_set,
            ref_audio_path=args.ref_audio,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            nfe_steps=args.steps,
            vocab_path=args.vocab_path,
        )
    else:
        parser.error("Provide either --checkpoint (generate+eval) or --gen_dir (eval only)")


if __name__ == "__main__":
    main()

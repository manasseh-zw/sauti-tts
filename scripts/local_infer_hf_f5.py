#!/usr/bin/env python3
"""
Run local inference against a published Hugging Face F5-TTS checkpoint.

This script downloads the public model artifacts from the Hugging Face Hub,
reuses the repo's `SautiInference` engine, and writes both a generated WAV and
an inference summary to disk for quick local benchmarking.

Example:
  python scripts/local_infer_hf_f5.py \
    --repo-id manassehzw/sna-f5-tts \
    --ref-audio /path/to/reference.wav \
    --ref-text "Mangwanani. Ndamuka zvakanaka nhasi." \
    --text "Mhoro shamwari yangu. Wakafamba sei nezuro?" \
    --output-dir outputs/local_hf_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_REPO_ID = "manassehzw/sna-f5-tts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "local_hf_test"
DEFAULT_OUTPUT_NAME = "generated.wav"
DEFAULT_SUMMARY_NAME = "inference_summary.json"
DEFAULT_CACHE_DIR = REPO_ROOT / ".hf-cache"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference from a Hugging Face F5-TTS model repo."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo ID.")
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision to download from.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--text",
        default="Mangwanani. Ndamuka zvakanaka nhasi, ndirikugadzirira kuenda kusvondo.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        required=True,
        help="Reference WAV path used for voice-cloning style synthesis.",
    )
    parser.add_argument(
        "--ref-text",
        default="",
        help="Transcript for the reference audio. Strongly recommended for quality.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated outputs into.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Output WAV filename.",
    )
    parser.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
        help="JSON summary filename.",
    )
    parser.add_argument("--nfe-steps", type=int, default=24, help="Diffusion steps.")
    parser.add_argument("--cfg-strength", type=float, default=2.0, help="CFG strength.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier.")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Optional generation seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (e.g. mps, cpu, cuda). Defaults to auto-detect.",
    )
    return parser.parse_args()


def _download_model(repo_id: str, revision: str, cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        allow_patterns=["model.pt", "vocab.txt", "README.md"],
    )
    return Path(snapshot_path)


def main() -> None:
    args = _parse_args()

    try:
        import soundfile as sf
        from scripts.inference import SautiInference
    except ImportError as exc:
        raise ImportError(
            "Local inference dependencies are missing. Install the repo requirements "
            "and the upstream F5-TTS package before running this script."
        ) from exc

    if not args.ref_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = _download_model(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir.resolve(),
    )

    checkpoint_path = snapshot_dir / "model.pt"
    vocab_path = snapshot_dir / "vocab.txt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint missing: {checkpoint_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Downloaded vocab missing: {vocab_path}")

    engine = SautiInference(
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(vocab_path),
        use_ema=False,
        device=args.device,
    )

    output_path = output_dir / args.output_name
    started_at = time.time()
    audio = engine.generate(
        text=args.text,
        ref_audio_path=str(args.ref_audio.resolve()),
        ref_text=args.ref_text,
        output_path=str(output_path),
        nfe_steps=args.nfe_steps,
        cfg_strength=args.cfg_strength,
        speed=args.speed,
        seed=args.seed,
    )
    elapsed = time.time() - started_at
    info = sf.info(str(output_path))
    duration_seconds = float(info.frames) / float(info.samplerate)
    rtf = elapsed / duration_seconds if duration_seconds > 0 else None

    summary = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "checkpoint_path": str(checkpoint_path),
        "vocab_path": str(vocab_path),
        "device": engine.device,
        "text": args.text,
        "ref_audio": str(args.ref_audio.resolve()),
        "ref_text": args.ref_text,
        "nfe_steps": args.nfe_steps,
        "cfg_strength": args.cfg_strength,
        "speed": args.speed,
        "seed": args.seed,
        "output_path": str(output_path),
        "elapsed_seconds": elapsed,
        "audio_duration_seconds": duration_seconds,
        "rtf": rtf,
        "num_samples": int(len(audio)),
        "sample_rate": int(info.samplerate),
    }

    summary_path = output_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("LOCAL F5-TTS INFERENCE COMPLETE")
    print("=" * 72)
    print(f"Repo:        {args.repo_id}@{args.revision}")
    print(f"Device:      {engine.device}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Output WAV:  {output_path}")
    print(f"Summary:     {summary_path}")
    print(f"Elapsed:     {elapsed:.2f}s")
    print(f"Audio len:   {duration_seconds:.2f}s")
    if rtf is not None:
        print(f"RTF:         {rtf:.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

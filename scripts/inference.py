#!/usr/bin/env python3
"""
Sauti TTS — Inference Script

Generate Swahili speech from text using fine-tuned Sauti TTS model.
Supports zero-shot voice cloning with reference audio.

Usage:
    # Single text
    python scripts/inference.py \
        --text "Habari, karibu kwenye Sauti TTS" \
        --ref_audio data/reference/sample.wav \
        --output outputs/generated.wav

    # Batch inference from file
    python scripts/inference.py \
        --text_file data/test_sentences.txt \
        --ref_audio data/reference/sample.wav \
        --output_dir outputs/batch/

    # Using F5-TTS CLI (alternative)
    f5-tts_infer-cli --model F5TTS_v1_Base \
        --ckpt_file ckpts/sauti_tts/model_last.pt \
        --ref_audio ref.wav \
        --ref_text "Reference text" \
        --gen_text "Text to generate"
"""

import os
import sys
import argparse
import logging
import time
import inspect
from pathlib import Path
from typing import Optional, List

import torch
import torchaudio
import numpy as np
import yaml
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sauti_tts.utils import (
    setup_logging,
    normalize_swahili_text,
    load_audio,
    save_audio,
    seed_everything,
)

logger = logging.getLogger(__name__)


def _patch_torchaudio_load_with_soundfile() -> None:
    """
    Force a soundfile-backed loader for environments where torchaudio delegates
    to torchcodec and the native shared libraries are unavailable.
    """
    if getattr(torchaudio.load, "_sauti_soundfile_patch", False):
        return

    def _load_with_soundfile(path: str, *args, **kwargs):
        audio_np, sample_rate = sf.read(path, dtype="float32")
        if audio_np.ndim == 1:
            audio_np = audio_np[:, None]
        waveform = torch.from_numpy(audio_np.T.copy())
        return waveform, sample_rate

    _load_with_soundfile._sauti_soundfile_patch = True
    torchaudio.load = _load_with_soundfile


def _load_vocoder_compat(device: str):
    from f5_tts.infer.utils_infer import load_vocoder as f5_load_vocoder

    load_vocoder_kwargs = {
        "vocoder_name": "vocos",
        "device": device,
    }
    load_vocoder_signature = inspect.signature(f5_load_vocoder)
    load_vocoder_kwargs = {
        key: value
        for key, value in load_vocoder_kwargs.items()
        if key in load_vocoder_signature.parameters
    }
    return f5_load_vocoder(**load_vocoder_kwargs)


def _apply_edge_fades(
    audio: np.ndarray,
    sample_rate: int,
    fade_in_sec: float = 0.005,
    fade_out_sec: float = 0.03,
) -> np.ndarray:
    """
    Add tiny edge fades to avoid clicks from abrupt waveform boundaries.
    """
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    result = audio.astype(np.float32, copy=True)

    fade_in_samples = min(len(result), max(0, int(sample_rate * fade_in_sec)))
    fade_out_samples = min(len(result), max(0, int(sample_rate * fade_out_sec)))

    if fade_in_samples > 1:
        result[:fade_in_samples] *= np.linspace(
            0.0, 1.0, fade_in_samples, dtype=np.float32
        )

    if fade_out_samples > 1:
        result[-fade_out_samples:] *= np.linspace(
            1.0, 0.0, fade_out_samples, dtype=np.float32
        )

    return result


def _smooth_quiet_pauses(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.01,
    min_silence_sec: float = 0.025,
    merge_gap_sec: float = 0.012,
    fade_sec: float = 0.01,
    zero_cross_search_sec: float = 0.008,
) -> np.ndarray:
    """
    Clean low-energy pause regions with short fades so tiny clicks inside pauses
    do not survive into the final waveform.
    """
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    source = audio.astype(np.float32, copy=False)
    result = source.copy()
    rms_window = max(1, int(sample_rate * 0.01))
    min_silence_samples = max(1, int(sample_rate * min_silence_sec))
    merge_gap_samples = max(0, int(sample_rate * merge_gap_sec))
    fade_samples = max(1, int(sample_rate * fade_sec))
    zero_cross_search = max(1, int(sample_rate * zero_cross_search_sec))

    rms = np.sqrt(
        np.convolve(source**2, np.ones(rms_window, dtype=np.float32) / rms_window, mode="same")
    )
    quiet = rms < rms_threshold

    def _nearest_zero_crossing(center: int) -> int:
        if len(source) < 2:
            return center

        lo = max(1, center - zero_cross_search)
        hi = min(len(source) - 1, center + zero_cross_search)
        window = source[lo - 1 : hi + 1]
        zero_crossings = np.where(np.signbit(window[:-1]) != np.signbit(window[1:]))[0]

        if zero_crossings.size:
            candidates = []
            for rel_idx in zero_crossings:
                left_idx = lo - 1 + int(rel_idx)
                right_idx = min(left_idx + 1, len(source) - 1)
                candidate = (
                    left_idx
                    if abs(source[left_idx]) <= abs(source[right_idx])
                    else right_idx
                )
                candidates.append(candidate)
            return min(
                candidates,
                key=lambda idx: (abs(idx - center), abs(source[idx])),
            )

        nearest = lo + int(np.argmin(np.abs(source[lo : hi + 1])))
        return nearest

    segments: list[list[int]] = []
    start = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            if i - start >= min_silence_samples:
                segments.append([start, i])
            start = None
    if start is not None and len(result) - start >= min_silence_samples:
        segments.append([start, len(result)])

    if not segments:
        return result

    merged: list[tuple[int, int]] = []
    current_start, current_end = segments[0]
    for next_start, next_end in segments[1:]:
        if next_start - current_end <= merge_gap_samples:
            current_end = next_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    for start, end in merged:
        snapped_start = _nearest_zero_crossing(start)
        snapped_end = _nearest_zero_crossing(end)

        if snapped_end <= snapped_start:
            snapped_start, snapped_end = start, end

        fade_in_start = max(0, snapped_start - fade_samples)
        fade_out_end = min(len(result), snapped_end + fade_samples)

        if snapped_start > fade_in_start:
            result[fade_in_start:snapped_start] *= np.linspace(
                1.0, 0.0, snapped_start - fade_in_start, dtype=np.float32
            )

        result[snapped_start:snapped_end] = 0.0

        if fade_out_end > snapped_end:
            result[snapped_end:fade_out_end] *= np.linspace(
                0.0, 1.0, fade_out_end - snapped_end, dtype=np.float32
            )

    return result


def _repair_quiet_micro_clicks(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.015,
    spike_threshold: float = 0.018,
    max_click_sec: float = 0.0002,
    search_window_sec: float = 0.004,
) -> np.ndarray:
    """
    Repair tiny impulse-like clicks in low-energy regions by interpolating over
    very short spike runs. This is intentionally conservative and only touches
    samples inside quiet areas.
    """
    if audio.ndim != 1 or len(audio) < 5:
        return audio

    result = audio.astype(np.float32, copy=True)
    rms_window = max(1, int(sample_rate * 0.005))
    max_click_samples = max(1, int(sample_rate * max_click_sec))
    search_window = max(2, int(sample_rate * search_window_sec))

    rms = np.sqrt(
        np.convolve(result**2, np.ones(rms_window, dtype=np.float32) / rms_window, mode="same")
    )
    quiet = rms < rms_threshold

    diff_prev = np.abs(result[1:-1] - result[:-2])
    diff_next = np.abs(result[1:-1] - result[2:])
    interp_dev = np.abs(result[1:-1] - 0.5 * (result[:-2] + result[2:]))

    candidate_core = (
        quiet[1:-1]
        & ((diff_prev > spike_threshold) | (diff_next > spike_threshold))
        & (interp_dev > spike_threshold * 0.6)
    )
    candidate_idx = np.where(candidate_core)[0] + 1

    if candidate_idx.size == 0:
        return result

    runs: list[tuple[int, int]] = []
    start = int(candidate_idx[0])
    prev = start
    for idx in candidate_idx[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = idx
        prev = idx
    runs.append((start, prev + 1))

    for start, end in runs:
        if end - start > max_click_samples:
            continue

        left = max(0, start - search_window)
        right = min(len(result), end + search_window)

        left_candidates = np.where(~quiet[left:start])[0]
        right_candidates = np.where(~quiet[end:right])[0]

        anchor_left = (
            left + int(left_candidates[-1])
            if left_candidates.size
            else max(0, start - 1)
        )
        anchor_right = (
            end + int(right_candidates[0])
            if right_candidates.size
            else min(len(result) - 1, end)
        )

        if anchor_right <= anchor_left:
            continue

        repair_start = max(anchor_left + 1, start)
        repair_end = min(anchor_right, end)
        if repair_end <= repair_start:
            continue

        interp = np.linspace(
            result[anchor_left],
            result[anchor_right],
            anchor_right - anchor_left + 1,
            dtype=np.float32,
        )
        result[repair_start:repair_end] = interp[
            repair_start - anchor_left : repair_end - anchor_left
        ]

    return result


def _trim_chunk_edges(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.01,
    window_sec: float = 0.01,
    pad_sec: float = 0.035,
    min_trim_sec: float = 0.02,
) -> np.ndarray:
    """
    Remove low-energy padding at the start/end of chunked generations before
    cross-fading them together. Keep a small pad so consonant onsets/offsets
    are not clipped.
    """
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    source = audio.astype(np.float32, copy=False)
    window = max(1, int(sample_rate * window_sec))
    pad = max(0, int(sample_rate * pad_sec))
    min_trim = max(0, int(sample_rate * min_trim_sec))

    rms = np.sqrt(
        np.convolve(source**2, np.ones(window, dtype=np.float32) / window, mode="same")
    )
    active = rms > rms_threshold
    if not np.any(active):
        return source.copy()

    start = int(np.argmax(active))
    end = len(source) - int(np.argmax(active[::-1]))

    start = max(0, start - pad)
    end = min(len(source), end + pad)

    if start < min_trim:
        start = 0
    if len(source) - end < min_trim:
        end = len(source)

    if end <= start:
        return source.copy()

    return source[start:end].copy()


def _gate_quiet_regions(
    audio: np.ndarray,
    sample_rate: int,
    rms_threshold: float = 0.017,
    min_silence_sec: float = 0.02,
    merge_gap_sec: float = 0.015,
    fade_sec: float = 0.012,
) -> np.ndarray:
    """
    Force very quiet pause regions to clean silence with short fades.
    This is more aggressive than de-clicking and is intended as the final
    cleanup pass for residual pause artifacts.
    """
    if audio.ndim != 1 or len(audio) == 0:
        return audio

    result = audio.astype(np.float32, copy=True)
    rms_window = max(1, int(sample_rate * 0.01))
    min_silence_samples = max(1, int(sample_rate * min_silence_sec))
    merge_gap_samples = max(0, int(sample_rate * merge_gap_sec))
    fade_samples = max(1, int(sample_rate * fade_sec))

    rms = np.sqrt(
        np.convolve(result**2, np.ones(rms_window, dtype=np.float32) / rms_window, mode="same")
    )
    quiet = rms < rms_threshold

    segments: list[list[int]] = []
    start = None
    for i, is_quiet in enumerate(quiet):
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            if i - start >= min_silence_samples:
                segments.append([start, i])
            start = None
    if start is not None and len(result) - start >= min_silence_samples:
        segments.append([start, len(result)])

    if not segments:
        return result

    merged: list[tuple[int, int]] = []
    current_start, current_end = segments[0]
    for next_start, next_end in segments[1:]:
        if next_start - current_end <= merge_gap_samples:
            current_end = next_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    for start, end in merged:
        fade_in_start = max(0, start - fade_samples)
        fade_out_end = min(len(result), end + fade_samples)

        if start > fade_in_start:
            result[fade_in_start:start] *= np.linspace(
                1.0, 0.0, start - fade_in_start, dtype=np.float32
            )

        result[start:end] = 0.0

        if fade_out_end > end:
            result[end:fade_out_end] *= np.linspace(
                0.0, 1.0, fade_out_end - end, dtype=np.float32
            )

    return result


class SautiInference:
    """
    Inference engine for Sauti TTS.
    Wraps F5-TTS inference utilities with Swahili-specific processing.
    """

    def __init__(
        self,
        checkpoint_path: str = "ckpts/sauti_tts/model_last.pt",
        model_type: str = "F5TTS_v1_Base",
        vocab_path: Optional[str] = None,
        use_ema: bool = True,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.use_ema = use_ema
        self.vocab_path = vocab_path

        self.model = None
        self.vocoder = None

        self._load_model()

    def _load_model(self):
        """Load the fine-tuned model and vocoder."""
        logger.info(f"Loading Sauti TTS model from {self.checkpoint_path}...")

        try:
            from f5_tts.infer.utils_infer import (
                load_model,
                load_vocoder,
            )
            from f5_tts.model import DiT

            # Load vocoder
            self.vocoder = _load_vocoder_compat(self.device)
            logger.info("vocos vocoder loaded")

            # Load model
            load_model_kwargs = {
                "model_cls": DiT,
                "model_cfg": dict(
                    dim=1024,
                    depth=22,
                    heads=16,
                    ff_mult=2,
                    text_dim=512,
                    conv_layers=4,
                ),
                "ckpt_path": self.checkpoint_path,
                "mel_spec_type": "vocos",
                "vocab_file": self.vocab_path or "",
                "is_local": True,
                "use_ema": self.use_ema,
                "device": self.device,
            }
            load_model_signature = inspect.signature(load_model)
            load_model_kwargs = {
                key: value
                for key, value in load_model_kwargs.items()
                if key in load_model_signature.parameters
            }

            try:
                self.model = load_model(**load_model_kwargs)
                logger.info("Model loaded successfully")
            except RuntimeError as e:
                logger.warning(
                    "Upstream F5 checkpoint loader could not load this checkpoint format. "
                    "Falling back to repo-local checkpoint loading."
                )
                logger.debug("Upstream loader failure detail: %s", e)
                self._load_model_local()

        except ImportError as e:
            logger.error(
                f"F5-TTS not available: {e}\n"
                "Install from source:\n"
                "  git clone https://github.com/SWivid/F5-TTS.git\n"
                "  cd F5-TTS && pip install -e ."
            )
            raise

    def _load_model_local(self):
        """Load repo-local checkpoints, including LoRA fine-tune outputs."""
        from sauti_tts.model import SautiTTS, SautiTTSConfig

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        use_lora = any("lora" in key.lower() for key in state_dict.keys())

        pretrained_path = os.path.join(
            "ckpts",
            "F5TTS_v1_Base",
            "model_1250000.safetensors",
        )
        config = SautiTTSConfig(
            model_type=self.model_type,
            pretrained_path=pretrained_path,
            vocab_path=self.vocab_path,
            use_lora=use_lora,
        )

        sauti = SautiTTS(config)
        sauti.build_model()
        sauti.load_checkpoint(self.checkpoint_path)
        self.model = sauti.model
        logger.info(
            "Loaded repo-local checkpoint successfully (use_lora=%s)",
            use_lora,
        )

    def generate(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        output_path: Optional[str] = None,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate speech from Swahili text.

        Args:
            text: Swahili text to synthesize
            ref_audio_path: Path to reference audio (for voice cloning)
            ref_text: Transcription of reference audio (improves quality)
            output_path: Optional path to save generated audio
            nfe_steps: Diffusion steps (16=fast, 32=balanced, 64=best)
            cfg_strength: Classifier-free guidance
            sway_sampling_coef: Sway sampling
            speed: Speech rate
            seed: Random seed

        Returns:
            Generated audio as numpy array (24kHz)
        """
        from f5_tts.infer.utils_infer import (
            infer_process,
            preprocess_ref_audio_text,
        )
        _patch_torchaudio_load_with_soundfile()

        # Normalize Swahili text
        text = normalize_swahili_text(text)
        logger.info(f"Normalized text: {text}")

        # Set seed
        if seed is not None:
            seed_everything(seed)

        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(
            ref_audio_path, ref_text
        )

        # Generate
        start_time = time.time()

        generated_audio, sample_rate, _ = infer_process(
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            model_obj=self.model,
            vocoder=self.vocoder,
            nfe_step=nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
        )
        generated_audio = _smooth_quiet_pauses(generated_audio, sample_rate)
        generated_audio = _repair_quiet_micro_clicks(generated_audio, sample_rate)
        generated_audio = _gate_quiet_regions(generated_audio, sample_rate)

        gen_time = time.time() - start_time
        gen_duration = len(generated_audio) / sample_rate
        rtf = gen_time / gen_duration if gen_duration > 0 else 0

        logger.info(
            f"Generated: {gen_duration:.2f}s audio in {gen_time:.2f}s "
            f"(RTF: {rtf:.2f}x)"
        )

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            generated_audio = _apply_edge_fades(generated_audio, sample_rate)
            waveform = torch.from_numpy(generated_audio).unsqueeze(0)
            save_audio(waveform, output_path, sample_rate)
            logger.info(f"Saved: {output_path}")

        return generated_audio

    def generate_batch(
        self,
        texts: List[str],
        ref_audio_path: str,
        ref_text: str = "",
        output_dir: str = "outputs/batch",
        **kwargs,
    ) -> List[str]:
        """Generate speech for multiple texts."""
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []

        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"gen_{i:04d}.wav")
            try:
                self.generate(
                    text=text,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                    output_path=output_path,
                    **kwargs,
                )
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Failed to generate #{i}: {e}")
                continue

        logger.info(f"Generated {len(output_paths)}/{len(texts)} files in {output_dir}")
        return output_paths

    def generate_long(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str = "",
        output_path: Optional[str] = None,
        max_chars_per_chunk: int = 180,
        cross_fade_sec: float = 0.25,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate long speech by chunking text and cross-fading.
        F5-TTS works best with shorter inputs.
        """
        # Split into chunks at sentence boundaries
        chunks = self._split_text(text, max_chars_per_chunk)
        logger.info(f"Split into {len(chunks)} chunks")

        # Generate each chunk
        audio_chunks = []
        base_seed = kwargs.get("seed")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            chunk_kwargs = dict(kwargs)
            if base_seed is not None:
                chunk_kwargs["seed"] = base_seed + i
            audio = self.generate(
                text=chunk,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                **chunk_kwargs,
            )
            audio_chunks.append(_trim_chunk_edges(audio, 24000))

        # Stitch sentence-like chunks with a short pause bridge.
        if len(audio_chunks) == 1:
            result = audio_chunks[0]
        else:
            result = self._stitch_with_pause(
                audio_chunks,
                sample_rate=24000,
                pause_sec=min(max(cross_fade_sec * 0.45, 0.06), 0.12),
                fade_sec=min(cross_fade_sec * 0.2, 0.03),
            )

        result = _smooth_quiet_pauses(result, 24000)
        result = _repair_quiet_micro_clicks(result, 24000)
        result = _gate_quiet_regions(result, 24000)
        result = _apply_edge_fades(result, 24000)

        if output_path:
            waveform = torch.from_numpy(result).unsqueeze(0)
            save_audio(waveform, output_path, 24000)
            logger.info(f"Saved long audio: {output_path}")

        return result

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        # Swahili sentence endings
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in ".!?;" and len(current) >= 20:
                sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        # Merge short sentences into chunks
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _crossfade_concat(
        self, chunks: List[np.ndarray], fade_sec: float
    ) -> np.ndarray:
        """Concatenate audio chunks with cross-fade."""
        fade_samples = int(fade_sec * 24000)

        if fade_samples == 0 or len(chunks) <= 1:
            return np.concatenate(chunks)

        result = chunks[0]
        for chunk in chunks[1:]:
            if len(result) < fade_samples or len(chunk) < fade_samples:
                result = np.concatenate([result, chunk])
                continue

            # Create fade curves
            fade_out = np.linspace(1, 0, fade_samples)
            fade_in = np.linspace(0, 1, fade_samples)

            # Apply cross-fade
            result[-fade_samples:] *= fade_out
            chunk[:fade_samples] *= fade_in

            # Overlap-add
            overlap = result[-fade_samples:] + chunk[:fade_samples]
            result = np.concatenate([result[:-fade_samples], overlap, chunk[fade_samples:]])

        return result

    def _stitch_with_pause(
        self,
        chunks: List[np.ndarray],
        sample_rate: int,
        pause_sec: float = 0.09,
        fade_sec: float = 0.03,
    ) -> np.ndarray:
        """
        Stitch sentence-like chunks with a short silence bridge instead of
        overlapping voiced regions. This hides chunk resets better for long-form
        speech where the text was already split at sentence boundaries.
        """
        if not chunks:
            return np.array([], dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]

        pause_samples = max(0, int(sample_rate * pause_sec))
        fade_samples = max(1, int(sample_rate * min(fade_sec, 0.04)))
        silence = np.zeros(pause_samples, dtype=np.float32)

        result = chunks[0].astype(np.float32, copy=True)
        for chunk in chunks[1:]:
            next_chunk = chunk.astype(np.float32, copy=True)

            if len(result) >= fade_samples:
                result[-fade_samples:] *= np.linspace(
                    1.0, 0.0, fade_samples, dtype=np.float32
                )
            if len(next_chunk) >= fade_samples:
                next_chunk[:fade_samples] *= np.linspace(
                    0.0, 1.0, fade_samples, dtype=np.float32
                )

            pieces = [result]
            if pause_samples:
                pieces.append(silence)
            pieces.append(next_chunk)
            result = np.concatenate(pieces)

        return result


def main():
    parser = argparse.ArgumentParser(description="Sauti TTS Inference")

    # Input
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--text_file", type=str, help="File with texts (one per line)")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio path")
    parser.add_argument("--ref_text", type=str, default="", help="Reference audio transcription")

    # Output
    parser.add_argument("--output", type=str, default="outputs/generated.wav")
    parser.add_argument("--output_dir", type=str, default="outputs/batch")

    # Model
    parser.add_argument("--checkpoint", type=str, default="ckpts/sauti_tts/model_last.pt")
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="Inference config YAML")

    # Generation parameters
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)

    # Long text
    parser.add_argument("--long_text", action="store_true", help="Enable long text chunking")
    parser.add_argument("--max_chars", type=int, default=180)

    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load inference config if provided
    if args.config:
        with open(args.config) as f:
            inf_config = yaml.safe_load(f)
        inf = inf_config.get("inference", {})
        args.steps = inf.get("nfe_steps", args.steps)
        args.cfg_strength = inf.get("cfg_strength", args.cfg_strength)
        args.speed = inf.get("speed", args.speed)

    # Initialize engine
    engine = SautiInference(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab_path,
    )

    gen_kwargs = dict(
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
        nfe_steps=args.steps,
        cfg_strength=args.cfg_strength,
        speed=args.speed,
        seed=args.seed,
    )

    if args.text_file:
        # Batch mode
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        engine.generate_batch(
            texts=texts,
            output_dir=args.output_dir,
            **gen_kwargs,
        )

    elif args.text:
        # Single text
        if args.long_text:
            engine.generate_long(
                text=args.text,
                output_path=args.output,
                max_chars_per_chunk=args.max_chars,
                **gen_kwargs,
            )
        else:
            engine.generate(
                text=args.text,
                output_path=args.output,
                **gen_kwargs,
            )
    else:
        # Interactive mode
        logger.info("\nSauti TTS — Interactive Mode")
        logger.info("Type Swahili text and press Enter to generate speech.")
        logger.info("Type 'quit' to exit.\n")

        i = 0
        while True:
            try:
                text = input("Swahili > ").strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if not text:
                    continue

                output_path = f"outputs/interactive_{i:04d}.wav"
                engine.generate(
                    text=text,
                    output_path=output_path,
                    **gen_kwargs,
                )
                i += 1

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()

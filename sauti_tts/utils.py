"""
Sauti TTS — Utility functions for audio processing and Swahili text normalization.
"""

import re
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# =============================================================================
# Swahili Text Normalization
# =============================================================================

# Swahili number words
SWAHILI_ONES = {
    0: "sifuri", 1: "moja", 2: "mbili", 3: "tatu", 4: "nne",
    5: "tano", 6: "sita", 7: "saba", 8: "nane", 9: "tisa",
}
SWAHILI_TENS = {
    10: "kumi", 20: "ishirini", 30: "thelathini", 40: "arobaini",
    50: "hamsini", 60: "sitini", 70: "sabini", 80: "themanini", 90: "tisini",
}
SWAHILI_POWERS = {
    100: "mia", 1000: "elfu", 1_000_000: "milioni", 1_000_000_000: "bilioni",
}


def number_to_swahili(n: int) -> str:
    """Convert an integer to Swahili words."""
    if n < 0:
        return "hasi " + number_to_swahili(-n)
    if n in SWAHILI_ONES:
        return SWAHILI_ONES[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return SWAHILI_TENS[tens]
        return f"{SWAHILI_TENS[tens]} na {SWAHILI_ONES[ones]}"
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        prefix = f"mia {SWAHILI_ONES[hundreds]}" if hundreds > 1 else "mia moja"
        if remainder == 0:
            return prefix
        return f"{prefix} na {number_to_swahili(remainder)}"
    if n < 1_000_000:
        thousands = n // 1000
        remainder = n % 1000
        prefix = f"elfu {number_to_swahili(thousands)}"
        if remainder == 0:
            return prefix
        return f"{prefix} na {number_to_swahili(remainder)}"
    if n < 1_000_000_000:
        millions = n // 1_000_000
        remainder = n % 1_000_000
        prefix = f"milioni {number_to_swahili(millions)}"
        if remainder == 0:
            return prefix
        return f"{prefix} na {number_to_swahili(remainder)}"
    # Fallback for very large numbers
    return str(n)


def normalize_swahili_text(text: str) -> str:
    """
    Normalize Swahili text for TTS:
    - Convert numbers to words
    - Expand common abbreviations
    - Clean punctuation
    - Normalize whitespace
    """
    # Expand common Swahili abbreviations
    abbreviations = {
        r"\bDkt\.\b": "Daktari",
        r"\bBw\.\b": "Bwana",
        r"\bBi\.\b": "Bibi",
        r"\bProf\.\b": "Profesa",
        r"\bMh\.\b": "Mheshimiwa",
        r"\bn\.k\.": "na kadhalika",
        r"\bk\.m\.": "kwa mfano",
        r"\bKsh\.?\s?": "shilingi ",
        r"\bTsh\.?\s?": "shilingi ",
        r"\bUSD\s?": "dola za Kimarekani ",
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Convert numbers to Swahili words
    def replace_number(match):
        num_str = match.group(0).replace(",", "")
        try:
            num = int(num_str)
            return number_to_swahili(num)
        except ValueError:
            try:
                # Handle decimals
                parts = num_str.split(".")
                integer_part = number_to_swahili(int(parts[0]))
                decimal_part = " ".join(
                    SWAHILI_ONES.get(int(d), d) for d in parts[1]
                )
                return f"{integer_part} nukta {decimal_part}"
            except (ValueError, IndexError):
                return num_str

    text = re.sub(r"\d[\d,]*\.?\d*", replace_number, text)

    # Normalize punctuation for natural pauses
    text = re.sub(r"[–—]", ", ", text)  # Em/en dashes to comma
    text = re.sub(r"[\"'`]", "", text)  # Remove quotes
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    text = text.strip()

    return text


# =============================================================================
# Audio Processing
# =============================================================================


def load_audio(
    path: str,
    target_sr: int = 24000,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio file, resample, and convert to mono."""
    waveform, sr = torchaudio.load(path)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr


def normalize_audio(
    waveform: torch.Tensor,
    target_lufs: float = -23.0,
) -> torch.Tensor:
    """Loudness-normalize audio to target LUFS."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(24000)
        audio_np = waveform.squeeze().numpy()
        loudness = meter.integrated_loudness(audio_np)

        if np.isinf(loudness):
            return waveform

        gain_db = target_lufs - loudness
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = audio_np * gain_linear

        # Prevent clipping
        peak = np.abs(normalized).max()
        if peak > 0.99:
            normalized = normalized * (0.99 / peak)

        return torch.from_numpy(normalized).unsqueeze(0)
    except ImportError:
        logger.warning("pyloudnorm not installed, using peak normalization")
        peak = waveform.abs().max()
        if peak > 0:
            return waveform / peak * 0.95
        return waveform


def trim_silence(
    waveform: torch.Tensor,
    sr: int = 24000,
    top_db: int = 30,
    min_silence_ms: int = 100,
) -> torch.Tensor:
    """Trim leading/trailing silence from audio."""
    audio_np = waveform.squeeze().numpy()

    # Simple energy-based trimming
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop

    energy = np.array([
        np.sum(audio_np[i:i + frame_length] ** 2)
        for i in range(0, len(audio_np) - frame_length, hop_length)
    ])

    if len(energy) == 0:
        return waveform

    threshold = np.max(energy) * (10 ** (-top_db / 10))
    active_frames = np.where(energy > threshold)[0]

    if len(active_frames) == 0:
        return waveform

    # Add small padding
    pad_frames = max(1, int(min_silence_ms / 10))
    start_frame = max(0, active_frames[0] - pad_frames)
    end_frame = min(len(energy) - 1, active_frames[-1] + pad_frames)

    start_sample = start_frame * hop_length
    end_sample = min(len(audio_np), (end_frame + 1) * hop_length + frame_length)

    trimmed = audio_np[start_sample:end_sample]
    return torch.from_numpy(trimmed).unsqueeze(0)


def get_audio_duration(path: str) -> float:
    """Get audio file duration in seconds."""
    info = sf.info(path)
    return info.frames / info.samplerate


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sample_rate: int = 24000,
) -> None:
    """Save waveform to file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    sf.write(path, waveform.squeeze(0).cpu().numpy(), sample_rate)


# =============================================================================
# Misc Helpers
# =============================================================================


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with rich formatting."""
    try:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except ImportError:
        logging.basicConfig(
            level=getattr(logging, level),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": trainable / total * 100 if total > 0 else 0,
    }


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
Sauti TTS — Evaluation metrics for TTS quality assessment.

Metrics:
1. MOS (Mean Opinion Score) estimation via UTMOS
2. Speaker Similarity (ECAPA-TDNN cosine similarity)
3. Intelligibility (Whisper CER/WER)
4. Signal quality (PESQ, STOI)
5. Duration accuracy
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSEvalResult:
    """Evaluation result for a single sample."""
    sample_id: str
    ref_text: str
    gen_text: str = ""
    mos_score: float = 0.0
    speaker_similarity: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    pesq_score: float = 0.0
    stoi_score: float = 0.0
    duration_ratio: float = 0.0  # gen_duration / ref_duration


@dataclass
class TTSEvalSummary:
    """Aggregated evaluation metrics."""
    num_samples: int = 0
    mean_mos: float = 0.0
    std_mos: float = 0.0
    mean_speaker_sim: float = 0.0
    mean_cer: float = 0.0
    mean_wer: float = 0.0
    mean_pesq: float = 0.0
    mean_stoi: float = 0.0
    mean_duration_ratio: float = 0.0
    results: List[TTSEvalResult] = field(default_factory=list)


class SautiEvaluator:
    """
    Comprehensive TTS evaluation suite for Sauti TTS.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        whisper_model: str = "medium",
        sample_rate: int = 24000,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sample_rate = sample_rate
        self.whisper_model_name = whisper_model

        # Lazy-loaded models
        self._whisper_model = None
        self._speaker_model = None
        self._utmos_model = None

    # -----------------------------------------------------------------
    # MOS Estimation (UTMOS / simple energy-based proxy)
    # -----------------------------------------------------------------

    def estimate_mos(self, audio: np.ndarray, sr: int = 24000) -> float:
        """
        Estimate Mean Opinion Score.
        Uses UTMOS if available, otherwise a proxy based on
        signal-to-noise ratio and spectral quality.
        """
        try:
            return self._utmos_score(audio, sr)
        except Exception:
            return self._proxy_mos(audio, sr)

    def _utmos_score(self, audio: np.ndarray, sr: int) -> float:
        """UTMOS-based MOS prediction."""
        try:
            if self._utmos_model is None:
                import torch

                self._utmos_model = torch.hub.load(
                    "tarepan/SpeechMOS:v1.2.0",
                    "utmos22_strong",
                    trust_repo=True,
                ).to(self.device)
                self._utmos_model.eval()

            wave = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            if sr != 16000:
                import torchaudio

                resampler = torchaudio.transforms.Resample(sr, 16000).to(self.device)
                wave = resampler(wave)

            with torch.no_grad():
                score = self._utmos_model(wave, sr=16000)

            return float(score.item())
        except Exception as e:
            logger.debug(f"UTMOS failed: {e}, using proxy MOS")
            raise

    def _proxy_mos(self, audio: np.ndarray, sr: int) -> float:
        """
        Simple proxy MOS based on audio statistics.
        Not a real MOS predictor, but useful for relative comparisons.
        """
        if len(audio) == 0:
            return 1.0

        # Signal energy
        energy = np.mean(audio ** 2)
        if energy < 1e-10:
            return 1.0

        # Dynamic range
        peak = np.max(np.abs(audio))
        rms = np.sqrt(energy)
        crest_factor = peak / (rms + 1e-10)

        # Spectral flatness (proxy for naturalness)
        import scipy.signal
        _, _, Sxx = scipy.signal.spectrogram(audio, fs=sr, nperseg=512)
        Sxx = np.maximum(Sxx, 1e-10)
        geo_mean = np.exp(np.mean(np.log(Sxx), axis=0))
        arith_mean = np.mean(Sxx, axis=0)
        spectral_flatness = np.mean(geo_mean / (arith_mean + 1e-10))

        # Heuristic MOS (1-5 scale)
        score = 2.5
        if 0.01 < rms < 0.3:
            score += 0.5  # Good volume
        if 3 < crest_factor < 15:
            score += 0.5  # Good dynamics
        if 0.01 < spectral_flatness < 0.5:
            score += 0.5  # Not too flat, not too tonal
        if len(audio) / sr > 0.5:
            score += 0.5  # Reasonable duration

        return min(5.0, max(1.0, score))

    # -----------------------------------------------------------------
    # Speaker Similarity (ECAPA-TDNN)
    # -----------------------------------------------------------------

    def compute_speaker_similarity(
        self,
        ref_audio: np.ndarray,
        gen_audio: np.ndarray,
        sr: int = 24000,
    ) -> float:
        """
        Compute speaker similarity using ECAPA-TDNN embeddings.
        Returns cosine similarity in [0, 1].
        """
        try:
            ref_emb = self._get_speaker_embedding(ref_audio, sr)
            gen_emb = self._get_speaker_embedding(gen_audio, sr)

            # Cosine similarity
            similarity = np.dot(ref_emb, gen_emb) / (
                np.linalg.norm(ref_emb) * np.linalg.norm(gen_emb) + 1e-10
            )
            return float(max(0, similarity))

        except Exception as e:
            logger.warning(f"Speaker similarity failed: {e}")
            return 0.0

    def _get_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract speaker embedding using resemblyzer or speechbrain."""
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav

            if self._speaker_model is None:
                self._speaker_model = VoiceEncoder()

            # Resample to 16kHz for resemblyzer
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            wav = preprocess_wav(audio, source_sr=16000)
            embedding = self._speaker_model.embed_utterance(wav)
            return embedding

        except ImportError:
            # Fallback: simple MFCC-based embedding
            import librosa

            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
            return np.mean(mfcc, axis=1)

    # -----------------------------------------------------------------
    # Intelligibility (Whisper ASR)
    # -----------------------------------------------------------------

    def compute_intelligibility(
        self,
        audio: np.ndarray,
        reference_text: str,
        sr: int = 24000,
        language: str = "sw",
    ) -> Tuple[float, float, str]:
        """
        Compute CER and WER using Whisper ASR.
        Returns (cer, wer, transcription).
        """
        try:
            transcription = self._transcribe(audio, sr, language)

            from jiwer import wer as compute_wer, cer as compute_cer

            ref_clean = reference_text.lower().strip()
            hyp_clean = transcription.lower().strip()

            if not ref_clean:
                return 0.0, 0.0, transcription

            cer_val = compute_cer(ref_clean, hyp_clean)
            wer_val = compute_wer(ref_clean, hyp_clean)

            return float(cer_val), float(wer_val), transcription

        except Exception as e:
            logger.warning(f"Intelligibility computation failed: {e}")
            return 1.0, 1.0, ""

    def _transcribe(
        self, audio: np.ndarray, sr: int, language: str = "sw"
    ) -> str:
        """Transcribe audio using Whisper."""
        try:
            import whisper

            if self._whisper_model is None:
                self._whisper_model = whisper.load_model(
                    self.whisper_model_name, device=str(self.device)
                )

            # Resample to 16kHz
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            audio_tensor = torch.from_numpy(audio).float()
            result = self._whisper_model.transcribe(
                audio_tensor.numpy(),
                language=language,
                task="transcribe",
            )
            return result["text"].strip()

        except ImportError:
            logger.warning("Whisper not installed. Install with: pip install openai-whisper")
            return ""

    # -----------------------------------------------------------------
    # Signal Quality (PESQ, STOI)
    # -----------------------------------------------------------------

    def compute_pesq(
        self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000
    ) -> float:
        """Compute PESQ (Perceptual Evaluation of Speech Quality)."""
        try:
            from pesq import pesq

            # PESQ requires 16kHz or 8kHz
            if sr != 16000:
                import librosa
                ref_audio = librosa.resample(ref_audio, orig_sr=sr, target_sr=16000)
                gen_audio = librosa.resample(gen_audio, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Match lengths
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            score = pesq(sr, ref_audio, gen_audio, "wb")
            return float(score)

        except Exception as e:
            logger.warning(f"PESQ computation failed: {e}")
            return 0.0

    def compute_stoi(
        self, ref_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 24000
    ) -> float:
        """Compute STOI (Short-Time Objective Intelligibility)."""
        try:
            from pystoi import stoi

            # Match lengths
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            score = stoi(ref_audio, gen_audio, sr, extended=True)
            return float(score)

        except Exception as e:
            logger.warning(f"STOI computation failed: {e}")
            return 0.0

    # -----------------------------------------------------------------
    # Full Evaluation Pipeline
    # -----------------------------------------------------------------

    def evaluate_sample(
        self,
        ref_audio: np.ndarray,
        gen_audio: np.ndarray,
        ref_text: str,
        sample_id: str = "",
        sr: int = 24000,
    ) -> TTSEvalResult:
        """Run all evaluations on a single sample pair."""
        result = TTSEvalResult(sample_id=sample_id, ref_text=ref_text)

        # MOS estimation
        result.mos_score = self.estimate_mos(gen_audio, sr)

        # Speaker similarity
        result.speaker_similarity = self.compute_speaker_similarity(
            ref_audio, gen_audio, sr
        )

        # Intelligibility
        cer, wer, gen_text = self.compute_intelligibility(gen_audio, ref_text, sr)
        result.cer = cer
        result.wer = wer
        result.gen_text = gen_text

        # Signal quality (only if we have matching reference)
        if len(ref_audio) > 0 and len(gen_audio) > 0:
            result.pesq_score = self.compute_pesq(ref_audio, gen_audio, sr)
            result.stoi_score = self.compute_stoi(ref_audio, gen_audio, sr)

        # Duration ratio
        ref_dur = len(ref_audio) / sr
        gen_dur = len(gen_audio) / sr
        result.duration_ratio = gen_dur / ref_dur if ref_dur > 0 else 0

        return result

    def evaluate_batch(
        self,
        results: List[TTSEvalResult],
    ) -> TTSEvalSummary:
        """Aggregate evaluation results."""
        summary = TTSEvalSummary(
            num_samples=len(results),
            results=results,
        )

        if not results:
            return summary

        mos_scores = [r.mos_score for r in results]
        summary.mean_mos = float(np.mean(mos_scores))
        summary.std_mos = float(np.std(mos_scores))
        summary.mean_speaker_sim = float(np.mean([r.speaker_similarity for r in results]))
        summary.mean_cer = float(np.mean([r.cer for r in results]))
        summary.mean_wer = float(np.mean([r.wer for r in results]))
        summary.mean_pesq = float(np.mean([r.pesq_score for r in results]))
        summary.mean_stoi = float(np.mean([r.stoi_score for r in results]))
        summary.mean_duration_ratio = float(np.mean([r.duration_ratio for r in results]))

        return summary

    def print_summary(self, summary: TTSEvalSummary) -> str:
        """Format evaluation summary as a table."""
        lines = [
            "",
            "=" * 55,
            "  Sauti TTS — Evaluation Summary",
            "=" * 55,
            f"  Samples evaluated:     {summary.num_samples}",
            f"  MOS (estimated):       {summary.mean_mos:.2f} ± {summary.std_mos:.2f}",
            f"  Speaker Similarity:    {summary.mean_speaker_sim:.3f}",
            f"  CER (Whisper):         {summary.mean_cer:.3f} ({summary.mean_cer*100:.1f}%)",
            f"  WER (Whisper):         {summary.mean_wer:.3f} ({summary.mean_wer*100:.1f}%)",
            f"  PESQ:                  {summary.mean_pesq:.2f} / 4.5",
            f"  STOI:                  {summary.mean_stoi:.3f}",
            f"  Duration Ratio:        {summary.mean_duration_ratio:.2f}x",
            "=" * 55,
        ]
        text = "\n".join(lines)
        logger.info(text)
        return text

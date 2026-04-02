"""
Sauti TTS — Model wrapper for F5-TTS fine-tuning.
Handles model loading, LoRA injection, checkpoint management.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SautiTTSConfig:
    """Configuration for Sauti TTS model."""

    # Model
    model_type: str = "F5TTS_v1_Base"
    pretrained_path: Optional[str] = None  # Path to pretrained F5-TTS checkpoint
    vocab_path: Optional[str] = None

    # LoRA (optional)
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("to_q", "to_k", "to_v", "to_out.0")

    # Audio
    sample_rate: int = 24000
    hop_length: int = 256
    n_mel_channels: int = 100

    # Model architecture (F5-TTS v1 Base defaults)
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    ff_mult: int = 2
    text_dim: int = 512
    conv_layers: int = 4

    @property
    def target_sample_rate(self):
        return self.sample_rate


class SautiTTS:
    """
    Wrapper around F5-TTS for Swahili fine-tuning.
    Handles model creation, LoRA injection, and checkpoint loading.
    """

    def __init__(self, config: SautiTTSConfig):
        self.config = config
        self.model = None
        self.vocoder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self) -> nn.Module:
        """Build the F5-TTS model with optional LoRA."""
        logger.info(f"Building {self.config.model_type} model...")

        try:
            from f5_tts.model import CFM, DiT
            from f5_tts.model.utils import get_tokenizer
        except ImportError:
            raise ImportError(
                "F5-TTS not installed. Run:\n"
                "  git clone https://github.com/SWivid/F5-TTS.git\n"
                "  cd F5-TTS && pip install -e ."
            )

        # Get tokenizer & vocab
        if self.config.vocab_path and os.path.exists(self.config.vocab_path):
            vocab_char_map, vocab_size = get_tokenizer(self.config.vocab_path, "custom")
            logger.info(f"Loaded vocab: {self.config.vocab_path} ({vocab_size} tokens)")
        else:
            # Use default F5-TTS vocab
            vocab_char_map, vocab_size = get_tokenizer(None, "pinyin")
            logger.info(f"Using default Emilia vocab ({vocab_size} tokens)")

        transformer = DiT(
            dim=self.config.dim,
            depth=self.config.depth,
            heads=self.config.heads,
            ff_mult=self.config.ff_mult,
            text_dim=self.config.text_dim,
            conv_layers=self.config.conv_layers,
            text_num_embeds=vocab_size,
            mel_dim=self.config.n_mel_channels,
        )

        mel_spec_kwargs = dict(
            n_fft=self.config.hop_length * 4,
            hop_length=self.config.hop_length,
            win_length=self.config.hop_length * 4,
            n_mel_channels=self.config.n_mel_channels,
            target_sample_rate=self.config.sample_rate,
            mel_spec_type="vocos",
        )

        self.model = CFM(
            transformer=transformer,
            mel_spec_kwargs=mel_spec_kwargs,
            vocab_char_map=vocab_char_map,
        )

        # Load pretrained checkpoint
        if self.config.pretrained_path:
            self._load_pretrained(self.config.pretrained_path)

        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()

        # Move to device
        self.model = self.model.to(self.device)

        # Log parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(
            f"Model parameters: {total_params:,} total, "
            f"{trainable_params:,} trainable "
            f"({trainable_params/total_params*100:.1f}%)"
        )

        return self.model

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained F5-TTS checkpoint."""
        logger.info(f"Loading pretrained: {path}")

        if not os.path.exists(path):
            # Try downloading from HuggingFace
            logger.info("Downloading pretrained F5-TTS v1 Base from HuggingFace...")
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(
                    repo_id="SWivid/F5-TTS",
                    filename="F5TTS_v1_Base/model_1250000.safetensors",
                    local_dir="ckpts",
                )
                logger.info(f"Downloaded to: {path}")
            except Exception as e:
                logger.warning(f"Could not download pretrained model: {e}")
                return

        # Load checkpoint
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(path)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "ema_model_state_dict" in checkpoint:
                state_dict = checkpoint["ema_model_state_dict"]
            else:
                state_dict = checkpoint

        # Filter and load (handle key mismatches gracefully)
        model_dict = self.model.state_dict()
        filtered = {}
        skipped = []
        prefix_map = (
            ("ema_model.online_model.", ""),
            ("online_model.", ""),
            ("ema_model.", ""),
            ("module.", ""),
        )
        bare_transformer_keys = {
            key[len("transformer.") :]
            for key in model_dict
            if key.startswith("transformer.")
        }

        for key, value in state_dict.items():
            normalized_keys = {key}
            for prefix, replacement in prefix_map:
                updated = set()
                for candidate in normalized_keys:
                    if candidate.startswith(prefix):
                        updated.add(replacement + candidate[len(prefix) :])
                normalized_keys.update(updated)

            candidates = []
            for candidate in normalized_keys:
                candidates.append(candidate)
                if candidate.startswith("transformer."):
                    candidates.append(candidate[len("transformer.") :])
                elif candidate in bare_transformer_keys:
                    candidates.append(f"transformer.{candidate}")

            target_key = None
            for candidate in candidates:
                if candidate in model_dict and model_dict[candidate].shape == value.shape:
                    target_key = candidate
                    break

            if target_key is None:
                skipped.append(key)
                continue

            filtered[target_key] = value

        self.model.load_state_dict(filtered, strict=False)
        logger.info(
            f"Loaded {len(filtered)}/{len(state_dict)} pretrained weights"
            f" (skipped {len(skipped)})"
        )
        if not filtered:
            logger.warning(
                "No pretrained weights matched. Sample checkpoint keys: %s | "
                "sample model keys: %s",
                list(state_dict.keys())[:5],
                list(model_dict.keys())[:5],
            )

    def _apply_lora(self) -> None:
        """Inject LoRA adapters into attention layers."""
        logger.info(
            f"Applying LoRA: rank={self.config.lora_rank}, "
            f"alpha={self.config.lora_alpha}"
        )

        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
                bias="none",
                task_type=None,  # Custom model
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        except ImportError:
            logger.warning(
                "PEFT not installed, falling back to manual LoRA injection"
            )
            self._manual_lora()

    def _manual_lora(self) -> None:
        """Manual LoRA injection without PEFT library."""

        class LoRALayer(nn.Module):
            def __init__(self, original: nn.Linear, rank: int, alpha: int):
                super().__init__()
                self.original = original
                self.lora_A = nn.Linear(original.in_features, rank, bias=False)
                self.lora_B = nn.Linear(rank, original.out_features, bias=False)
                self.scaling = alpha / rank
                nn.init.kaiming_uniform_(self.lora_A.weight)
                nn.init.zeros_(self.lora_B.weight)
                original.weight.requires_grad = False
                if original.bias is not None:
                    original.bias.requires_grad = False

            def forward(self, x):
                return self.original(x) + self.lora_B(self.lora_A(x)) * self.scaling

        count = 0
        for name, module in self.model.named_modules():
            for target_name in self.config.lora_target_modules:
                if target_name in name and isinstance(module, nn.Linear):
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(
                        parent,
                        child_name,
                        LoRALayer(
                            module,
                            self.config.lora_rank,
                            self.config.lora_alpha,
                        ),
                    )
                    count += 1
        logger.info(f"Injected LoRA into {count} layers")

    def load_vocoder(self):
        """Load the Vocos vocoder (used by F5-TTS)."""
        try:
            from f5_tts.model.utils import load_vocoder as f5_load_vocoder

            self.vocoder = f5_load_vocoder()
            logger.info("Loaded Vocos vocoder")
        except ImportError:
            logger.warning("Could not load vocoder from F5-TTS")

    def save_checkpoint(
        self,
        path: str,
        epoch: int = 0,
        step: int = 0,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "config": vars(self.config),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }

        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path} (step {step})")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint and return metadata."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        logger.info(
            f"Loaded checkpoint: {path} "
            f"(step {checkpoint.get('step', '?')})"
        )
        return checkpoint

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        ref_audio: torch.Tensor,
        ref_text: str = "",
        steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Synthesize speech from text with voice cloning.

        Args:
            text: Swahili text to synthesize
            ref_audio: Reference audio tensor [1, T] at 24kHz
            ref_text: Transcription of reference audio
            steps: Number of diffusion steps (more = better quality)
            cfg_strength: Classifier-free guidance strength
            sway_sampling_coef: Sway sampling coefficient (-1 for default)
            speed: Speed factor (1.0 = normal)
            seed: Random seed for reproducibility

        Returns:
            Generated audio tensor [1, T] at 24kHz
        """
        from sauti_tts.utils import normalize_swahili_text

        # Normalize text
        text = normalize_swahili_text(text)

        try:
            from f5_tts.infer.utils_infer import (
                infer_process,
                preprocess_ref_audio_text,
            )

            if seed is not None:
                torch.manual_seed(seed)

            # Preprocess reference
            ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
                ref_audio, ref_text
            )

            # Generate
            generated, sr, _ = infer_process(
                ref_audio=ref_audio_processed,
                ref_text=ref_text_processed,
                gen_text=text,
                model_obj=self.model,
                vocoder=self.vocoder,
                nfe_step=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=speed,
            )

            return torch.from_numpy(generated).unsqueeze(0)

        except ImportError:
            raise ImportError(
                "F5-TTS inference utilities not available. "
                "Install F5-TTS from source."
            )

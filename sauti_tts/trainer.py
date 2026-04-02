"""
Sauti TTS — Training loop for F5-TTS fine-tuning.
Implements best practices from F5-TTS community:
- Frame-based batching
- EMA model tracking
- Gradient clipping
- Warmup + cosine decay schedule
- Checkpoint management
- W&B / TensorBoard logging
"""

import os
import json
import time
import inspect
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Experiment
    exp_name: str = "sauti_tts"
    output_dir: str = "ckpts/sauti_tts"
    seed: int = 42

    # Dataset
    dataset_dir: str = "data/waxal_swahili"
    metadata_file: str = "metadata.csv"

    # Training
    epochs: int = 100
    learning_rate: float = 1e-5       # Conservative for fine-tuning
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1

    # Batch size (frame-based)
    batch_size_per_gpu: int = 1600    # Frame count, not sample count
    batch_size_type: str = "frame"     # "frame" or "sample"
    max_samples: int = 64             # Max samples per batch if frame-based

    # Scheduler
    num_warmup_updates: int = 500
    lr_scheduler_type: str = "cosine"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Checkpointing
    save_per_updates: int = 5000
    last_per_steps: int = 10000
    keep_last_n_checkpoints: int = 5

    # Logging
    log_per_updates: int = 50
    logger_type: str = "tensorboard"   # "wandb" or "tensorboard"
    wandb_project: str = "sauti-tts"
    wandb_run_name: Optional[str] = None

    # Resume
    resume_from: Optional[str] = None
    finetune: bool = True             # True = fine-tuning from pretrained

    # Hardware
    mixed_precision: str = "fp16"     # "fp16", "bf16", or "no"
    num_workers: int = 4

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Critical for stable TTS training — EMA weights typically sound better.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply(self, model: nn.Module):
        """Apply EMA weights (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights (after evaluation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return dict(self.shadow)

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class SoundfileCustomDataset(Dataset):
    """
    F5-TTS-compatible dataset that avoids torchaudio.load/torchcodec.
    """

    def __init__(
        self,
        hf_dataset,
        durations=None,
        target_sample_rate: int = 24_000,
        hop_length: int = 256,
        n_mel_channels: int = 100,
        n_fft: int = 1024,
        win_length: int = 1024,
        mel_spec_type: str = "vocos",
        preprocessed_mel: bool = False,
        mel_spec_module: nn.Module | None = None,
    ):
        from f5_tts.model.modules import MelSpec

        self.data = hf_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = mel_spec_module or MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            )

    def get_frame_len(self, index):
        if self.durations is not None:
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            duration = row["duration"]
            if 0.3 <= duration <= 30:
                break
            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio_np, source_sample_rate = sf.read(
                row["audio_path"], dtype="float32"
            )
            if audio_np.ndim == 1:
                audio_np = audio_np[:, None]
            audio = torch.from_numpy(audio_np.T)

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if source_sample_rate != self.target_sample_rate:
                audio = torchaudio.functional.resample(
                    audio, source_sample_rate, self.target_sample_rate
                )

            mel_spec = self.mel_spectrogram(audio).squeeze(0)

        return {
            "mel_spec": mel_spec,
            "text": row["text"],
        }


class SautiTrainer:
    """
    Fine-tuning trainer for Sauti TTS.
    Wraps around F5-TTS training utilities with Swahili-specific optimizations.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
    ):
        self.config = config
        self.model = model
        self.device = config.device

        # Create output directory
        self.ckpt_dir = Path(config.output_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.ckpt_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(config), f, indent=2, default=str)

        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # Setup components
        self._setup_optimizer()
        self._setup_ema()
        self._setup_logger()

    def _setup_optimizer(self):
        """Setup optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "ln" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate schedulers
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.num_warmup_updates,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=100000,  # Will be updated based on actual steps
            eta_min=self.config.learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.num_warmup_updates],
        )

    def _setup_ema(self):
        """Setup EMA tracking."""
        if self.config.use_ema:
            self.ema = EMAModel(self.model, decay=self.config.ema_decay)
            logger.info(f"EMA enabled with decay={self.config.ema_decay}")
        else:
            self.ema = None
            logger.info("EMA disabled (for short fine-tuning, this may be better)")

    def _setup_logger(self):
        """Setup training logger (W&B or TensorBoard)."""
        self.writer = None

        if self.config.logger_type == "wandb":
            try:
                import wandb

                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name or self.config.exp_name,
                    config=vars(self.config),
                )
                self.writer = wandb
                logger.info("W&B logging enabled")
            except (ImportError, Exception) as e:
                logger.warning(f"W&B not available: {e}. Falling back to TensorBoard.")
                self.config.logger_type = "tensorboard"

        if self.config.logger_type == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter

                log_dir = self.ckpt_dir / "tensorboard"
                self.writer = SummaryWriter(str(log_dir))
                logger.info(f"TensorBoard logging: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available. Logging to console only.")

    def train_with_f5tts(self):
        """
        Launch training using F5-TTS's built-in trainer.
        This is the recommended approach — uses F5-TTS's optimized
        data pipeline, loss computation, and training loop.
        """
        logger.info("=" * 60)
        logger.info("Sauti TTS — Training with F5-TTS Engine")
        logger.info("=" * 60)

        try:
            from datasets import Dataset as HFDataset
            from datasets import load_from_disk
            from f5_tts.model.trainer import Trainer as F5Trainer

            trainer_kwargs = {
                "model": self.model,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "num_warmup_updates": self.config.num_warmup_updates,
                "save_per_updates": self.config.save_per_updates,
                "keep_last_n_checkpoints": self.config.keep_last_n_checkpoints,
                "last_per_updates": self.config.last_per_steps,
                "checkpoint_path": str(self.ckpt_dir),
                "batch_size_per_gpu": self.config.batch_size_per_gpu,
                "batch_size_type": self.config.batch_size_type,
                "max_samples": self.config.max_samples,
                "grad_accumulation_steps": self.config.grad_accumulation_steps,
                "max_grad_norm": self.config.max_grad_norm,
                "logger": self.config.logger_type,
                "wandb_project": self.config.wandb_project,
                "wandb_run_name": self.config.wandb_run_name or self.config.exp_name,
                "noise_scheduler": "logit_normal",
            }

            trainer_signature = inspect.signature(F5Trainer.__init__)
            trainer_kwargs = {
                key: value
                for key, value in trainer_kwargs.items()
                if key in trainer_signature.parameters
            }
            trainer = F5Trainer(**trainer_kwargs)

            raw_dataset_path = os.path.join(self.config.dataset_dir, "raw")
            raw_arrow_path = os.path.join(self.config.dataset_dir, "raw.arrow")
            duration_path = os.path.join(self.config.dataset_dir, "duration.json")

            try:
                hf_dataset = load_from_disk(raw_dataset_path)
            except Exception:
                hf_dataset = HFDataset.from_file(raw_arrow_path)

            with open(duration_path, "r", encoding="utf-8") as f:
                durations = json.load(f)["duration"]

            train_dataset = SoundfileCustomDataset(
                hf_dataset,
                durations=durations,
                preprocessed_mel=False,
            )

            train_kwargs = {
                "num_workers": self.config.num_workers,
                "resumable_with_seed": self.config.seed,
            }
            train_signature = inspect.signature(trainer.train)
            train_kwargs = {
                key: value
                for key, value in train_kwargs.items()
                if key in train_signature.parameters
            }
            trainer.train(
                train_dataset,
                **train_kwargs,
            )

        except ImportError as e:
            logger.warning(f"F5-TTS trainer not available ({e}), using custom loop")
            self.train_custom()

    def train_custom(self):
        """
        Custom training loop as fallback.
        Implements the core F5-TTS flow-matching training objective.
        """
        logger.info("=" * 60)
        logger.info("Sauti TTS — Custom Training Loop")
        logger.info("=" * 60)

        from sauti_tts.utils import seed_everything

        seed_everything(self.config.seed)

        # Resume if specified
        if self.config.resume_from:
            self._resume_checkpoint(self.config.resume_from)

        self.model.train()
        self.model = self.model.to(self.device)

        # Training metrics
        running_loss = 0.0
        step_times = []
        start_time = time.time()

        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Batch size: {self.config.batch_size_per_gpu} ({self.config.batch_size_type})")
        logger.info(f"Grad accumulation: {self.config.grad_accumulation_steps}")
        logger.info(f"Mixed precision: {self.config.mixed_precision}")

        # AMP scaler
        use_amp = self.config.mixed_precision in ("fp16", "bf16")
        amp_dtype = torch.float16 if self.config.mixed_precision == "fp16" else torch.bfloat16
        scaler = torch.amp.GradScaler("cuda", enabled=(self.config.mixed_precision == "fp16"))

        logger.info("\nTraining started. Use Ctrl+C to stop gracefully.\n")

        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch

                # NOTE: In practice, you'd iterate over the F5-TTS dataset here.
                # This skeleton demonstrates the training step logic.
                # The actual data loading uses F5-TTS's frame-based batching.

                logger.info(f"\n--- Epoch {epoch + 1}/{self.config.epochs} ---")

                # Placeholder: Replace with actual data iteration
                # for batch in dataloader:
                #     loss = self._training_step(batch, scaler, amp_dtype)
                #     ... (see _training_step below)

                logger.info(
                    "Custom training loop requires F5-TTS dataset loader. "
                    "Use train_with_f5tts() for the full pipeline, or "
                    "use the accelerate CLI:\n\n"
                    "  accelerate launch scripts/train.py "
                    f"--config configs/finetune_single_gpu.yaml"
                )
                break

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted. Saving checkpoint...")
            self._save_checkpoint(tag="interrupted")

        total_time = time.time() - start_time
        logger.info(f"\nTraining complete. Total time: {total_time/3600:.2f}h")

    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
        scaler: torch.amp.GradScaler,
        amp_dtype: torch.dtype,
    ) -> float:
        """
        Single training step implementing flow-matching loss.

        The F5-TTS flow-matching objective:
        1. Sample random timestep t ~ U(0, 1)
        2. Create noisy audio: x_t = (1-t) * x_0 + t * noise
        3. Predict velocity: v_pred = model(x_t, t, text)
        4. Loss = MSE(v_pred, noise - x_0)
        """
        step_start = time.time()

        mel = batch["mel"].to(self.device)         # [B, T, mel_dim]
        text_ids = batch["text"].to(self.device)    # [B, L]
        mel_lens = batch["mel_lens"].to(self.device)
        text_lens = batch["text_lens"].to(self.device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
            # Sample timestep
            t = torch.rand(mel.shape[0], device=self.device)

            # Sample noise
            noise = torch.randn_like(mel)

            # Create noisy mel
            t_expanded = t[:, None, None]
            x_t = (1 - t_expanded) * mel + t_expanded * noise

            # Predict velocity
            v_pred = self.model(
                x_t,
                cond=text_ids,
                time=t,
                lens=mel_lens,
            )

            # Flow-matching loss
            target = noise - mel
            loss = F.mse_loss(v_pred, target, reduction="none")

            # Mask padded regions
            mask = torch.arange(mel.shape[1], device=self.device)[None, :] < mel_lens[:, None]
            mask = mask.unsqueeze(-1)
            loss = (loss * mask).sum() / mask.sum()

        # Gradient accumulation
        loss = loss / self.config.grad_accumulation_steps
        scaler.scale(loss).backward()

        if (self.global_step + 1) % self.config.grad_accumulation_steps == 0:
            scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

        self.global_step += 1
        loss_val = loss.item() * self.config.grad_accumulation_steps

        # Logging
        if self.global_step % self.config.log_per_updates == 0:
            step_time = time.time() - step_start
            lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Step {self.global_step} | "
                f"Loss: {loss_val:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {step_time:.2f}s"
            )

            if self.writer is not None:
                if self.config.logger_type == "wandb":
                    self.writer.log({
                        "train/loss": loss_val,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/step_time": step_time,
                    }, step=self.global_step)
                else:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        # Checkpointing
        if self.global_step % self.config.save_per_updates == 0:
            self._save_checkpoint()

        return loss_val

    def _save_checkpoint(self, tag: Optional[str] = None):
        """Save model checkpoint."""
        if tag:
            filename = f"model_{tag}.pt"
        else:
            filename = f"model_{self.global_step}.pt"

        path = self.ckpt_dir / filename

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": vars(self.config),
        }

        if self.ema is not None:
            save_dict["ema_state_dict"] = self.ema.state_dict()

        torch.save(save_dict, path)
        logger.info(f"Saved checkpoint: {path}")

        # Also save as model_last.pt
        last_path = self.ckpt_dir / "model_last.pt"
        torch.save(save_dict, last_path)

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Keep only the last N checkpoints."""
        ckpts = sorted(
            self.ckpt_dir.glob("model_[0-9]*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        while len(ckpts) > self.config.keep_last_n_checkpoints:
            oldest = ckpts.pop(0)
            oldest.unlink()
            logger.info(f"Removed old checkpoint: {oldest}")

    def _resume_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        logger.info(
            f"Resumed: step={self.global_step}, epoch={self.epoch}, "
            f"best_loss={self.best_loss:.4f}"
        )

#!/usr/bin/env python3
"""
Sauti TTS — Training / Fine-tuning Script

Fine-tunes F5-TTS v1 Base on the WaxalNLP Swahili TTS dataset.
Supports:
- Full fine-tuning (best quality)
- LoRA fine-tuning (lower VRAM)
- Single GPU and multi-GPU (via accelerate)

Usage:
    # Single GPU
    python scripts/train.py --config configs/finetune_single_gpu.yaml

    # Multi-GPU (setup accelerate first: accelerate config)
    accelerate launch scripts/train.py --config configs/finetune_multi_gpu.yaml

    # LoRA (12GB VRAM)
    python scripts/train.py --config configs/finetune_lora.yaml

    # F5-TTS CLI (alternative - uses built-in F5-TTS trainer)
    python scripts/train.py --use_f5tts_cli \
        --dataset_dir data/waxal_swahili \
        --exp_name sauti_tts
"""

import os
import sys
import argparse
import logging
import subprocess
import shutil
from pathlib import Path

import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sauti_tts.utils import setup_logging, seed_everything, count_parameters
from sauti_tts.model import SautiTTS, SautiTTSConfig
from sauti_tts.trainer import SautiTrainer, TrainingConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_pretrained_checkpoint(config: dict) -> str:
    """
    Download and setup pretrained F5-TTS checkpoint.
    Returns path to checkpoint file.
    """
    model_cfg = config.get("model", {})
    repo_id = model_cfg.get("pretrained", "SWivid/F5-TTS")
    ckpt_name = model_cfg.get(
        "pretrained_ckpt", "F5TTS_v1_Base/model_1250000.safetensors"
    )

    # Check if already downloaded
    local_path = os.path.join("ckpts", ckpt_name)
    if os.path.exists(local_path):
        logger.info(f"Pretrained checkpoint found: {local_path}")
        return local_path

    # Download from HuggingFace
    logger.info(f"Downloading pretrained model from {repo_id}...")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename=ckpt_name,
            local_dir="ckpts",
        )
        logger.info(f"Downloaded: {path}")
        return path
    except Exception as e:
        logger.error(
            f"Failed to download pretrained model: {e}\n"
            f"Please manually download from https://huggingface.co/{repo_id}\n"
            f"and place at: {local_path}"
        )
        sys.exit(1)


def train_with_config(config_path: str):
    """Main training entry point using our config."""
    config = load_config(config_path)

    # Setup
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    ckpt_cfg = config.get("checkpoint", {})
    log_cfg = config.get("logging", {})

    seed = config.get("seed", 42)
    seed_everything(seed)

    logger.info("=" * 60)
    logger.info("  Sauti TTS — Fine-tuning")
    logger.info("=" * 60)
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Model: {model_cfg.get('type', 'F5TTS_v1_Base')}")
    logger.info(f"  LoRA: {model_cfg.get('use_lora', False)}")
    logger.info(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 60)

    # Step 1: Setup pretrained checkpoint
    pretrained_path = setup_pretrained_checkpoint(config)

    # Step 2: Setup vocab
    vocab_path = model_cfg.get("vocab_path")
    if vocab_path is None:
        # Try dataset directory
        dataset_vocab = os.path.join(dataset_cfg.get("dir", ""), "vocab.txt")
        if os.path.exists(dataset_vocab):
            vocab_path = dataset_vocab
            logger.info(f"Using vocab from dataset: {vocab_path}")

    # Step 3: Build model
    model_config = SautiTTSConfig(
        model_type=model_cfg.get("type", "F5TTS_v1_Base"),
        pretrained_path=pretrained_path,
        vocab_path=vocab_path,
        use_lora=model_cfg.get("use_lora", False),
        lora_rank=model_cfg.get("lora_rank", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        lora_target_modules=tuple(
            model_cfg.get("lora_target_modules", ("to_q", "to_k", "to_v", "to_out.0"))
        ),
        dim=model_cfg.get("dim", 1024),
        depth=model_cfg.get("depth", 22),
        heads=model_cfg.get("heads", 16),
        ff_mult=model_cfg.get("ff_mult", 2),
        text_dim=model_cfg.get("text_dim", 512),
        conv_layers=model_cfg.get("conv_layers", 4),
    )

    sauti = SautiTTS(model_config)
    model = sauti.build_model()

    params = count_parameters(model)
    logger.info(f"\nParameters: {params['total']:,} total, {params['trainable']:,} trainable ({params['trainable_pct']:.1f}%)")

    # Step 4: Setup trainer
    training_config = TrainingConfig(
        exp_name=config.get("exp_name", "sauti_tts"),
        output_dir=config.get("output_dir", "ckpts/sauti_tts"),
        seed=seed,
        dataset_dir=dataset_cfg.get("dir", "data/waxal_swahili"),
        metadata_file=dataset_cfg.get("metadata", "metadata.csv"),
        epochs=train_cfg.get("epochs", 100),
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        grad_accumulation_steps=train_cfg.get("grad_accumulation_steps", 1),
        batch_size_per_gpu=train_cfg.get("batch_size_per_gpu", 1600),
        batch_size_type=train_cfg.get("batch_size_type", "frame"),
        max_samples=train_cfg.get("max_samples", 64),
        num_warmup_updates=train_cfg.get("num_warmup_updates", 500),
        use_ema=train_cfg.get("use_ema", True),
        ema_decay=train_cfg.get("ema_decay", 0.9999),
        mixed_precision=train_cfg.get("mixed_precision", "fp16"),
        num_workers=train_cfg.get("num_workers", 4),
        save_per_updates=ckpt_cfg.get("save_per_updates", 5000),
        last_per_steps=ckpt_cfg.get("last_per_steps", 10000),
        keep_last_n_checkpoints=ckpt_cfg.get("keep_last_n", 5),
        log_per_updates=log_cfg.get("log_per_updates", 50),
        logger_type=log_cfg.get("type", "tensorboard"),
        wandb_project=log_cfg.get("wandb_project", "sauti-tts"),
        wandb_run_name=log_cfg.get("wandb_run_name"),
        finetune=True,
    )

    trainer = SautiTrainer(training_config, model)

    # Step 5: Train
    logger.info("\nStarting fine-tuning...")
    trainer.train_with_f5tts()


def train_with_f5tts_cli(
    dataset_dir: str,
    exp_name: str = "sauti_tts",
    epochs: int = 100,
    lr: float = 1e-5,
    batch_size: int = 1600,
    use_lora: bool = False,
):
    """
    Alternative: Use F5-TTS's built-in CLI for training.
    This is the simplest approach if F5-TTS is installed.
    """
    logger.info("=" * 60)
    logger.info("  Sauti TTS — Training via F5-TTS CLI")
    logger.info("=" * 60)

    # Check if F5-TTS CLI is available
    try:
        result = subprocess.run(
            ["f5-tts_finetune-gradio", "--help"],
            capture_output=True,
            text=True,
        )
        has_gradio = result.returncode == 0
    except FileNotFoundError:
        has_gradio = False

    if has_gradio:
        logger.info("F5-TTS Gradio fine-tuning UI available!")
        logger.info("Launching Gradio interface...")
        subprocess.run(["f5-tts_finetune-gradio"])
    else:
        # Use accelerate CLI
        cmd = [
            "accelerate", "launch",
            "src/f5_tts/train/train.py" if os.path.exists("src/f5_tts") else "F5-TTS/src/f5_tts/train/train.py",
            "--config-name", "F5TTS_v1_Base.yaml",
            f"++datasets.name={exp_name}",
            f"++datasets.batch_size_per_gpu={batch_size}",
            f"++optim.learning_rate={lr}",
            f"++trainer.epochs={epochs}",
        ]

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("\nAlternatively, prepare data with F5-TTS scripts:")
        logger.info(f"  python F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py \\")
        logger.info(f"    {dataset_dir}/metadata.csv \\")
        logger.info(f"    F5-TTS/data/{exp_name}")
        logger.info(f"\nThen copy pretrained checkpoint:")
        logger.info(f"  mkdir -p ckpts/{exp_name}")
        logger.info(f"  cp ckpts/F5TTS_v1_Base/model_1250000.safetensors ckpts/{exp_name}/model_last.pt")
        logger.info(f"\nThen train:")
        logger.info(f"  {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"F5-TTS CLI training failed: {e}")
            logger.info(
                "\nMake sure F5-TTS is installed from source:\n"
                "  git clone https://github.com/SWivid/F5-TTS.git\n"
                "  cd F5-TTS && pip install -e ."
            )


def main():
    parser = argparse.ArgumentParser(description="Sauti TTS Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--use_f5tts_cli",
        action="store_true",
        help="Use F5-TTS built-in CLI instead of custom trainer",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/waxal_swahili",
        help="Dataset directory (for F5-TTS CLI mode)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="sauti_tts",
        help="Experiment name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1600,
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA fine-tuning",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.use_f5tts_cli:
        train_with_f5tts_cli(
            dataset_dir=args.dataset_dir,
            exp_name=args.exp_name,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            use_lora=args.lora,
        )
    elif args.config:
        train_with_config(args.config)
    else:
        # Default config
        default_config = "configs/finetune_single_gpu.yaml"
        if os.path.exists(default_config):
            train_with_config(default_config)
        else:
            logger.error(
                "No config provided. Usage:\n"
                "  python scripts/train.py --config configs/finetune_single_gpu.yaml\n"
                "  python scripts/train.py --use_f5tts_cli --dataset_dir data/waxal_swahili"
            )


if __name__ == "__main__":
    main()

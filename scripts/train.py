#!/usr/bin/env python3
"""Train Task Compressor (single-stage QA).

Usage:
  # Single GPU
  python scripts/train.py --config configs/default.yaml

  # Multi-GPU (DDP via torchrun)
  torchrun --nproc_per_node=8 scripts/train.py --config configs/default.yaml

  # Override config values
  python scripts/train.py --config configs/default.yaml \
      model.lora_rank=64 training.total_steps=20000
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_compressor.config import Config
from task_compressor.data import QACollator, QADataset
from task_compressor.models.task_compressor_model import TaskCompressorModel
from task_compressor.trainer import Trainer

from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Task Compressor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g. model.lora_rank=64)",
    )
    return parser.parse_args()


def load_checkpoint(model: TaskCompressorModel, ckpt_dir: str) -> None:
    """Load checkpoint into model (LoRA adapter + task compressor modules)."""
    ckpt_path = Path(ckpt_dir)
    logger.info(f"Loading checkpoint from {ckpt_path}")

    # Load task compressor modules (perceiver, prompt_encoder, context_tokens, separator)
    modules_path = ckpt_path / "task_compressor_modules.pt"
    if modules_path.exists():
        state = torch.load(modules_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded task_compressor_modules.pt "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    else:
        logger.warning(f"No task_compressor_modules.pt found in {ckpt_path}")

    # Load LoRA adapter weights
    lora_path = ckpt_path / "lora_adapter"
    if lora_path.exists():
        from peft import set_peft_model_state_dict
        lora_state = torch.load(
            lora_path / "adapter_model.safetensors"
            if (lora_path / "adapter_model.safetensors").exists()
            else lora_path / "adapter_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        set_peft_model_state_dict(model.base_model, lora_state)
        logger.info(f"Loaded LoRA adapter from {lora_path}")
    else:
        logger.warning(f"No lora_adapter directory found in {ckpt_path}")


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────
    config = Config.from_yaml(args.config)
    if args.overrides:
        overrides = {}
        for item in args.overrides:
            k, v = item.split("=", 1)
            overrides[k] = v
        config.merge_overrides(overrides)

    # ── Logging ──────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    log_level = logging.INFO if local_rank == 0 else logging.WARNING
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=log_level, format=log_format)

    # Also log to file (only on main process)
    if local_rank == 0:
        log_dir = Path(config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "train.log", mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

    if local_rank == 0:
        logger.info(f"Config: {config.to_dict()}")

    # ── Distributed init ─────────────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    set_seed(config.seed)

    # ── Tokenizer ────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer from {config.model.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Datasets ─────────────────────────────────────────────────────────
    logger.info("Loading datasets ...")

    train_dataset = QADataset(
        data_path=config.data.train_file,
        tokenizer=tokenizer,
        max_context_length=config.data.max_context_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
    )
    dev_dataset = QADataset(
        data_path=config.data.dev_file,
        tokenizer=tokenizer,
        max_context_length=config.data.max_context_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
    )
    collator = QACollator(
        pad_token_id=tokenizer.pad_token_id,
        max_context_length=config.data.max_context_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
    )

    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Dev:   {len(dev_dataset)} samples")

    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_gpu_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.training.per_gpu_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {config.model.base_model}")
    torch_dtype = torch.bfloat16 if config.training.bf16 else torch.float32
    model = TaskCompressorModel(config.model, torch_dtype=torch_dtype)

    # Resume from checkpoint if specified
    if config.training.resume_from:
        load_checkpoint(model, config.training.resume_from)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Parameters: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(model, config, train_loader, dev_loader, tokenizer=tokenizer)
    logger.info("Starting training ...")
    trainer.train()
    logger.info("Training complete!")

    # ── Cleanup ──────────────────────────────────────────────────────────
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

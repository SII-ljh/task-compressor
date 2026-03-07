#!/usr/bin/env python3
"""Train Task Compressor.

Usage:
  # Single GPU
  python scripts/train.py --config configs/default.yaml

  # Multi-GPU (DDP via torchrun)
  torchrun --nproc_per_node=8 scripts/train.py --config configs/default.yaml

  # Override config values
  python scripts/train.py --config configs/default.yaml \\
      model.lora_rank=64 training.total_steps=20000

  # Ablation: different k values
  python scripts/train.py --config configs/default.yaml \\
      model.n_prompt_tokens=0 model.n_context_tokens=16
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
    logging.basicConfig(
        level=logging.INFO if local_rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

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
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model, trust_remote_code=True
    )
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
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Dev:   {len(dev_dataset)} samples")

    collator = QACollator(pad_token_id=tokenizer.pad_token_id)

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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Parameters: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )

    # ── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(model, config, train_loader, dev_loader)
    logger.info("Starting training ...")
    trainer.train()
    logger.info("Training complete!")

    # ── Cleanup ──────────────────────────────────────────────────────────
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

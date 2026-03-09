"""Shared utilities for diagnostic experiments.

Provides device detection, tensor statistics, model building, data preparation,
wandb helpers, and CLI argument parsing shared by pre/mid/post training scripts.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from task_compressor.config import Config, ModelConfig
from task_compressor.data import QACollator, QADataset
from task_compressor.models import TaskCompressorModel


# ─── device / tensor helpers ─────────────────────────────────────────


def detect_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_dtype(device: torch.device, no_bf16: bool = False) -> torch.dtype:
    """Auto-detect best dtype for the device."""
    if device.type == "cuda" and not no_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def grad_norm(module: nn.Module) -> float:
    """Total L2 gradient norm across all parameters of *module*."""
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.data.float().norm(2).item() ** 2
    return math.sqrt(sq)


def stats_str(t: torch.Tensor) -> str:
    """One-line tensor statistics string."""
    with torch.no_grad():
        t_f = t.float()
        return (
            f"mean={t_f.mean().item():+.4f}  std={t_f.std().item():.4f}  "
            f"min={t_f.min().item():+.4f}  max={t_f.max().item():+.4f}"
        )


def to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch dict to *device*."""
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


# ─── statistical tools ───────────────────────────────────────────────


def pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        x: (N, D) tensor
    Returns:
        (N, N) cosine similarity matrix
    """
    x_norm = torch.nn.functional.normalize(x.float(), dim=-1)
    return x_norm @ x_norm.T


def effective_rank(x: torch.Tensor) -> float:
    """Compute effective rank via SVD entropy method.

    Args:
        x: (N, D) or (B, N, D) tensor
    Returns:
        effective rank (float)
    """
    if x.dim() == 3:
        x = x[0]
    s = torch.linalg.svdvals(x.float().cpu())
    s = s / s.sum().clamp(min=1e-10)
    entropy = -(s * (s + 1e-10).log()).sum()
    return entropy.exp().item()


# ─── model loading ───────────────────────────────────────────────────


def load_model(
    config: Config,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> TaskCompressorModel:
    """Build a TaskCompressorModel, optionally loading checkpoint weights.

    Args:
        config: full Config object
        checkpoint_path: path to checkpoint .pt file
        device: target device (auto-detected if None)
        torch_dtype: model dtype (auto-detected if None)
    Returns:
        TaskCompressorModel on *device*
    """
    if device is None:
        device = detect_device()
    if torch_dtype is None:
        torch_dtype = detect_dtype(device)

    model = TaskCompressorModel(config.model, torch_dtype=torch_dtype)

    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        print(f"  Loaded checkpoint from {checkpoint_path}")

    model.to(device)
    return model


# ─── data preparation ────────────────────────────────────────────────


def prepare_qa_batch(
    config: Config,
    data_path: str,
    batch_size: int,
    device: torch.device,
    tokenizer_name: Optional[str] = None,
) -> Tuple[dict, AutoTokenizer]:
    """Load a single QA batch and tokenizer for diagnostic use.

    Returns:
        (batch_on_device, tokenizer)
    """
    model_name = tokenizer_name or config.model.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = QADataset(
        data_path,
        tokenizer,
        max_context_length=config.data.max_context_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
    )
    collator = QACollator(pad_token_id=tokenizer.pad_token_id)

    n = min(batch_size, len(ds))
    subset = Subset(ds, list(range(n)))

    batch = next(
        iter(DataLoader(subset, batch_size=n, shuffle=False, collate_fn=collator))
    )
    batch = to_device(batch, device)
    return batch, tokenizer


def prepare_qa_loader(
    config: Config,
    data_path: str,
    batch_size: int,
    max_samples: int = 0,
    tokenizer_name: Optional[str] = None,
) -> Tuple[DataLoader, AutoTokenizer]:
    """Build a QA DataLoader for evaluation diagnostics.

    Returns:
        (data_loader, tokenizer)
    """
    model_name = tokenizer_name or config.model.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = QADataset(
        data_path,
        tokenizer,
        max_context_length=config.data.max_context_length,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
    )
    if max_samples > 0 and max_samples < len(ds):
        ds = Subset(ds, list(range(max_samples)))

    collator = QACollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return loader, tokenizer


def auto_detect_data_path() -> Optional[str]:
    """Auto-detect QA data path."""
    candidates = [
        "data/qa_tiny_train.json",
        "data/qa_dev.json",
        "data/qa_train.json",
        "../deep_compressor/data/qa_tiny_train.json",
        "../deep_compressor/data/qa_dev.json",
        "../deep_compressor/data/qa_train.json",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ─── wandb helpers ────────────────────────────────────────────────────


def init_wandb(
    enabled: bool,
    project: str,
    run_name: str,
    config: Optional[dict] = None,
    entity: Optional[str] = None,
):
    """Initialize a wandb run if enabled."""
    if not enabled:
        return None
    try:
        import os
        import wandb

        os.environ["WANDB_MODE"] = "offline"
        kwargs = dict(project=project, name=run_name, mode="offline")
        if config is not None:
            kwargs["config"] = config
        if entity is not None:
            kwargs["entity"] = entity
        return wandb.init(**kwargs)
    except ImportError:
        print("WARNING: wandb not installed, skipping wandb logging")
        return None


def log_wandb(run, metrics: dict, step: Optional[int] = None):
    """Log metrics to wandb if run is active."""
    if run is None:
        return
    if step is not None:
        run.log(metrics, step=step)
    else:
        run.log(metrics)


def finish_wandb(run):
    """Finish wandb run if active."""
    if run is not None:
        run.finish()


# ─── CLI factory ──────────────────────────────────────────────────────


def base_parser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with common diagnostic arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_path", default=None,
                        help="Path to data file (auto-detected if not set)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 mixed precision")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="tc-diagnostic",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (team/user)")
    return parser

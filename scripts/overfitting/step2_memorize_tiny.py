#!/usr/bin/env python3
"""Step 2: Memorize a small batch — GPU overfit test (~10 min).

Train Task Compressor on N samples (default: 32) as one batch.
No val split — pure memorization. Goal: loss drops hard, PPL → 1.
Supports both Stage 1 (NTP) and Stage 2 (QA) modes.

Usage:
  # Stage 1 (NTP) — default
  python scripts/overfitting/step2_memorize_tiny.py --stage 1

  # Stage 2 (QA)
  python scripts/overfitting/step2_memorize_tiny.py --stage 2

  # Custom settings
  python scripts/overfitting/step2_memorize_tiny.py --stage 1 --steps 3000 --lr 5e-4
  python scripts/overfitting/step2_memorize_tiny.py --stage 2 --num_samples 16 --batch_size 16
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_compressor.config import Config
from task_compressor.data import NTPCollator, NTPDataset, QACollator, QADataset
from task_compressor.models import TaskCompressorModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step2_memorize_tiny")


# ── Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, stage, use_bf16=False):
    """Compute average loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            if stage == 1:
                outputs = model(
                    mode="ntp",
                    doc_ids=batch_dev["doc_ids"],
                    doc_mask=batch_dev["doc_mask"],
                    segment_ids=batch_dev["segment_ids"],
                    segment_mask=batch_dev["segment_mask"],
                    segment_labels=batch_dev["segment_labels"],
                )
                loss_key = "ntp_loss"
            else:
                outputs = model(
                    mode="qa",
                    context_ids=batch_dev["context_ids"],
                    context_mask=batch_dev["context_mask"],
                    prompt_ids=batch_dev["prompt_ids"],
                    prompt_mask=batch_dev["prompt_mask"],
                    response_ids=batch_dev["response_ids"],
                    response_mask=batch_dev["response_mask"],
                    distill_alpha=0.0,
                )
                loss_key = "qa_loss"
        bs = next(v for v in batch_dev.values() if isinstance(v, torch.Tensor)).shape[0]
        total_loss += outputs[loss_key].float().item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return {"loss": avg_loss, "perplexity": ppl}


# ── Training loop ─────────────────────────────────────────────────────

def _build_optimizer(model, lr, lora_lr):
    """Build optimizer with differential learning rates (perceiver vs LoRA)."""
    lora_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora" in name.lower():
            lora_params.append(param)
        else:
            new_params.append(param)
    groups = [
        {"params": new_params, "lr": lr},
        {"params": lora_params, "lr": lora_lr},
    ]
    logger.info(f"Optimizer: {len(new_params)} new params (lr={lr:.2e}), "
                f"{len(lora_params)} LoRA params (lr={lora_lr:.2e})")
    return torch.optim.AdamW(groups, weight_decay=0.0)


def train(model, train_loader, device, args, use_bf16=False):
    """Batch memorization loop — no val split."""

    optimizer = _build_optimizer(model, args.lr, args.lora_lr)

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        if args.scheduler == "cosine":
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    csv_path = os.path.join(args.output_dir, "learning_curve.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "train_ppl", "lr", "elapsed_s"])

    step = 0
    epoch = 0
    running_loss = 0.0
    micro_steps = 0
    best_loss = float("inf")
    t0 = time.time()

    if use_bf16:
        logger.info("Using bf16 mixed precision (autocast)")

    # Initial eval
    metrics = evaluate(model, train_loader, device, args.stage, use_bf16=use_bf16)
    logger.info(f"[init] train_loss={metrics['loss']:.4f}  train_ppl={metrics['perplexity']:.1f}")
    csv_writer.writerow([0, f"{metrics['loss']:.4f}", f"{metrics['perplexity']:.1f}",
                         f"{args.lr:.2e}", "0"])
    csv_file.flush()

    model.train()

    while step < args.steps:
        epoch += 1
        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                if args.stage == 1:
                    outputs = model(
                        mode="ntp",
                        doc_ids=batch_dev["doc_ids"],
                        doc_mask=batch_dev["doc_mask"],
                        segment_ids=batch_dev["segment_ids"],
                        segment_mask=batch_dev["segment_mask"],
                        segment_labels=batch_dev["segment_labels"],
                    )
                    loss_key = "ntp_loss"
                else:
                    outputs = model(
                        mode="qa",
                        context_ids=batch_dev["context_ids"],
                        context_mask=batch_dev["context_mask"],
                        prompt_ids=batch_dev["prompt_ids"],
                        prompt_mask=batch_dev["prompt_mask"],
                        response_ids=batch_dev["response_ids"],
                        response_mask=batch_dev["response_mask"],
                        distill_alpha=0.0,
                    )
                    loss_key = "qa_loss"

            loss = outputs["loss"].float() / args.grad_accum  # fp32 backward for stability
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf loss at step {step+1}, stopping training.")
                logger.error("Try: lower --lr or --lora_lr, or use --no_bf16")
                metrics = {"loss": float("nan"), "perplexity": float("inf")}
                csv_file.close()
                return metrics
            loss.backward()
            running_loss += outputs[loss_key].float().item()
            micro_steps += 1

            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    avg = running_loss / micro_steps
                    ppl = math.exp(min(avg, 20))
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {step}/{args.steps}  "
                        f"loss={avg:.4f}  ppl={ppl:.1f}  lr={lr:.2e}  "
                        f"elapsed={elapsed:.0f}s  epoch={epoch}"
                    )
                    csv_writer.writerow([step, f"{avg:.4f}", f"{ppl:.1f}",
                                         f"{lr:.2e}", f"{elapsed:.0f}"])
                    csv_file.flush()

                    if avg < best_loss:
                        best_loss = avg

                    running_loss = 0.0
                    micro_steps = 0

                if step % args.save_every == 0:
                    _save_trainable(model, os.path.join(
                        args.output_dir, f"step_{step}.pt"))

                if step >= args.steps:
                    break

    # Final eval
    metrics = evaluate(model, train_loader, device, args.stage, use_bf16=use_bf16)
    elapsed = time.time() - t0
    logger.info(
        f"[FINAL] train_loss={metrics['loss']:.4f}  "
        f"train_ppl={metrics['perplexity']:.1f}  best_loss={best_loss:.4f}"
    )
    csv_writer.writerow([step, f"{metrics['loss']:.4f}", f"{metrics['perplexity']:.1f}",
                         "", f"{elapsed:.0f}"])
    csv_file.close()
    _save_trainable(model, os.path.join(args.output_dir, "final.pt"))

    logger.info(f"Training done. {step} steps in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Learning curve saved to: {csv_path}")

    return metrics


def _save_trainable(model, path):
    """Save only trainable weights."""
    state = {}
    for k, v in model.state_dict().items():
        if "base_model.base_model.model" in k and "lora" not in k.lower():
            continue
        state[k] = v.cpu()
    torch.save(state, path)
    logger.info(f"  Saved checkpoint: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Memorize a small batch (GPU, ~10 min)")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage: 1=NTP, 2=QA (default: 1)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Data path (default: auto-detect based on stage)")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="Number of samples to use (default: 32)")

    parser.add_argument("--steps", type=int, default=2000,
                        help="Total training steps (default: 2000)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate for perceiver/tokens (default: 1e-4)")
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="LoRA learning rate (default: lr/5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup steps (default: 100)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "constant"])

    parser.add_argument("--log_every", type=int, default=20,
                        help="Log every N steps (default: 20)")
    parser.add_argument("--save_every", type=int, default=500,
                        help="Save every N steps (default: 500)")

    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 mixed precision")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/overfit_step2")

    args = parser.parse_args()

    # Default LoRA lr = lr / 5 (matches main trainer's 5:1 ratio)
    if args.lora_lr is None:
        args.lora_lr = args.lr / 5

    # Auto-detect data path based on stage
    if args.data_path is None:
        if args.stage == 1:
            candidates = [
                "data/ntp_tiny.jsonl",
                "data/ntp_train.jsonl",
                "../deep_compressor/data/ntp_tiny.jsonl",
                "../deep_compressor/data/ntp_train.jsonl",
            ]
        else:
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
                args.data_path = p
                break
        if args.data_path is None:
            logger.error("No data found. Provide --data_path.")
            sys.exit(1)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    use_bf16 = (device.type == "cuda" and not args.no_bf16
                and torch.cuda.is_bf16_supported())

    mode_str = "NTP (Stage 1)" if args.stage == 1 else "QA (Stage 2)"
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info(f"bf16: {'ON' if use_bf16 else 'OFF'}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Steps: {args.steps}  LR: {args.lr}  LoRA_LR: {args.lora_lr}  "
                f"Batch: {args.batch_size}x{args.grad_accum} "
                f"(effective {args.batch_size * args.grad_accum})")

    # Load config & model
    config = Config.from_yaml(args.config)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Tokenizer
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        config.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset: first N samples
    if args.stage == 1:
        full_ds = NTPDataset(
            args.data_path, tokenizer,
            max_context_length=config.data.max_context_length,
            ntp_segment_len=config.data.ntp_segment_len,
        )
        collator = NTPCollator(pad_token_id=tokenizer.pad_token_id)
    else:
        full_ds = QADataset(
            args.data_path, tokenizer,
            max_context_length=config.data.max_context_length,
            max_prompt_length=config.data.max_prompt_length,
            max_response_length=config.data.max_response_length,
        )
        collator = QACollator(pad_token_id=tokenizer.pad_token_id)

    n_use = min(args.num_samples, len(full_ds))
    train_ds = Subset(full_ds, list(range(n_use)))
    logger.info(f"Dataset: {len(full_ds)} total, using {n_use} samples (no val)")

    num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )

    # Model
    logger.info(f"Loading {config.model.base_model} + TaskCompressor...")
    model = TaskCompressorModel(config.model, torch_dtype=torch_dtype)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.1f}%)")

    # Train
    metrics = train(model, train_loader, device, args, use_bf16=use_bf16)

    # Summary
    print("\n" + "=" * 60)
    print(f"  STEP 2: MEMORIZE {n_use} SAMPLES ({mode_str})")
    print("=" * 60)
    print(f"  Device:       {device} {'(bf16)' if use_bf16 else '(fp32)'}")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Data:         {args.data_path} ({n_use} samples)")
    print(f"  Batch:        {args.batch_size}x{args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Steps:        {args.steps}")
    print(f"  Final loss:   {metrics['loss']:.4f}")
    print(f"  Final PPL:    {metrics['perplexity']:.1f}")
    print(f"  Output:       {args.output_dir}/")
    print(f"  Curve CSV:    {args.output_dir}/learning_curve.csv")

    if metrics["perplexity"] < 5:
        print("\n  PASS: PPL < 5 — model memorized the batch")
    elif metrics["perplexity"] < 50:
        print("\n  PARTIAL: PPL 5-50 — learning but not fully memorized, try more steps")
    else:
        print("\n  FAIL: PPL > 50 — not memorizing, investigate architecture")
    print("=" * 60)


if __name__ == "__main__":
    main()

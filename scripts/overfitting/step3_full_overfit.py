#!/usr/bin/env python3
"""Step 3: Full overfit test on ablation data with train/val split.

Train Task Compressor on ablation data (or custom data) with 90/10 train/val split.
Supports both Stage 1 (NTP) and Stage 2 (QA) modes.
Stage 2 includes optional distillation and can resume from Stage 1 checkpoint.

Usage:
  # Stage 1 (NTP) — default
  python scripts/overfitting/step3_full_overfit.py --stage 1

  # Stage 2 (QA), resume from Stage 1
  python scripts/overfitting/step3_full_overfit.py --stage 2 \
      --checkpoint outputs/overfit_step3/best.pt

  # Stage 2 with distillation
  python scripts/overfitting/step3_full_overfit.py --stage 2 --distill_alpha 0.5

  # Custom data and steps
  python scripts/overfitting/step3_full_overfit.py --stage 1 \
      --data_path ../deep_compressor/data/ntp_tiny.jsonl --steps 5000
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
logger = logging.getLogger("step3_full_overfit")


# ── Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, stage, use_bf16=False, distill_alpha=0.0):
    """Compute average loss and perplexity on a loader."""
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
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
                task_loss_key = "ntp_loss"
            else:
                # Build teacher inputs if distillation is on
                teacher_kwargs = {}
                if distill_alpha > 0 and "teacher_input_ids" in batch_dev:
                    teacher_kwargs.update(
                        teacher_input_ids=batch_dev["teacher_input_ids"],
                        teacher_attention_mask=batch_dev["teacher_attention_mask"],
                        response_starts=batch_dev["response_starts"],
                        response_lens=batch_dev["response_lens"],
                    )
                outputs = model(
                    mode="qa",
                    context_ids=batch_dev["context_ids"],
                    context_mask=batch_dev["context_mask"],
                    prompt_ids=batch_dev["prompt_ids"],
                    prompt_mask=batch_dev["prompt_mask"],
                    response_ids=batch_dev["response_ids"],
                    response_mask=batch_dev["response_mask"],
                    distill_alpha=distill_alpha,
                    **teacher_kwargs,
                )
                task_loss_key = "qa_loss"

        bs = next(v for v in batch_dev.values() if isinstance(v, torch.Tensor)).shape[0]
        total_loss += outputs["loss"].float().item() * bs
        total_task_loss += outputs[task_loss_key].float().item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_task_loss = total_task_loss / max(total_samples, 1)
    ppl = math.exp(min(avg_task_loss, 20))
    model.train()
    return {"loss": avg_loss, "task_loss": avg_task_loss, "perplexity": ppl}


# ── Training loop ─────────────────────────────────────────────────────

def train(model, train_loader, val_loader, device, args, use_bf16=False):
    """Training loop with periodic val evaluation."""

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

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
    csv_writer.writerow([
        "step", "train_loss", "val_loss", "val_task_loss", "val_ppl",
        "lr", "elapsed_s"
    ])

    step = 0
    epoch = 0
    running_loss = 0.0
    micro_steps = 0
    best_ppl = float("inf")
    t0 = time.time()

    if use_bf16:
        logger.info("Using bf16 mixed precision (autocast)")

    # Initial val eval
    if val_loader is not None:
        metrics = evaluate(model, val_loader, device, args.stage,
                           use_bf16=use_bf16, distill_alpha=args.distill_alpha)
        logger.info(f"[init] val_loss={metrics['task_loss']:.4f}  val_ppl={metrics['perplexity']:.1f}")
        csv_writer.writerow([0, "", f"{metrics['task_loss']:.4f}",
                             f"{metrics['task_loss']:.4f}", f"{metrics['perplexity']:.1f}",
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
                    task_loss_key = "ntp_loss"
                else:
                    teacher_kwargs = {}
                    if args.distill_alpha > 0 and "teacher_input_ids" in batch_dev:
                        teacher_kwargs.update(
                            teacher_input_ids=batch_dev["teacher_input_ids"],
                            teacher_attention_mask=batch_dev["teacher_attention_mask"],
                            response_starts=batch_dev["response_starts"],
                            response_lens=batch_dev["response_lens"],
                        )
                    outputs = model(
                        mode="qa",
                        context_ids=batch_dev["context_ids"],
                        context_mask=batch_dev["context_mask"],
                        prompt_ids=batch_dev["prompt_ids"],
                        prompt_mask=batch_dev["prompt_mask"],
                        response_ids=batch_dev["response_ids"],
                        response_mask=batch_dev["response_mask"],
                        distill_alpha=args.distill_alpha,
                        **teacher_kwargs,
                    )
                    task_loss_key = "qa_loss"

            loss = outputs["loss"] / args.grad_accum
            loss.backward()
            running_loss += outputs[task_loss_key].float().item()
            micro_steps += 1

            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                # Logging
                if step % args.log_every == 0:
                    avg = running_loss / micro_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {step}/{args.steps}  "
                        f"loss={avg:.4f}  lr={lr:.2e}  "
                        f"elapsed={elapsed:.0f}s  epoch={epoch}"
                    )
                    running_loss = 0.0
                    micro_steps = 0

                # Val eval
                if val_loader is not None and step % args.eval_every == 0:
                    metrics = evaluate(model, val_loader, device, args.stage,
                                       use_bf16=use_bf16, distill_alpha=args.distill_alpha)
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]

                    improved = ""
                    if metrics["perplexity"] < best_ppl:
                        best_ppl = metrics["perplexity"]
                        improved = " ** best **"
                        _save_trainable(model, os.path.join(args.output_dir, "best.pt"))

                    logger.info(
                        f"[EVAL] step {step}  val_loss={metrics['task_loss']:.4f}  "
                        f"val_ppl={metrics['perplexity']:.1f}  "
                        f"best_ppl={best_ppl:.1f}{improved}"
                    )
                    csv_writer.writerow([
                        step, "", f"{metrics['loss']:.4f}",
                        f"{metrics['task_loss']:.4f}", f"{metrics['perplexity']:.1f}",
                        f"{lr:.2e}", f"{elapsed:.0f}",
                    ])
                    csv_file.flush()
                    model.train()

                # Save checkpoint
                if step % args.save_every == 0:
                    _save_trainable(model, os.path.join(
                        args.output_dir, f"step_{step}.pt"))

                if step >= args.steps:
                    break

    # Final eval
    if val_loader is not None:
        metrics = evaluate(model, val_loader, device, args.stage,
                           use_bf16=use_bf16, distill_alpha=args.distill_alpha)
        elapsed = time.time() - t0
        if metrics["perplexity"] < best_ppl:
            best_ppl = metrics["perplexity"]
            _save_trainable(model, os.path.join(args.output_dir, "best.pt"))
        logger.info(
            f"[FINAL] val_loss={metrics['task_loss']:.4f}  "
            f"val_ppl={metrics['perplexity']:.1f}  best_ppl={best_ppl:.1f}"
        )
        csv_writer.writerow([
            step, "", f"{metrics['loss']:.4f}",
            f"{metrics['task_loss']:.4f}", f"{metrics['perplexity']:.1f}",
            "", f"{elapsed:.0f}",
        ])

    csv_file.close()
    _save_trainable(model, os.path.join(args.output_dir, "final.pt"))

    elapsed = time.time() - t0
    logger.info(f"Training done. {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    logger.info(f"Best val PPL: {best_ppl:.1f}")
    logger.info(f"Learning curve saved to: {csv_path}")

    return best_ppl


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
        description="Step 3: Full overfit test with train/val split")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Training stage: 1=NTP, 2=QA (default: 1)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Data path (default: auto-detect based on stage)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")

    parser.add_argument("--steps", type=int, default=20000,
                        help="Total training steps (default: 20000)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--warmup", type=int, default=500,
                        help="Warmup steps (default: 500)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "constant"])

    parser.add_argument("--distill_alpha", type=float, default=0.0,
                        help="Distillation weight (0 = no distillation, only for stage 2)")

    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N steps (default: 50)")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate val PPL every N steps (default: 500)")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N steps (default: 5000)")

    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 mixed precision")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/overfit_step3")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint (e.g. Stage 1 checkpoint for Stage 2)")

    args = parser.parse_args()

    # Distillation only makes sense for stage 2
    if args.stage == 1 and args.distill_alpha > 0:
        logger.warning("Distillation is only available for Stage 2 (QA). Setting distill_alpha=0.")
        args.distill_alpha = 0.0

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
    logger.info(f"Steps: {args.steps}  LR: {args.lr}  "
                f"Batch: {args.batch_size}x{args.grad_accum} "
                f"(effective {args.batch_size * args.grad_accum})")
    if args.stage == 2:
        logger.info(f"Distillation alpha: {args.distill_alpha}")

    # Load config & model
    config = Config.from_yaml(args.config)
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Tokenizer
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        config.model.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset & split
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

    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_ds = Subset(full_ds, list(range(n_train)))
    val_ds = Subset(full_ds, list(range(n_train, n_total)))
    logger.info(f"Dataset: {n_total} total -> {n_train} train / {n_val} val")

    num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )

    # Model
    logger.info(f"Loading {config.model.base_model} + TaskCompressor...")
    model = TaskCompressorModel(config.model, torch_dtype=torch_dtype)

    if args.checkpoint:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.1f}%)")

    # Train
    best_ppl = train(model, train_loader, val_loader, device, args,
                     use_bf16=use_bf16)

    # Summary
    print("\n" + "=" * 60)
    print(f"  STEP 3: FULL OVERFIT TEST WITH TRAIN/VAL SPLIT ({mode_str})")
    print("=" * 60)
    print(f"  Device:       {device} {'(bf16)' if use_bf16 else '(fp32)'}")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Data:         {args.data_path} ({n_train} train / {n_val} val)")
    print(f"  Batch:        {args.batch_size}x{args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    if args.stage == 2:
        print(f"  Distill:      alpha={args.distill_alpha}")
    print(f"  Steps:        {args.steps}")
    print(f"  Best val PPL: {best_ppl:.1f}")
    print(f"  Output:       {args.output_dir}/")
    print(f"  Curve CSV:    {args.output_dir}/learning_curve.csv")

    if best_ppl < 50:
        print("\n  PASS: val PPL < 50 — compression is working, prefix carries info")
    elif best_ppl < 200:
        print("\n  PARTIAL: val PPL 50-200 — partial compression, may need more steps")
    else:
        print("\n  FAIL: val PPL > 200 — compression not effective, investigate architecture")
    print("=" * 60)


if __name__ == "__main__":
    main()

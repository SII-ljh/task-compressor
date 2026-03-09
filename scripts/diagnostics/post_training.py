#!/usr/bin/env python
"""Post-training diagnostic experiments for Task Compressor.

Exp 6 — Compression Fidelity: Multi-batch cosine similarity between compressed and original.
Exp 7 — Context Length Scaling: Loss vs context token length.

Requires a trained checkpoint and evaluation data.

Usage:
    python scripts/diagnostics/post_training.py \
        --config configs/default.yaml \
        --checkpoint outputs/checkpoint.pt \
        --experiments 6,7
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scripts.diagnostics.common import (
    auto_detect_data_path,
    base_parser,
    detect_device,
    detect_dtype,
    finish_wandb,
    init_wandb,
    load_model,
    log_wandb,
    prepare_qa_loader,
    to_device,
)
from task_compressor.config import Config
from task_compressor.data import QACollator, QADataset


ALL_EXPERIMENTS = ["6", "7"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 6 — Compression Fidelity
# ═══════════════════════════════════════════════════════════════════════

def run_compression_fidelity(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Measure how well compressed soft prompts preserve context information."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 6: Compression Fidelity (QA)")
    print("=" * 76)

    model.eval()

    cos_sims = []
    random_cos_sims = []
    compressed_norms = []
    encoder_norms = []

    num_batches = 0
    for batch in eval_loader:
        if num_batches >= max_batches:
            break
        batch = to_device(batch, device)
        num_batches += 1

        with torch.no_grad():
            encoder_hidden = model.encode(
                batch["context_ids"], batch["context_mask"]
            )
            encoder_mask = batch["context_mask"]
            compressed = model.compress(
                encoder_hidden, encoder_mask,
                batch["prompt_ids"], batch["prompt_mask"],
            )

            # Mean pooled representations
            compressed_pooled = compressed.float().mean(dim=1)
            mask_f = encoder_mask.unsqueeze(-1).float()
            encoder_pooled = (
                (encoder_hidden.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            )

            # Cosine similarity
            cos_sim = F.cosine_similarity(compressed_pooled, encoder_pooled, dim=-1)
            cos_sims.append(cos_sim.mean().item())

            # Random baseline
            random_comp = torch.randn_like(compressed) * 0.02
            random_pooled = random_comp.float().mean(dim=1)
            random_cos = F.cosine_similarity(random_pooled, encoder_pooled, dim=-1)
            random_cos_sims.append(random_cos.mean().item())

            compressed_norms.append(compressed.float().norm(dim=-1).mean().item())
            encoder_norms.append(encoder_hidden.float().norm(dim=-1).mean().item())

    if num_batches == 0:
        print("  WARNING: No batches processed")
        return {}

    avg_cos = sum(cos_sims) / len(cos_sims)
    avg_random = sum(random_cos_sims) / len(random_cos_sims)
    avg_comp_norm = sum(compressed_norms) / len(compressed_norms)
    avg_enc_norm = sum(encoder_norms) / len(encoder_norms)
    improvement = avg_cos - avg_random

    print(f"\n  Aggregated over {num_batches} batches:")
    print(f"\n  {'Metric':<45}  {'Value':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'Global cosine sim (compressed↔encoder)':<45}  {avg_cos:>12.4f}")
    print(f"  {'Random baseline cosine sim':<45}  {avg_random:>12.4f}")
    print(f"  {'Improvement over random':<45}  {improvement:>+12.4f}")
    print(f"  {'Compressed norm (mean)':<45}  {avg_comp_norm:>12.4f}")
    print(f"  {'Encoder hidden norm (mean)':<45}  {avg_enc_norm:>12.4f}")
    print(f"  {'─' * 60}")

    print(f"\n  DIAGNOSIS:")
    if improvement > 0.1:
        print("  PASS — Compressed representation preserves meaningful context structure")
    elif improvement > 0:
        print("  MARGINAL — Some structured information, but weak")
    else:
        print("  FAIL — Compressed no better than random noise")

    return {
        "global_cos_sim": avg_cos,
        "random_cos_sim": avg_random,
        "improvement_over_random": improvement,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 7 — Context Length Scaling
# ═══════════════════════════════════════════════════════════════════════

def run_length_scaling(
    model,
    data_path: str,
    config: Config,
    device: torch.device,
    length_buckets: list[int] | None = None,
    samples_per_bucket: int = 20,
    tokenizer=None,
) -> dict:
    """Measure loss as a function of context length."""

    print("\n" + "=" * 76)
    print("  EXPERIMENT 7: Context Length Scaling (QA)")
    print("=" * 76)

    model.eval()

    if length_buckets is None:
        length_buckets = [256, 512, 1024, 2048, 4096]

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    ds = QADataset(
        data_path,
        tokenizer,
        max_context_length=max(length_buckets) + 256,
        max_prompt_length=config.data.max_prompt_length,
        max_response_length=config.data.max_response_length,
    )
    collator = QACollator(pad_token_id=tokenizer.pad_token_id)
    len_key = "context_ids"

    print(f"\n  Scanning {len(ds)} samples for length distribution...")
    buckets: dict[int, list[int]] = {b: [] for b in length_buckets}

    for idx in range(len(ds)):
        sample = ds[idx]
        ctx_len = sample[len_key].shape[0]
        for b in sorted(length_buckets):
            if ctx_len >= b and len(buckets[b]) < samples_per_bucket:
                buckets[b].append(idx)
                break

    results = {}
    print(
        f"\n  {'Bucket':<12}  {'Samples':>8}  {'Mean Loss':>12}  {'Std Loss':>12}"
    )
    print(f"  {'─' * 50}")

    bucket_losses = []
    bucket_lengths = []

    for bucket_len in sorted(length_buckets):
        indices = buckets[bucket_len]
        if len(indices) < 2:
            print(f"  {bucket_len:<12}  {len(indices):>8}  {'(skip)':>12}")
            continue

        subset = Subset(ds, indices)
        loader = DataLoader(
            subset, batch_size=min(4, len(indices)),
            shuffle=False, collate_fn=collator
        )

        batch_losses = []
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                outputs = model(
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    prompt_ids=batch["prompt_ids"],
                    prompt_mask=batch["prompt_mask"],
                    response_ids=batch["response_ids"],
                    response_mask=batch["response_mask"],
                )
                batch_losses.append(outputs["qa_loss"].item())

        mean_loss = sum(batch_losses) / len(batch_losses)
        std_loss = (
            (sum((l - mean_loss) ** 2 for l in batch_losses) / len(batch_losses))
            ** 0.5
            if len(batch_losses) > 1
            else 0.0
        )

        results[f"loss_{bucket_len}"] = mean_loss
        results[f"std_{bucket_len}"] = std_loss
        bucket_losses.append(mean_loss)
        bucket_lengths.append(bucket_len)

        print(f"  {bucket_len:<12}  {len(indices):>8}  {mean_loss:>12.4f}  {std_loss:>12.4f}")

    print(f"  {'─' * 50}")

    if len(bucket_losses) >= 2:
        import math
        log_lengths = [math.log(l) for l in bucket_lengths]
        n = len(log_lengths)
        mean_x = sum(log_lengths) / n
        mean_y = sum(bucket_losses) / n
        cov_xy = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(log_lengths, bucket_losses)
        )
        var_x = sum((x - mean_x) ** 2 for x in log_lengths)
        slope = cov_xy / max(var_x, 1e-10)
        results["degradation_slope"] = slope

        print(f"\n  Degradation slope (loss vs log(length)): {slope:+.4f}")
        print(f"\n  DIAGNOSIS:")
        if slope < 0.05:
            print("  PASS — Loss stable across context lengths")
        elif slope < 0.2:
            print("  INFO — Moderate degradation with length")
        else:
            print("  WARNING — Significant loss increase with context length")
    else:
        print("\n  WARNING: Not enough buckets with data for slope analysis")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Task Compressor post-training diagnostics (Exp 6-7)")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint",
    )
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Max batches per experiment")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    parser.add_argument(
        "--experiments", type=str, default="6,7",
        help="Comma-separated list of experiments to run (6-7)",
    )
    parser.add_argument(
        "--length_buckets", type=str, default="256,512,1024,2048,4096",
        help="Comma-separated context length buckets for Exp 7",
    )
    parser.set_defaults(wandb_project="tc-diagnostic-post")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]

    device = detect_device()
    torch_dtype = detect_dtype(device, args.no_bf16)

    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name="post-training-diag",
        config={"checkpoint": args.checkpoint,
                "experiments": experiments},
        entity=args.wandb_entity,
    )

    print(f"Device: {device}  dtype: {torch_dtype}")
    config = Config.from_yaml(args.config)

    print("\nLoading model with checkpoint...")
    model = load_model(config, checkpoint_path=args.checkpoint,
                       device=device, torch_dtype=torch_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    data_path = args.data_path
    if data_path is None:
        data_path = auto_detect_data_path()
        if data_path is None:
            print("ERROR: No data found. Provide --data_path.")
            sys.exit(1)

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "6" in experiments:
        eval_loader, tokenizer = prepare_qa_loader(
            config, data_path, args.batch_size,
            max_samples=args.max_samples,
        )
        print(f"\n  Eval loader: {len(eval_loader)} batches")

        torch.manual_seed(42)
        exp6 = run_compression_fidelity(
            model, eval_loader, device,
            max_batches=args.max_batches,
        )
        all_results["fidelity"] = exp6
        if wandb_run:
            log_wandb(wandb_run, {f"exp6/{k}": v for k, v in exp6.items()})

    if "7" in experiments:
        torch.manual_seed(42)
        length_buckets = [int(x) for x in args.length_buckets.split(",")]
        exp7 = run_length_scaling(
            model, data_path, config, device,
            length_buckets=length_buckets,
        )
        all_results["length_scaling"] = exp7
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp7/{k}": v for k, v in exp7.items()
                 if isinstance(v, (int, float))},
            )

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

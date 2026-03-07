#!/usr/bin/env python
"""Post-training diagnostic experiments for Task Compressor.

Exp 6 — Compression Fidelity: Multi-batch cosine similarity between compressed and original.
Exp 7 — Context Length Scaling: QA loss vs context token length.
Exp 8 — Distillation Effectiveness: KL divergence, token prediction agreement.

Requires a trained checkpoint and evaluation data.

Usage:
    python scripts/diagnostics/post_training.py \
        --config configs/default.yaml \
        --checkpoint outputs/checkpoint.pt \
        --data_path data/qa_dev.json \
        --experiments 6,7,8
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scripts.diagnostics.common import (
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
from task_compressor.data import QADataset, QACollator


ALL_EXPERIMENTS = ["6", "7", "8"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 6 — Compression Fidelity
# ═══════════════════════════════════════════════════════════════════════

def run_compression_fidelity(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Measure how well compressed soft prompts preserve context information.

    Compares cosine similarity between compressed representations and
    encoder hidden states against random baselines.
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 6: Compression Fidelity")
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
            # Encode context
            encoder_hidden = model.encode(
                batch["context_ids"], batch["context_mask"]
            )

            # Compress
            compressed = model.compress(
                encoder_hidden, batch["context_mask"],
                batch["prompt_ids"], batch["prompt_mask"],
            )

            # Mean pooled representations
            compressed_pooled = compressed.float().mean(dim=1)
            mask_f = batch["context_mask"].unsqueeze(-1).float()
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
    """Measure QA loss as a function of context length.

    Loads QA data, buckets by context token count, and runs forward pass per bucket.
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 7: Context Length Scaling")
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

    print(f"\n  Scanning {len(ds)} samples for length distribution...")
    buckets: dict[int, list[int]] = {b: [] for b in length_buckets}

    for idx in range(len(ds)):
        sample = ds[idx]
        ctx_len = sample["context_ids"].shape[0]
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
                    distill_alpha=0.0,
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
#  Experiment 8 — Distillation Effectiveness
# ═══════════════════════════════════════════════════════════════════════

def run_distillation_effectiveness(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Compare student and teacher outputs on evaluation data.

    Measures:
      - KL divergence between student and teacher logit distributions
      - Token prediction agreement (argmax match rate)
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 8: Distillation Effectiveness")
    print("=" * 76)

    model.eval()

    kl_divs = []
    agreement_rates = []

    num_batches = 0
    for batch in eval_loader:
        if num_batches >= max_batches:
            break

        # Skip if no teacher fields
        if "teacher_input_ids" not in batch:
            continue

        batch = to_device(batch, device)
        num_batches += 1

        with torch.no_grad():
            # Student path
            encoder_hidden = model.encode(
                batch["context_ids"], batch["context_mask"]
            )
            compressed = model.compress(
                encoder_hidden, batch["context_mask"],
                batch["prompt_ids"], batch["prompt_mask"],
            )
            dec_out = model.decode_train(
                compressed,
                batch["prompt_ids"], batch["prompt_mask"],
                batch["response_ids"], batch["response_mask"],
            )

            # Teacher path
            teacher_logits = model.teacher_forward(
                batch["teacher_input_ids"],
                batch["teacher_attention_mask"],
            )

            # Extract response-predicting logits
            prefix_len = dec_out["prefix_len"]
            L_t = batch["response_ids"].shape[1]
            student_resp = dec_out["logits"][:, prefix_len - 1: prefix_len - 1 + L_t, :]

            B = batch["context_ids"].shape[0]
            V = student_resp.shape[-1]
            teacher_resp = torch.zeros(B, L_t, V, device=device, dtype=student_resp.dtype)
            for i in range(B):
                start = batch["response_starts"][i].item() - 1
                length = min(batch["response_lens"][i].item(), L_t)
                teacher_resp[i, :length] = teacher_logits[i, start: start + length].to(student_resp.dtype)

            resp_mask = batch["response_mask"].float()

            # KL divergence
            T = 2.0
            student_log_p = F.log_softmax(student_resp / T, dim=-1)
            teacher_p = F.softmax(teacher_resp / T, dim=-1)
            kl = F.kl_div(student_log_p, teacher_p, reduction="none").sum(-1)
            kl = (kl * resp_mask).sum() / resp_mask.sum().clamp(min=1)
            kl_divs.append(kl.item())

            # Token prediction agreement
            student_preds = student_resp.argmax(dim=-1)
            teacher_preds = teacher_resp.argmax(dim=-1)
            match = (student_preds == teacher_preds).float() * resp_mask
            agreement = (match.sum() / resp_mask.sum().clamp(min=1)).item()
            agreement_rates.append(agreement)

    if num_batches == 0:
        print("  WARNING: No batches with teacher data found")
        return {}

    avg_kl = sum(kl_divs) / len(kl_divs)
    avg_agree = sum(agreement_rates) / len(agreement_rates)

    print(f"\n  Aggregated over {num_batches} batches:")
    print(f"\n  {'Metric':<45}  {'Value':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'KL divergence (student || teacher)':<45}  {avg_kl:>12.4f}")
    print(f"  {'Token prediction agreement':<45}  {avg_agree:>12.2%}")
    print(f"  {'─' * 60}")

    print(f"\n  DIAGNOSIS:")
    if avg_agree > 0.8:
        print(f"  PASS — High teacher-student agreement ({avg_agree:.1%})")
    elif avg_agree > 0.5:
        print(f"  INFO — Moderate agreement ({avg_agree:.1%})")
    else:
        print(f"  WARNING — Low agreement ({avg_agree:.1%})")

    if avg_kl < 1.0:
        print(f"  KL divergence is low ({avg_kl:.4f}) — distributions are close")
    elif avg_kl < 5.0:
        print(f"  KL divergence is moderate ({avg_kl:.4f})")
    else:
        print(f"  KL divergence is high ({avg_kl:.4f}) — significant distribution gap")

    return {
        "kl_divergence": avg_kl,
        "token_agreement": avg_agree,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Task Compressor post-training diagnostics (Exp 6-8)")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained checkpoint",
    )
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Max batches per experiment")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    parser.add_argument(
        "--experiments", type=str, default="6,7,8",
        help="Comma-separated list of experiments to run (6-8)",
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
        config={"checkpoint": args.checkpoint, "experiments": experiments},
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
        for p in ["data/qa_dev.json", "data/qa_tiny.json"]:
            if os.path.exists(p):
                data_path = p
                break
        if data_path is None:
            print("ERROR: No QA data found. Provide --data_path.")
            sys.exit(1)

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if any(e in experiments for e in ("6", "8")):
        eval_loader, tokenizer = prepare_qa_loader(
            config, data_path, args.batch_size,
            max_samples=args.max_samples,
        )
        print(f"\n  QA eval loader: {len(eval_loader)} batches")

    if "6" in experiments:
        torch.manual_seed(42)
        exp6 = run_compression_fidelity(
            model, eval_loader, device, max_batches=args.max_batches
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

    if "8" in experiments:
        torch.manual_seed(42)
        exp8 = run_distillation_effectiveness(
            model, eval_loader, device, max_batches=args.max_batches
        )
        all_results["distillation"] = exp8
        if wandb_run:
            log_wandb(wandb_run, {f"exp8/{k}": v for k, v in exp8.items()})

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

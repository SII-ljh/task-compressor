#!/usr/bin/env python
"""Mid-training diagnostic experiments for Task Compressor.

Exp 4 — Query Diversity/Utilization: Check query collapse, dead queries, effective rank.
Exp 5 — Perceiver Layer-wise Information Gain: Measure loss at each layer's output.

Supports both Stage 1 (NTP) and Stage 2 (QA) modes.

Usage (standalone):
    # Stage 1 (NTP)
    python scripts/diagnostics/mid_training.py --stage 1 \
        --config configs/default.yaml --experiments 4,5

    # Stage 2 (QA), with checkpoint
    python scripts/diagnostics/mid_training.py --stage 2 \
        --config configs/default.yaml \
        --checkpoint outputs/checkpoint.pt \
        --data_path ../deep_compressor/data/qa_dev.json
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F

from scripts.diagnostics.common import (
    auto_detect_data_path,
    base_parser,
    detect_device,
    detect_dtype,
    effective_rank,
    finish_wandb,
    init_wandb,
    load_model,
    log_wandb,
    pairwise_cosine_similarity,
    prepare_ntp_batch,
    prepare_qa_batch,
    to_device,
)
from task_compressor.config import Config


ALL_EXPERIMENTS = ["4", "5"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 4 — Query Diversity / Utilization
# ═══════════════════════════════════════════════════════════════════════

def run_query_diversity(
    model,
    batch: dict,
    device: torch.device,
    stage: int,
) -> dict:
    """Analyze query diversity and utilization after Perceiver compression.

    Checks for query collapse, dead queries, and effective utilization of
    the soft prompt bottleneck.
    """

    mode_str = "NTP" if stage == 1 else "QA"
    print("\n" + "=" * 76)
    print(f"  EXPERIMENT 4: Query Diversity / Utilization ({mode_str})")
    print("=" * 76)

    model.eval()
    k = model.config.n_prompt_tokens + model.config.n_context_tokens

    with torch.no_grad():
        if stage == 1:
            encoder_hidden = model.encode(batch["doc_ids"], batch["doc_mask"])
            compressed = model.compress_ntp(encoder_hidden, batch["doc_mask"])
        else:
            encoder_hidden = model.encode(batch["context_ids"], batch["context_mask"])
            compressed = model.compress(
                encoder_hidden, batch["context_mask"],
                batch["prompt_ids"], batch["prompt_mask"],
            )
        # (B, k, D)

        # Analyze first sample's compressed vectors
        latent = compressed[0]  # (k, D)

        # Pairwise cosine similarity
        cos_sim = pairwise_cosine_similarity(latent)
        mask = ~torch.eye(k, device=device, dtype=torch.bool)
        off_diag = cos_sim[mask]

        mean_pairwise = off_diag.mean().item()
        max_pairwise = off_diag.max().item()

        # Effective rank
        eff_rank = effective_rank(latent)

        # Dead query detection (norm < 1% of mean)
        query_norms = latent.float().norm(dim=-1)
        norm_mean = query_norms.mean().item()
        norm_std = query_norms.std().item()
        threshold = norm_mean * 0.01
        num_dead = (query_norms < threshold).sum().item()

        # Analyze prompt latents vs context tokens separately
        n_p = model.config.n_prompt_tokens
        n_c = model.config.n_context_tokens
        prompt_part = latent[:n_p]
        context_part = latent[n_p:]

        prompt_rank = effective_rank(prompt_part) if n_p > 1 else 0.0
        context_rank = effective_rank(context_part) if n_c > 1 else 0.0

    results = {
        "mean_pairwise_cosine": mean_pairwise,
        "max_pairwise_cosine": max_pairwise,
        "effective_rank": eff_rank,
        "num_dead_queries": num_dead,
        "query_norm_mean": norm_mean,
        "query_norm_std": norm_std,
        "prompt_latent_rank": prompt_rank,
        "context_token_rank": context_rank,
    }

    # ── report ────────────────────────────────────────────────────────
    print(f"\n  {'Metric':<40}  {'Value':>12}")
    print(f"  {'─' * 55}")
    print(f"  {'Mean pairwise cosine similarity':<40}  {mean_pairwise:>12.4f}")
    print(f"  {'Max pairwise cosine similarity':<40}  {max_pairwise:>12.4f}")
    print(f"  {'Effective rank (all k tokens)':<40}  {eff_rank:>12.2f}")
    print(f"  {'  Prompt latent rank (n_p={n_p})':<40}  {prompt_rank:>12.2f}")
    print(f"  {'  Context token rank (n_c={n_c})':<40}  {context_rank:>12.2f}")
    print(f"  {'Num soft prompts (k)':<40}  {k:>12d}")
    print(f"  {'Dead queries (norm < 1% mean)':<40}  {num_dead:>12d}")
    print(f"  {'Query norm mean':<40}  {norm_mean:>12.4f}")
    print(f"  {'Query norm std':<40}  {norm_std:>12.4f}")
    print(f"  {'─' * 55}")

    # ── diagnosis ─────────────────────────────────────────────────────
    print(f"\n  DIAGNOSIS:")
    warnings = []
    if eff_rank < k * 0.3:
        warnings.append(
            f"  WARNING: Low effective rank ({eff_rank:.1f} < {k * 0.3:.0f}) "
            f"— queries under-utilized"
        )
    if num_dead > 0:
        warnings.append(
            f"  WARNING: {num_dead} dead queries detected — norm near zero"
        )
    if mean_pairwise > 0.9:
        warnings.append(
            f"  WARNING: High mean cosine similarity ({mean_pairwise:.3f} > 0.9) "
            f"— possible query collapse"
        )

    if warnings:
        for w in warnings:
            print(w)
    else:
        print(
            f"  PASS — Queries are diverse (rank={eff_rank:.1f}/{k}, "
            f"cos={mean_pairwise:.3f}, {num_dead} dead)"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 5 — Perceiver Layer-wise Information Gain
# ═══════════════════════════════════════════════════════════════════════

def run_layerwise_info_gain(
    model,
    batch: dict,
    device: torch.device,
    stage: int,
) -> dict:
    """Measure loss at progressive truncation points through the Perceiver layers.

    Runs query through each Perceiver layer incrementally and measures the
    decode loss at each stage.
    """

    mode_str = "NTP" if stage == 1 else "QA"
    print("\n" + "=" * 76)
    print(f"  EXPERIMENT 5: Perceiver Layer-wise Information Gain ({mode_str})")
    print("=" * 76)

    model.eval()
    perceiver = model.perceiver

    if stage == 1:
        B = batch["doc_ids"].shape[0]
        segment_ids = batch["segment_ids"]
        segment_mask = batch["segment_mask"]
        segment_labels = batch["segment_labels"]
    else:
        B = batch["context_ids"].shape[0]

    with torch.no_grad():
        # Encode
        if stage == 1:
            encoder_hidden = model.encode(batch["doc_ids"], batch["doc_mask"])
            encoder_mask = batch["doc_mask"]
            # Build query (same as compress_ntp)
            prompt_latents = model.prompt_encoder.latent_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            encoder_hidden = model.encode(batch["context_ids"], batch["context_mask"])
            encoder_mask = batch["context_mask"]
            # Build query (same as compress)
            prompt_latents = model.prompt_encoder(batch["prompt_ids"], batch["prompt_mask"])

        ctx_tokens = model.context_tokens.unsqueeze(0).expand(B, -1, -1)
        query = torch.cat([prompt_latents, ctx_tokens], dim=1)

        # Helper: compute loss from intermediate query state
        def _loss_from_query(q: torch.Tensor) -> float:
            normed = perceiver.final_ln(q)

            if stage == 1:
                # NTP decode: [compressed | sep | segment]
                gc_was_on = model._pause_gradient_checkpointing()
                model._disable_adapter_forward_only()

                embed_layer = model.get_embedding_layer()
                seg_embeds = embed_layer(segment_ids)
                sep = model.separator.unsqueeze(0).expand(B, -1, -1)
                inputs_embeds = torch.cat([normed, sep, seg_embeds], dim=1)

                k_len = normed.shape[1]
                prefix_mask = torch.ones(B, k_len + 1, device=device, dtype=segment_mask.dtype)
                attn_mask = torch.cat([prefix_mask, segment_mask], dim=1)

                outputs = model.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    return_dict=True,
                )
                logits = outputs.logits

                model._enable_adapter_forward_only()
                model._resume_gradient_checkpointing(gc_was_on)

                L_s = segment_ids.shape[1]
                prefix_len = k_len + 1
                total_len = inputs_embeds.shape[1]
                labels = torch.full((B, total_len), -100, dtype=torch.long, device=device)
                labels[:, prefix_len: prefix_len + L_s] = segment_labels
                pad_positions = segment_mask == 0
                labels[:, prefix_len: prefix_len + L_s][pad_positions] = -100

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
            else:
                dec_out = model.decode_train(
                    normed,
                    batch["prompt_ids"], batch["prompt_mask"],
                    batch["response_ids"], batch["response_mask"],
                )
                shift_logits = dec_out["logits"][:, :-1, :].contiguous()
                shift_labels = dec_out["labels"][:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss.item()

        # Collect truncation points
        checkpoints = {}
        x = query

        # Point 0: initial query (no Perceiver processing)
        checkpoints["initial_query"] = _loss_from_query(x)

        # Run through each layer
        for i, layer in enumerate(perceiver.layers):
            x = layer(x, encoder_hidden, encoder_mask)
            checkpoints[f"after_layer_{i}"] = _loss_from_query(x)

    # ── report ────────────────────────────────────────────────────────
    stages = list(checkpoints.keys())
    losses = list(checkpoints.values())

    print(f"\n  {'Stage':<25}  {'Loss':>10}  {'Gain':>10}  {'Gain %':>10}")
    print(f"  {'─' * 60}")

    gains = {}
    total_gain = losses[0] - losses[-1] if len(losses) > 1 else 0.0

    for i, (stage_name, loss) in enumerate(zip(stages, losses)):
        if i == 0:
            gain = 0.0
            gain_pct = 0.0
        else:
            gain = losses[i - 1] - loss
            gain_pct = (gain / max(abs(total_gain), 1e-10)) * 100
        gains[stage_name] = gain
        print(f"  {stage_name:<25}  {loss:>10.4f}  {gain:>+10.4f}  {gain_pct:>9.1f}%")

    print(f"  {'─' * 60}")
    print(f"  {'Total gain':<25}  {'':>10}  {total_gain:>+10.4f}  {'100.0':>9}%")

    # ── diagnosis ─────────────────────────────────────────────────────
    print(f"\n  DIAGNOSIS:")
    if total_gain < 0.01:
        print("  WARNING: Negligible total gain — Perceiver may not be learning")
    else:
        gain_items = [(k, v) for k, v in gains.items() if k != "initial_query"]
        if gain_items:
            best_stage = max(gain_items, key=lambda x: x[1])
            print(f"  Largest contribution: {best_stage[0]} ({best_stage[1]:+.4f})")

        negative = [(k, v) for k, v in gain_items if v < -0.01]
        if negative:
            for k, v in negative:
                print(f"  WARNING: {k} increases loss by {-v:.4f}")
        else:
            print("  PASS — All Perceiver layers contribute positively")

    results = {"checkpoints": checkpoints, "gains": gains, "total_gain": total_gain}
    for stage_name, loss in checkpoints.items():
        results[f"loss_{stage_name}"] = loss
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Task Compressor mid-training diagnostics (Exp 4-5)")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint (optional, uses random init if not provided)",
    )
    parser.add_argument(
        "--experiments", type=str, default="4,5",
        help="Comma-separated list of experiments to run (4,5)",
    )
    parser.set_defaults(wandb_project="tc-diagnostic-mid")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]
    device = detect_device()
    torch_dtype = detect_dtype(device, args.no_bf16)

    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=f"mid-training-diag-stage{args.stage}",
        config={"stage": args.stage, "checkpoint": args.checkpoint,
                "experiments": experiments},
        entity=args.wandb_entity,
    )

    mode_str = "NTP (Stage 1)" if args.stage == 1 else "QA (Stage 2)"
    print(f"Mode: {mode_str}")
    print(f"Device: {device}  dtype: {torch_dtype}")
    config = Config.from_yaml(args.config)

    print("\nLoading model...")
    model = load_model(config, checkpoint_path=args.checkpoint,
                       device=device, torch_dtype=torch_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    data_path = args.data_path
    if data_path is None:
        data_path = auto_detect_data_path(args.stage)
        if data_path is None:
            print("ERROR: No data found. Provide --data_path.")
            sys.exit(1)

    print(f"\nPreparing single batch from {data_path}...")
    if args.stage == 1:
        batch, tokenizer = prepare_ntp_batch(
            config, data_path, args.batch_size, device
        )
    else:
        batch, tokenizer = prepare_qa_batch(
            config, data_path, args.batch_size, device
        )

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "4" in experiments:
        torch.manual_seed(42)
        exp4 = run_query_diversity(model, batch, device, stage=args.stage)
        all_results["query_diversity"] = exp4
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp4/{k}": v for k, v in exp4.items()
                 if isinstance(v, (int, float))},
            )

    if "5" in experiments:
        torch.manual_seed(42)
        exp5 = run_layerwise_info_gain(model, batch, device, stage=args.stage)
        all_results["layerwise_gain"] = exp5
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp5/{k}": v for k, v in exp5.items()
                 if isinstance(v, (int, float))},
            )

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

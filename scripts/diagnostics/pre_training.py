#!/usr/bin/env python
"""Pre-training diagnostic experiments for Task Compressor.

Exp 1 — Overfitting: Train compression pipeline on single batch, verify loss decreases.
Exp 2 — Gradient Flow: Single forward+backward, report per-module gradient norms.
Exp 3 — Information Bottleneck: Compare random vs learnable soft prompts.

Supports both Stage 1 (NTP) and Stage 2 (QA) modes.

Usage:
    # Stage 1 (NTP) — default
    python scripts/diagnostics/pre_training.py --stage 1 \
        --config configs/default.yaml --experiments 1,2,3

    # Stage 2 (QA)
    python scripts/diagnostics/pre_training.py --stage 2 \
        --config configs/default.yaml --data_path ../deep_compressor/data/qa_dev.json
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from scripts.diagnostics.common import (
    auto_detect_data_path,
    base_parser,
    detect_device,
    detect_dtype,
    finish_wandb,
    grad_norm,
    init_wandb,
    load_model,
    log_wandb,
    prepare_ntp_batch,
    prepare_qa_batch,
    stats_str,
    to_device,
)
from task_compressor.config import Config


ALL_EXPERIMENTS = ["1", "2", "3"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1 — Overfitting
# ═══════════════════════════════════════════════════════════════════════

def run_overfit(
    model,
    batch: dict,
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
    stage: int,
) -> dict:
    """Overfit the full pipeline on one fixed batch."""

    mode_str = "NTP" if stage == 1 else "QA"
    print("\n" + "=" * 76)
    print(f"  EXPERIMENT 1: Overfitting ({mode_str}, 1 batch, {steps} steps, lr={lr})")
    print("=" * 76)

    model.train()

    # Collect trainable parameter groups by component
    modules = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora" in name.lower():
            modules.setdefault("lora", []).append(p)
        elif "perceiver" in name:
            modules.setdefault("perceiver", []).append(p)
        elif "prompt_encoder" in name:
            modules.setdefault("prompt_encoder", []).append(p)
        elif "context_tokens" in name or "separator" in name:
            modules.setdefault("learnable_tokens", []).append(p)
        else:
            modules.setdefault("other", []).append(p)

    all_params = [p for ps in modules.values() for p in ps]
    optimizer = AdamW(all_params, lr=lr, weight_decay=0.01)

    losses: list[float] = []
    grad_norms: dict[str, list[float]] = {n: [] for n in modules}

    mod_names = list(modules.keys())
    hdr = f"  {'step':>5}  {'loss':>10}"
    for n in mod_names:
        hdr += f"  {'|' + n + '|':>14}"
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))

    for step in range(1, steps + 1):
        optimizer.zero_grad()

        if stage == 1:
            outputs = model(
                mode="ntp",
                doc_ids=batch["doc_ids"],
                doc_mask=batch["doc_mask"],
                segment_ids=batch["segment_ids"],
                segment_mask=batch["segment_mask"],
                segment_labels=batch["segment_labels"],
            )
        else:
            outputs = model(
                mode="qa",
                context_ids=batch["context_ids"],
                context_mask=batch["context_mask"],
                prompt_ids=batch["prompt_ids"],
                prompt_mask=batch["prompt_mask"],
                response_ids=batch["response_ids"],
                response_mask=batch["response_mask"],
                distill_alpha=0.0,
            )

        loss = outputs["loss"]
        loss.backward()

        lv = loss.item()
        losses.append(lv)

        # Compute per-group gradient norms
        gn = {}
        for name, params in modules.items():
            sq = 0.0
            for p in params:
                if p.grad is not None:
                    sq += p.grad.data.float().norm(2).item() ** 2
            gn[name] = sq ** 0.5
            grad_norms[name].append(gn[name])

        if step == 1 or step % log_every == 0:
            line = f"  {step:>5}  {lv:>10.4f}"
            for n in mod_names:
                line += f"  {gn[n]:>14.4e}"
            print(line)

        optimizer.step()

    # ── diagnosis ─────────────────────────────────────────────────────
    first, last, best = losses[0], losses[-1], min(losses)
    reduction_pct = (first - last) / max(abs(first), 1e-10) * 100

    W = min(20, max(1, steps // 5))
    if len(losses) > W:
        wins = sum(
            1 for i in range(len(losses) - W) if losses[i + W] < losses[i]
        )
        trend_pct = wins / (len(losses) - W) * 100
    else:
        trend_pct = 100.0 if last < first else 0.0

    print(f"\n  DIAGNOSIS:")
    print(
        f"    loss  {first:.4f} -> {last:.4f}  (min {best:.4f}, {reduction_pct:+.1f}%)"
    )
    print(f"    decreasing-window rate ({W}-step windows): {trend_pct:.0f}%")

    if reduction_pct > 10 and trend_pct > 50:
        print("    PASS     — loss consistently decreases; gradient flow is healthy")
    elif reduction_pct > 5:
        print("    MARGINAL — loss decreasing slowly; consider higher LR or more steps")
    else:
        print("    FAIL     — loss not decreasing; check gradient flow or architecture")

    # gradient health
    for name, gn_list in grad_norms.items():
        avg = sum(gn_list) / len(gn_list)
        if avg < 1e-7:
            print(f"    WARNING  {name}: avg |grad| = {avg:.2e} (near zero)")
        elif avg > 100:
            print(f"    WARNING  {name}: avg |grad| = {avg:.2e} (very large)")

    return {"losses": losses, "grad_norms": grad_norms}


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 2 — Gradient Flow Analysis
# ═══════════════════════════════════════════════════════════════════════

def run_gradient_flow(
    model,
    batch: dict,
    device: torch.device,
    stage: int,
) -> dict:
    """Single forward+backward pass, report per-layer gradient norms."""

    mode_str = "NTP" if stage == 1 else "QA"
    print("\n" + "=" * 76)
    print(f"  EXPERIMENT 2: Gradient Flow Analysis ({mode_str})")
    print("=" * 76)

    model.train()
    model.zero_grad()

    if stage == 1:
        outputs = model(
            mode="ntp",
            doc_ids=batch["doc_ids"],
            doc_mask=batch["doc_mask"],
            segment_ids=batch["segment_ids"],
            segment_mask=batch["segment_mask"],
            segment_labels=batch["segment_labels"],
        )
    else:
        outputs = model(
            mode="qa",
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            response_ids=batch["response_ids"],
            response_mask=batch["response_mask"],
            distill_alpha=0.0,
        )
    outputs["loss"].backward()

    # Collect per-module, per-layer gradient norms
    results = {}

    print(
        f"\n  {'Module':<30}  {'Layer':<35}  {'|grad|':>12}  {'Status'}"
    )
    print(f"  {'─' * 95}")

    for mod_name, module in [
        ("lora (via base_model)", model.base_model),
        ("perceiver", model.perceiver),
        ("prompt_encoder", model.prompt_encoder),
    ]:
        layer_norms = {}
        for name, param in module.named_parameters():
            if param.grad is not None and param.requires_grad:
                gn = param.grad.data.float().norm(2).item()
                layer_norms[name] = gn

                if gn < 1e-8:
                    status = "VANISHING"
                elif gn > 100:
                    status = "EXPLODING"
                else:
                    status = "OK"

                # Only print weight layers and problematic parameters
                if gn < 1e-7 or gn > 10 or "weight" in name:
                    short_name = name[:35]
                    print(
                        f"  {mod_name:<30}  {short_name:<35}  {gn:>12.4e}  {status}"
                    )

        results[mod_name] = layer_norms

    # Summary
    all_norms = [
        gn for layer_norms in results.values() for gn in layer_norms.values()
    ]
    if all_norms:
        min_gn = min(all_norms)
        max_gn = max(all_norms)
        mean_gn = sum(all_norms) / len(all_norms)
        print(f"\n  Summary: min={min_gn:.2e}, max={max_gn:.2e}, mean={mean_gn:.2e}")

        vanishing = sum(1 for g in all_norms if g < 1e-7)
        exploding = sum(1 for g in all_norms if g > 100)
        if vanishing > 0:
            print(
                f"  WARNING: {vanishing} parameters with vanishing gradients (<1e-7)"
            )
        if exploding > 0:
            print(
                f"  WARNING: {exploding} parameters with exploding gradients (>100)"
            )
        if vanishing == 0 and exploding == 0:
            print("  PASS — No gradient pathologies detected")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 3 — Information Bottleneck
# ═══════════════════════════════════════════════════════════════════════

def run_bottleneck(
    model,
    batch: dict,
    device: torch.device,
    steps: int,
    lr: float,
    log_every: int,
    stage: int,
) -> dict:
    """Compare random soft prompts vs pipeline-compressed soft prompts.

    Tests whether the Perceiver actually extracts useful information from
    the encoder hidden states, or if learnable tokens alone suffice.

    For Stage 1 (NTP): uses compress_ntp + decode with segment
    For Stage 2 (QA): uses compress + decode with prompt+response
    """

    mode_str = "NTP" if stage == 1 else "QA"
    print("\n" + "=" * 76)
    print(f"  EXPERIMENT 3: Information Bottleneck ({mode_str}, Random vs Compressed)")
    print("=" * 76)

    model.eval()
    k = model.config.n_prompt_tokens + model.config.n_context_tokens
    D = model.hidden_size

    if stage == 1:
        B = batch["doc_ids"].shape[0]
        doc_ids = batch["doc_ids"]
        doc_mask = batch["doc_mask"]
        segment_ids = batch["segment_ids"]
        segment_mask = batch["segment_mask"]
        segment_labels = batch["segment_labels"]
    else:
        B = batch["context_ids"].shape[0]

    # Helper: compute loss from soft prompts
    def _loss_from_compressed(compressed: torch.Tensor) -> float:
        if stage == 1:
            # NTP decode: [compressed | sep | segment]
            gc_was_on = model._pause_gradient_checkpointing()
            model._disable_adapter_forward_only()

            embed_layer = model.get_embedding_layer()
            seg_embeds = embed_layer(segment_ids)
            sep = model.separator.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([compressed, sep, seg_embeds], dim=1)

            k_len = compressed.shape[1]
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

            # Build labels
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
                compressed,
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

    # ── baseline: random soft prompts ────────────────────────────────
    with torch.no_grad():
        random_compressed = torch.randn(B, k, D, device=device, dtype=next(model.parameters()).dtype) * 0.02
        random_loss = _loss_from_compressed(random_compressed)
    print(f"\n  Random soft-prompt baseline loss: {random_loss:.4f}")

    # ── pipeline: encode + compress ──────────────────────────────────
    with torch.no_grad():
        if stage == 1:
            encoder_hidden = model.encode(doc_ids, doc_mask)
            pipeline_compressed = model.compress_ntp(encoder_hidden, doc_mask)
        else:
            encoder_hidden = model.encode(batch["context_ids"], batch["context_mask"])
            pipeline_compressed = model.compress(
                encoder_hidden, batch["context_mask"],
                batch["prompt_ids"], batch["prompt_mask"],
            )
        pipeline_loss = _loss_from_compressed(pipeline_compressed)
    print(f"  Pipeline (encode+compress) loss:  {pipeline_loss:.4f}")

    # ── learnable-only: optimize random soft prompts directly ────────
    learnable_prefix = torch.nn.Parameter(
        torch.randn(B, k, D, device=device, dtype=next(model.parameters()).dtype) * 0.02
    )
    optimizer = AdamW([learnable_prefix], lr=lr, weight_decay=0.01)

    print(f"\n  Training learnable soft prompts ({steps} steps)...")
    print(f"  {'step':>5}  {'loss':>10}")
    print(f"  {'─' * 5}  {'─' * 10}")

    learn_losses = []
    for step_i in range(1, steps + 1):
        optimizer.zero_grad()

        if stage == 1:
            gc_was_on = model._pause_gradient_checkpointing()
            model._disable_adapter_forward_only()

            embed_layer = model.get_embedding_layer()
            seg_embeds = embed_layer(segment_ids)
            sep = model.separator.unsqueeze(0).expand(B, -1, -1)
            inputs_embeds = torch.cat([learnable_prefix, sep, seg_embeds], dim=1)

            k_len = learnable_prefix.shape[1]
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
                learnable_prefix,
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
        loss.backward()
        learn_losses.append(loss.item())
        if step_i == 1 or step_i % log_every == 0:
            print(f"  {step_i:>5}  {loss.item():>10.4f}")
        optimizer.step()

    # ── table ─────────────────────────────────────────────────────────
    print(f"\n  {'─' * 60}")
    print(f"  {'Method':<35}  {'Loss':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'Random soft prompts':<35}  {random_loss:>12.4f}")
    print(f"  {'Learnable soft prompts (optimized)':<35}  {learn_losses[-1]:>12.4f}")
    print(f"  {'Pipeline (encode + compress)':<35}  {pipeline_loss:>12.4f}")
    print(f"  {'─' * 60}")

    print(f"\n  DIAGNOSIS:")
    gap = pipeline_loss - learn_losses[-1]
    if pipeline_loss < random_loss:
        print(f"    Pipeline ({pipeline_loss:.4f}) < Random ({random_loss:.4f})")
        print("    => Perceiver extracts meaningful information from context")
        if pipeline_loss < learn_losses[-1]:
            print("    => Pipeline outperforms even optimized learnable prompts — PASS")
        else:
            print(f"    => But optimized learnable prompts are better by {-gap:.4f}")
    else:
        print(f"    Pipeline ({pipeline_loss:.4f}) >= Random ({random_loss:.4f})")
        print("    => Pipeline not extracting useful info — investigate architecture")

    return {
        "random_loss": random_loss,
        "pipeline_loss": pipeline_loss,
        "learnable_losses": learn_losses,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Task Compressor pre-training diagnostics (Exp 1-3)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Training steps for overfit (Exp 1)")
    parser.add_argument("--probe_steps", type=int, default=100,
                        help="Training steps for bottleneck probe (Exp 3)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3",
        help="Comma-separated list of experiments to run (1-3)",
    )
    parser.set_defaults(wandb_project="tc-diagnostic-pre")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]
    device = detect_device()
    torch_dtype = detect_dtype(device, args.no_bf16)

    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=f"pre-training-diag-stage{args.stage}",
        config={"stage": args.stage, "steps": args.steps,
                "probe_steps": args.probe_steps,
                "lr": args.lr, "experiments": experiments},
        entity=args.wandb_entity,
    )

    mode_str = "NTP (Stage 1)" if args.stage == 1 else "QA (Stage 2)"
    print(f"Mode: {mode_str}")
    print(f"Device: {device}  dtype: {torch_dtype}")
    config = Config.from_yaml(args.config)

    print("\nLoading model...")
    model = load_model(config, device=device, torch_dtype=torch_dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    # Auto-detect data path
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
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:<25} {str(tuple(v.shape)):>20}")

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "1" in experiments:
        torch.manual_seed(42)
        exp1 = run_overfit(
            model, batch, device, args.steps, args.lr, args.log_every,
            stage=args.stage,
        )
        all_results["overfit"] = exp1
        if wandb_run:
            for i, loss in enumerate(exp1["losses"]):
                log_wandb(wandb_run, {"exp1/loss": loss}, step=i + 1)

    if "2" in experiments:
        torch.manual_seed(42)
        exp2 = run_gradient_flow(model, batch, device, stage=args.stage)
        all_results["gradient_flow"] = exp2
        if wandb_run:
            for mod_name, layer_norms in exp2.items():
                for layer_name, gn in layer_norms.items():
                    log_wandb(wandb_run, {f"exp2/{mod_name}/{layer_name}": gn})

    if "3" in experiments:
        torch.manual_seed(42)
        exp3 = run_bottleneck(
            model, batch, device, args.probe_steps, args.lr, args.log_every,
            stage=args.stage,
        )
        all_results["bottleneck"] = exp3
        if wandb_run:
            log_wandb(
                wandb_run,
                {
                    "exp3/random_loss": exp3["random_loss"],
                    "exp3/pipeline_loss": exp3["pipeline_loss"],
                    "exp3/learnable_final": exp3["learnable_losses"][-1],
                },
            )

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

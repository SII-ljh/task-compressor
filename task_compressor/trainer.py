"""Trainer for Task Compressor.

Single-stage QA training with differential learning rates:
  - Perceiver / PromptEncoder / separator: perceiver_lr (5e-5)
  - LoRA adapters: lora_lr (1e-5)

Supports DDP and optional DeepSpeed ZeRO-2.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .config import Config

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for Task Compressor."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        train_loader: DataLoader,
        dev_loader: DataLoader | None = None,
        tokenizer=None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.tokenizer = tokenizer
        self.global_step = 0

        # Running loss accumulator for smoothed logging
        self._running_loss = 0.0
        self._running_count = 0
        self._last_grad_norm = 0.0

        # Best eval loss for early-stopping / best checkpoint
        self.best_eval_loss = float("inf")

        # Distributed setup
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.local_rank == 0

        if self.world_size > 1:
            self.device = torch.device(f"cuda:{self.local_rank}")
            model = model.to(self.device)
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=True,
            )
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(self.device)
            self.model = model

        self.raw_model = model  # unwrapped model reference

        # Optimizer & scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.use_amp = config.training.bf16 and self.device.type == "cuda"

        # wandb (offline mode — cluster has no internet)
        self._wandb = None
        if self.is_main:
            try:
                import wandb

                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_dict(),
                    mode="offline",
                )
                self._wandb = wandb
            except Exception:
                logger.warning("wandb init failed, logging disabled")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.config.training
        lora_params = []
        new_params = []

        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name.lower():
                lora_params.append(param)
            else:
                new_params.append(param)

        groups = [
            {"params": new_params, "lr": tc.perceiver_lr},
            {"params": lora_params, "lr": tc.lora_lr},
        ]
        return torch.optim.AdamW(
            groups, weight_decay=tc.weight_decay, eps=1e-5
        )

    def _build_scheduler(self):
        tc = self.config.training
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, total_iters=tc.warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, tc.total_steps - tc.warmup_steps)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[tc.warmup_steps]
        )

    def train(self):
        tc = self.config.training
        self.model.train()
        self.optimizer.zero_grad()

        epoch = 0
        while self.global_step < tc.total_steps:
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            for batch in self.train_loader:
                if self.global_step >= tc.total_steps:
                    break
                metrics = self._train_step(batch)

                if (self.global_step + 1) % tc.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), tc.max_grad_norm
                    )
                    self._last_grad_norm = grad_norm.item()
                    metrics["grad_norm"] = self._last_grad_norm

                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                    else:
                        logger.warning(
                            f"[Step {self.global_step}] Non-finite grad_norm "
                            f"({grad_norm:.4f}), skipping optimizer step"
                        )
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.is_main and self.global_step % tc.logging_steps == 0:
                    self._log(metrics)

                # Evaluation
                if (
                    self.dev_loader is not None
                    and self.global_step % tc.eval_steps == 0
                ):
                    self._evaluate()
                    self.model.train()

                # Save checkpoint
                if self.is_main and self.global_step % tc.save_steps == 0:
                    self._save_checkpoint()

            epoch += 1

        # Final save
        if self.is_main:
            self._save_checkpoint(tag="final")

    def _train_step(self, batch: dict) -> dict:
        tc = self.config.training
        batch = {k: v.to(self.device) for k, v in batch.items()}

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=self.use_amp,
        ):
            outputs = self.model(
                context_ids=batch["context_ids"],
                context_mask=batch["context_mask"],
                prompt_ids=batch["prompt_ids"],
                prompt_mask=batch["prompt_mask"],
                response_ids=batch["response_ids"],
                response_mask=batch["response_mask"],
            )

        loss = outputs["loss"].float() / tc.gradient_accumulation_steps
        if not torch.isfinite(loss):
            logger.error(
                f"[Step {self.global_step}] *** NaN DETECTED *** loss={loss.item():.4f}"
            )
            # Save diagnostic information for debugging
            self._save_nan_diagnostics(batch, outputs)
            self.optimizer.zero_grad()
            return {"loss": float("nan")}
        loss.backward()

        # Accumulate running loss for smoothed logging
        cur_loss = outputs["qa_loss"].item()
        self._running_loss += cur_loss
        self._running_count += 1

        metrics = {
            "loss": cur_loss,
            "qa_loss": cur_loss,
            "lr_perceiver": self.optimizer.param_groups[0]["lr"],
            "lr_lora": self.optimizer.param_groups[1]["lr"],
        }
        return metrics

    @torch.no_grad()
    def _evaluate(self):
        if self.dev_loader is None:
            return
        tc = self.config.training
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        n_samples = 0
        max_eval = tc.max_eval_samples
        sample_batch = None

        for batch in self.dev_loader:
            batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
            bs = batch["context_ids"].shape[0]

            # Stop if we've evaluated enough samples
            if max_eval > 0 and n_samples >= max_eval:
                break

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(
                    context_ids=batch_on_device["context_ids"],
                    context_mask=batch_on_device["context_mask"],
                    prompt_ids=batch_on_device["prompt_ids"],
                    prompt_mask=batch_on_device["prompt_mask"],
                    response_ids=batch_on_device["response_ids"],
                    response_mask=batch_on_device["response_mask"],
                )
            total_loss += outputs["qa_loss"].float().item() * bs
            n_batches += 1
            n_samples += bs

            # Save first batch for sample predictions
            if sample_batch is None:
                sample_batch = batch_on_device

        avg_loss = total_loss / max(n_samples, 1)
        ppl = math.exp(min(avg_loss, 20.0))

        # Track best eval loss and save best checkpoint
        improved = ""
        if avg_loss < self.best_eval_loss:
            self.best_eval_loss = avg_loss
            improved = "  ** best **"
            if self.is_main:
                self._save_checkpoint(tag="best")

        if self.is_main:
            logger.info(
                f"[Eval] step {self.global_step}/{tc.total_steps}  "
                f"eval_loss={avg_loss:.4f}  eval_ppl={ppl:.2f}  "
                f"best={self.best_eval_loss:.4f}  "
                f"samples={n_samples}{improved}"
            )
            metrics = {
                "eval/loss": avg_loss,
                "eval/ppl": ppl,
                "eval/best_loss": self.best_eval_loss,
                "eval/samples": n_samples,
                "step": self.global_step,
            }
            if self._wandb:
                self._wandb.log(metrics, step=self.global_step)

            # Sample predictions
            if sample_batch is not None and tc.num_sample_predictions > 0:
                self._log_sample_predictions(sample_batch, tc.num_sample_predictions)

    def _log_sample_predictions(self, batch: dict, num_samples: int):
        """Generate and log sample predictions from a batch."""
        raw = self.raw_model
        raw.eval()

        bs = batch["context_ids"].shape[0]
        num_samples = min(num_samples, bs)

        # Slice batch to num_samples
        sample = {k: v[:num_samples] for k, v in batch.items()}

        try:
            # Encode → compress → generate
            encoder_hidden = raw.encode(sample["context_ids"], sample["context_mask"])
            compressed = raw.compress(
                encoder_hidden, sample["context_mask"],
                sample["prompt_ids"], sample["prompt_mask"],
            )
            eos_id = self.tokenizer.eos_token_id if self.tokenizer else None
            generated_ids = raw.generate(
                compressed,
                sample["prompt_ids"], sample["prompt_mask"],
                max_new_tokens=128,
                temperature=0.0,  # greedy for deterministic samples
                eos_token_id=eos_id,
            )

            logger.info("── Sample Predictions ──")
            for i in range(num_samples):
                # Decode prompt
                prompt_ids_i = sample["prompt_ids"][i]
                prompt_mask_i = sample["prompt_mask"][i]
                prompt_len = prompt_mask_i.sum().item()
                prompt_text = self._decode(prompt_ids_i[:int(prompt_len)])

                # Decode ground truth response
                resp_ids_i = sample["response_ids"][i]
                resp_mask_i = sample["response_mask"][i]
                resp_len = resp_mask_i.sum().item()
                ref_text = self._decode(resp_ids_i[:int(resp_len)])

                # Decode prediction
                pred_text = self._decode(generated_ids[i])

                logger.info(f"  [{i+1}] Prompt:     {prompt_text[:200]}")
                logger.info(f"      Reference:  {ref_text[:200]}")
                logger.info(f"      Predicted:  {pred_text[:200]}")
            logger.info("────────────────────────")

        except Exception as e:
            logger.warning(f"Sample prediction failed: {e}")

    def _decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to string."""
        if self.tokenizer is None:
            return str(token_ids.tolist())
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _log(self, metrics: dict):
        tc = self.config.training

        # Use running average for smoothed loss
        if self._running_count > 0:
            avg_loss = self._running_loss / self._running_count
        else:
            avg_loss = metrics.get("loss", 0.0)

        ppl = math.exp(min(avg_loss, 20.0))
        lr = metrics.get("lr_perceiver", 0.0)
        grad_norm = metrics.get("grad_norm", self._last_grad_norm)

        logger.info(
            f"step {self.global_step}/{tc.total_steps}  "
            f"loss={avg_loss:.4f}  ppl={ppl:.2f}  "
            f"lr={lr:.2e}  grad_norm={grad_norm:.4f}"
        )

        wandb_metrics = {
            "train/loss": avg_loss,
            "train/ppl": ppl,
            "train/lr_perceiver": metrics.get("lr_perceiver", 0.0),
            "train/lr_lora": metrics.get("lr_lora", 0.0),
            "train/grad_norm": grad_norm,
            "step": self.global_step,
        }
        if self._wandb:
            self._wandb.log(wandb_metrics, step=self.global_step)

        # Reset running accumulator
        self._running_loss = 0.0
        self._running_count = 0

    def _save_nan_diagnostics(self, batch: dict, outputs: dict):
        """Save detailed diagnostics when NaN is detected."""
        from pathlib import Path

        debug_dir = Path(self.config.output_dir) / "nan_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        dump_path = debug_dir / f"nan_step_{self.global_step}.pt"

        # Collect diagnostic info
        diagnostics = {
            "global_step": self.global_step,
            "lr_perceiver": self.optimizer.param_groups[0]["lr"],
            "lr_lora": self.optimizer.param_groups[1]["lr"],
            "batch": {k: v.cpu() for k, v in batch.items()},
            "outputs": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()},
        }

        # Save optimizer state (exp_avg, exp_avg_sq)
        try:
            diagnostics["optimizer_state"] = {
                k: {
                    "exp_avg_mean": v["exp_avg"].float().mean().item() if "exp_avg" in v else None,
                    "exp_avg_max": v["exp_avg"].float().abs().max().item() if "exp_avg" in v else None,
                    "exp_avg_sq_mean": v["exp_avg_sq"].float().mean().item() if "exp_avg_sq" in v else None,
                    "exp_avg_sq_min": v["exp_avg_sq"].float().min().item() if "exp_avg_sq" in v else None,
                }
                for k, v in list(self.optimizer.state.items())[:10]  # Sample first 10 params
            }
        except Exception as e:
            logger.warning(f"Failed to extract optimizer state: {e}")

        # Save parameter statistics
        param_stats = {}
        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue
            try:
                param_stats[name] = {
                    "mean": param.float().mean().item(),
                    "max": param.float().abs().max().item(),
                    "has_nan": torch.isnan(param).any().item(),
                    "has_inf": torch.isinf(param).any().item(),
                }
                if param.grad is not None:
                    param_stats[name]["grad_mean"] = param.grad.float().mean().item()
                    param_stats[name]["grad_max"] = param.grad.float().abs().max().item()
                    param_stats[name]["grad_has_nan"] = torch.isnan(param.grad).any().item()
            except Exception:
                pass

        diagnostics["param_stats"] = param_stats

        # Save to disk
        torch.save(diagnostics, dump_path)
        logger.error(f"NaN diagnostics saved to {dump_path}")
        logger.error(f"  Current LRs: perceiver={diagnostics['lr_perceiver']:.2e}, lora={diagnostics['lr_lora']:.2e}")
        logger.error(f"  Batch keys: {list(batch.keys())}")

        # Optionally save full optimizer state (large file)
        full_optim_path = debug_dir / f"optimizer_state_step_{self.global_step}.pt"
        try:
            torch.save(self.optimizer.state_dict(), full_optim_path)
            logger.error(f"Full optimizer state saved to {full_optim_path}")
        except Exception as e:
            logger.warning(f"Failed to save full optimizer state: {e}")

    def _save_checkpoint(self, tag: str | None = None):
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = tag or f"step_{self.global_step}"
        ckpt_dir = out_dir / name

        # Save LoRA adapter
        self.raw_model.base_model.save_pretrained(str(ckpt_dir / "lora_adapter"))

        # Save perceiver, prompt encoder, and other new parameters
        new_state = {}
        for k, v in self.raw_model.state_dict().items():
            if any(
                x in k
                for x in ["perceiver", "prompt_encoder", "context_tokens", "separator"]
            ):
                new_state[k] = v
        torch.save(new_state, ckpt_dir / "task_compressor_modules.pt")

        # Save optimizer & scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
            },
            ckpt_dir / "training_state.pt",
        )
        logger.info(f"Checkpoint saved to {ckpt_dir}")

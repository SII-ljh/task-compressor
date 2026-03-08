"""Trainer for Task Compressor.

Two-stage training with differential learning rates:
  - Perceiver / PromptEncoder / separator: perceiver_lr (5e-5)
  - LoRA adapters: lora_lr (1e-5)

Stage 1 (NTP):  compress doc → soft prompts → predict continuation (CE only)
Stage 2 (QA):   compress doc → soft prompts → answer question (CE + distillation)

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
from torch.utils.data import DataLoader, DistributedSampler

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
        mode: str = "qa",
    ):
        self.config = config
        self.mode = mode  # "qa" or "ntp"
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.global_step = 0

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

        # wandb
        self._wandb = None
        if self.is_main:
            try:
                import wandb

                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_dict(),
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
            groups, weight_decay=tc.weight_decay
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
                    torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(), tc.max_grad_norm
                    )
                    self.optimizer.step()
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
            if self.mode == "ntp":
                outputs = self.model(
                    mode="ntp",
                    doc_ids=batch["doc_ids"],
                    doc_mask=batch["doc_mask"],
                    segment_ids=batch["segment_ids"],
                    segment_mask=batch["segment_mask"],
                    segment_labels=batch["segment_labels"],
                )
            else:
                outputs = self.model(
                    mode="qa",
                    context_ids=batch["context_ids"],
                    context_mask=batch["context_mask"],
                    prompt_ids=batch["prompt_ids"],
                    prompt_mask=batch["prompt_mask"],
                    response_ids=batch["response_ids"],
                    response_mask=batch["response_mask"],
                    teacher_input_ids=batch["teacher_input_ids"],
                    teacher_attention_mask=batch["teacher_attention_mask"],
                    response_starts=batch["response_starts"],
                    response_lens=batch["response_lens"],
                    distill_alpha=tc.distill_alpha,
                    distill_temperature=tc.distill_temperature,
                    distill_method=tc.distill_method,
                )

        loss = outputs["loss"] / tc.gradient_accumulation_steps
        loss.backward()

        metrics = {
            "loss": outputs["loss"].item(),
            "lr_perceiver": self.optimizer.param_groups[0]["lr"],
            "lr_lora": self.optimizer.param_groups[1]["lr"],
        }
        if self.mode == "ntp":
            metrics["ntp_loss"] = outputs["ntp_loss"].item()
        else:
            metrics["qa_loss"] = outputs["qa_loss"].item()
            metrics["distill_loss"] = outputs["distill_loss"].item()
        return metrics

    @torch.no_grad()
    def _evaluate(self):
        if self.dev_loader is None:
            return
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in self.dev_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                if self.mode == "ntp":
                    outputs = self.model(
                        mode="ntp",
                        doc_ids=batch["doc_ids"],
                        doc_mask=batch["doc_mask"],
                        segment_ids=batch["segment_ids"],
                        segment_mask=batch["segment_mask"],
                        segment_labels=batch["segment_labels"],
                    )
                else:
                    outputs = self.model(
                        mode="qa",
                        context_ids=batch["context_ids"],
                        context_mask=batch["context_mask"],
                        prompt_ids=batch["prompt_ids"],
                        prompt_mask=batch["prompt_mask"],
                        response_ids=batch["response_ids"],
                        response_mask=batch["response_mask"],
                        distill_alpha=0.0,
                    )
            total_loss += outputs["loss"].item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        metrics: dict = {
            "eval/loss": avg_loss,
            "step": self.global_step,
        }
        if self.mode == "ntp":
            metrics["eval/perplexity"] = math.exp(min(avg_loss, 20))
        if self.is_main:
            logger.info(f"[Eval step={self.global_step}] {metrics}")
            if self._wandb:
                self._wandb.log(metrics, step=self.global_step)

    def _log(self, metrics: dict):
        metrics["step"] = self.global_step
        logger.info(f"[Step {self.global_step}] {metrics}")
        if self._wandb:
            self._wandb.log(metrics, step=self.global_step)

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

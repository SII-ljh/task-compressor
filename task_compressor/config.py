"""Configuration dataclasses for Task Compressor."""

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class ModelConfig:
    base_model: str = "models/Qwen3-0.6B"
    lora_rank: int = 128
    lora_alpha: int = 128
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_dropout: float = 0.0
    n_prompt_tokens: int = 16
    n_context_tokens: int = 48
    num_perceiver_layers: int = 2
    perceiver_ffn_mult: int = 4
    gradient_checkpointing: bool = True

    @property
    def num_soft_prompt_tokens(self) -> int:
        return self.n_prompt_tokens + self.n_context_tokens


@dataclass
class TrainingConfig:
    per_gpu_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 500
    total_steps: int = 50000
    lora_lr: float = 1e-5
    perceiver_lr: float = 5e-5
    max_grad_norm: float = 2.0
    weight_decay: float = 0.01
    bf16: bool = True
    deepspeed: Optional[str] = None
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10
    max_eval_samples: int = 5000
    num_sample_predictions: int = 3
    resume_from: Optional[str] = None  # checkpoint path to resume training


@dataclass
class DataConfig:
    max_context_length: int = 4096
    max_prompt_length: int = 256
    max_response_length: int = 512
    train_file: str = "data/qa_train.json"
    dev_file: str = "data/qa_dev.json"
    num_workers: int = 4


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs"
    wandb_project: str = "task-compressor"
    wandb_run_name: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = cls()

        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(config.data, k):
                    setattr(config.data, k, v)
        for k in ["output_dir", "wandb_project", "wandb_run_name", "seed"]:
            if k in raw:
                setattr(config, k, raw[k])

        # Resolve relative paths to absolute so from_pretrained() finds local dirs
        config._resolve_paths()
        return config

    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute if they exist on disk."""
        for attr in ("base_model",):
            val = getattr(self.model, attr)
            if val and not os.path.isabs(val) and os.path.isdir(val):
                setattr(self.model, attr, os.path.abspath(val))
        for attr in ("train_file", "dev_file"):
            val = getattr(self.data, attr)
            if val and not os.path.isabs(val) and os.path.exists(val):
                setattr(self.data, attr, os.path.abspath(val))

    @staticmethod
    def _cast(current_value, new_value: str):
        """Cast a CLI string to the type of the current field value."""
        if current_value is None:
            # Optional field currently None — treat "null"/"none" as None,
            # otherwise keep as string.
            if new_value.lower() in ("null", "none"):
                return None
            return new_value
        field_type = type(current_value)
        return field_type(new_value)

    def merge_overrides(self, overrides: dict) -> None:
        """Merge a flat dict of overrides (e.g. from CLI) into this config.

        Keys use dot notation: ``model.lora_rank=64``.
        """
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) == 2:
                group, attr = parts
                sub = getattr(self, group, None)
                if sub is not None and hasattr(sub, attr):
                    setattr(sub, attr, self._cast(getattr(sub, attr), value))
            elif len(parts) == 1 and hasattr(self, parts[0]):
                setattr(self, parts[0], self._cast(getattr(self, parts[0]), value))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

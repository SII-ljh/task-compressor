# CLAUDE.md

## Project Overview

**Task Compressor** — compresses ultra-long texts into fixed-length latent representations for downstream tasks.

Architecture: LoRA Encoder encodes context → Query-Conditioned Perceiver compresses to k=64 soft prompts guided by user prompt → Frozen Decoder generates response. Encoder and Decoder share one Qwen model (LoRA toggled on/off).

## Environment Setup

```bash
conda create --name qwen3 python=3.11
conda activate qwen3
pip install -r requirements.txt
```

## Model Download

```bash
# List available Qwen3 models
python scripts/download_models.py --list

# Download default model (Qwen3-0.6B)
python scripts/download_models.py

# Download specific models
python scripts/download_models.py --models 0.6B,1.7B,4B,8B

# Download all models
python scripts/download_models.py --models all
```

Models are saved to `models/Qwen3-{size}/`.

## Data Preparation

```bash
# Full download (~1-2 hours)
python scripts/prepare_data.py

# Small test subset
python scripts/prepare_data.py --test

# Extract tiny subset (~50 samples) from existing data
python scripts/prepare_data.py --make-tiny

# Extract dev subset (~500-2000 samples)
python scripts/prepare_data.py --make-dev

# Extract ablation subset (5% of total)
python scripts/prepare_data.py --make-ablation

# All subsets at once
python scripts/prepare_data.py --make-all-subsets
```

Data is saved to `data/`.

## Training

```bash
# Single GPU
python scripts/train.py --config configs/default.yaml

# Multi-GPU (DDP)
torchrun --nproc_per_node=8 scripts/train.py --config configs/default.yaml

# Override config values
python scripts/train.py --config configs/default.yaml \
    model.lora_rank=64 training.total_steps=20000

# Ablation: different soft prompt count
python scripts/train.py --config configs/default.yaml \
    model.n_prompt_tokens=0 model.n_context_tokens=16
```

Checkpoints saved to `outputs/`.

## Evaluation

```bash
# QA evaluation (BLEU/ROUGE/EM)
python scripts/evaluate.py --checkpoint outputs/final --data data/qa_dev.json

# Benchmark latency
python scripts/evaluate.py --checkpoint outputs/final --benchmark
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_perceiver.py -v
pytest tests/test_losses.py -v
```

## Project Structure

```
task_compressor/
├── requirements.txt              # Dependencies
├── CLAUDE.md                     # This file
├── configs/
│   └── default.yaml              # Default hyperparameters
├── scripts/
│   ├── download_models.py        # Qwen3 model downloader
│   ├── prepare_data.py           # NTP + QA dataset preparation
│   ├── train.py                  # Training entry point
│   └── evaluate.py               # Evaluation & benchmarking
├── task_compressor/              # Python package
│   ├── __init__.py
│   ├── config.py                 # Dataclass configs with YAML loading
│   ├── losses.py                 # L_QA (CE) + L_distill (KL/MSE)
│   ├── data.py                   # QADataset + QACollator
│   ├── trainer.py                # Training loop (DDP, wandb, checkpointing)
│   ├── inference.py              # EncoderCacheManager + Pipeline
│   └── models/
│       ├── __init__.py
│       ├── perceiver.py          # QueryConditionedPerceiver
│       ├── prompt_encoder.py     # PromptEncoder (cross-attn over prompt)
│       └── task_compressor_model.py  # Full model assembly
├── tests/                        # Unit tests
│   ├── conftest.py
│   ├── test_perceiver.py
│   ├── test_prompt_encoder.py
│   ├── test_losses.py
│   ├── test_task_compressor.py
│   └── test_data.py
├── models/                       # Downloaded models
└── data/                         # Prepared datasets
```

## Architecture Summary

| Component | Description | Parameters |
|-----------|-------------|------------|
| LoRA Encoder | Qwen + LoRA (rank=128, q/v_proj) | ~1% of base model |
| Prompt Encoder | n_p=16 learnable latents × cross-attn | ~2M |
| Perceiver | 2 layers × (cross-attn + self-attn + FFN) | ~50M |
| Separator | 1 learnable token | hidden_size |
| Context Tokens | n_c=48 learnable query tokens | 48 × hidden_size |
| Decoder | Frozen Qwen (LoRA off) | 0 (shared weights) |

## Key Design Decisions

1. **Single Qwen instance**: encoder/decoder share weights, LoRA toggled via peft enable/disable
2. **Single-stage training**: LoRA init delta≈0, Perceiver starts with high-quality Qwen hidden states
3. **Differential LR**: Perceiver/PromptEncoder 5e-5, LoRA 1e-5
4. **Distillation**: KL(teacher‖student) with T=2.0, teacher = frozen Qwen on full context

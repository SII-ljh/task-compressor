# CLAUDE.md

## Project Overview

**Task Compressor** — compresses ultra-long texts into fixed-length latent representations for downstream tasks. This repository currently contains data preparation and model download utilities. Model architecture is under development.

## Environment Setup

```bash
conda create --name task_compressor python=3.11
conda activate task_compressor
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

## Project Structure

```
task_compressor/
├── requirements.txt          # Dependencies
├── CLAUDE.md                 # This file
├── scripts/
│   ├── download_models.py    # Qwen3 model downloader
│   └── prepare_data.py       # NTP + QA dataset preparation
├── models/                   # Downloaded models
├── data/                     # Prepared datasets
└── task_compressor/          # Python package (skeleton)
    └── __init__.py
```

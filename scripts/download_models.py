#!/usr/bin/env python3
"""Download Qwen3 instruction-tuned models.

Supports Qwen3 dense models from 0.6B to 8B.

Available models:
  Qwen3-0.6B   hidden=1024, layers=28   (~1.2 GB bf16)
  Qwen3-1.7B   hidden=2048, layers=28   (~3.4 GB bf16)
  Qwen3-4B     hidden=2560, layers=36   (~8 GB bf16)
  Qwen3-8B     hidden=4096, layers=36   (~16 GB bf16)

Usage:
  python scripts/download_models.py                           # download default (0.6B)
  python scripts/download_models.py --models 0.6B,1.7B,4B,8B # download specific models
  python scripts/download_models.py --models all              # download all models
  python scripts/download_models.py --list                    # list available models
  python scripts/download_models.py --dtype float32           # force float32
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_BASE_DIR = PROJECT_ROOT / "models"

# Qwen3 instruction-tuned model registry
QWEN3_MODELS = {
    "0.6B": {
        "repo": "Qwen/Qwen3-0.6B",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
    },
    "1.7B": {
        "repo": "Qwen/Qwen3-1.7B",
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
    },
    "4B": {
        "repo": "Qwen/Qwen3-4B",
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
    },
    "8B": {
        "repo": "Qwen/Qwen3-8B",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
    },
}


def list_models():
    """Print available Qwen3 models."""
    print("Available Qwen3 instruction-tuned models:")
    print("-" * 65)
    for size, info in QWEN3_MODELS.items():
        print(f"  {size:6s}  {info['repo']:20s}  "
              f"hidden={info['hidden_size']}, layers={info['num_hidden_layers']}")
    print()


def download_model(size: str, torch_dtype: str = "auto"):
    """Download a single Qwen3 model and tokenizer to local directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if size not in QWEN3_MODELS:
        raise ValueError(
            f"Unknown model size: {size}. "
            f"Available: {', '.join(QWEN3_MODELS.keys())}"
        )

    info = QWEN3_MODELS[size]
    repo = info["repo"]
    model_dir = MODEL_BASE_DIR / f"Qwen3-{size}"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[Model] Downloading {repo}")
    print(f"  hidden_size={info['hidden_size']}, "
          f"layers={info['num_hidden_layers']}, "
          f"vocab={info['vocab_size']}")

    print(f"[Model] Downloading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(repo)
    tokenizer.save_pretrained(str(model_dir))
    print(f"  -> tokenizer saved to {model_dir}")

    dtype_arg = torch_dtype if torch_dtype != "auto" else torch_dtype
    print(f"[Model] Downloading model weights (dtype={torch_dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype_arg)
    model.save_pretrained(str(model_dir))

    total_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    print(f"  -> model saved to {model_dir}  ({total_bytes / 1e9:.2f} GB)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3 instruction-tuned models"
    )
    parser.add_argument(
        "--models", type=str, default="0.6B",
        help="Comma-separated model sizes (e.g., '0.6B,4B,8B') or 'all' (default: 0.6B)",
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype for saving (default: auto, usually bfloat16)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.models.lower() == "all":
        sizes = list(QWEN3_MODELS.keys())
    else:
        sizes = [s.strip() for s in args.models.split(",")]
        for s in sizes:
            if s not in QWEN3_MODELS:
                print(f"ERROR: Unknown model size '{s}'")
                print(f"Available: {', '.join(QWEN3_MODELS.keys())}")
                return

    print("=" * 60)
    print("Task Compressor - Model Download")
    print(f"Models: {', '.join(sizes)}")
    print(f"Target: {MODEL_BASE_DIR}")
    print(f"Dtype:  {args.dtype}")
    print("=" * 60)
    print()

    for size in sizes:
        download_model(size, torch_dtype=args.dtype)

    print("=" * 60)
    print("Download complete!")
    print("-" * 60)
    for size in sizes:
        model_dir = MODEL_BASE_DIR / f"Qwen3-{size}"
        if model_dir.exists():
            total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            print(f"  Qwen3-{size:6s}  {total / 1e9:.2f} GB  ({model_dir})")
    print()


if __name__ == "__main__":
    main()

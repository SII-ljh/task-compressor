#!/usr/bin/env python3
"""Evaluate Task Compressor on QA datasets.

Metrics:
  - BLEU (sentence-level)
  - ROUGE-L (F1)
  - Exact Match

Usage:
  python scripts/evaluate.py \\
      --checkpoint outputs/final \\
      --data data/qa_dev.json \\
      --max-samples 500

  # Benchmark latency
  python scripts/evaluate.py \\
      --checkpoint outputs/final \\
      --benchmark
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from task_compressor.config import Config
from task_compressor.inference import TaskCompressorPipeline
from task_compressor.models.task_compressor_model import TaskCompressorModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_dir: str,
    config: Config,
    device: str = "cuda",
) -> tuple[TaskCompressorModel, AutoTokenizer]:
    """Load model from checkpoint directory."""
    ckpt_dir = Path(checkpoint_dir)
    torch_dtype = torch.bfloat16 if config.training.bf16 else torch.float32

    # Build model
    model = TaskCompressorModel(config.model, torch_dtype=torch_dtype)

    # Load LoRA adapter
    lora_path = ckpt_dir / "lora_adapter"
    if lora_path.exists():
        from peft import PeftModel

        model.base_model = PeftModel.from_pretrained(
            model.base_model.base_model,  # unwrap existing peft
            str(lora_path),
            torch_dtype=torch_dtype,
        )

    # Load task compressor modules
    modules_path = ckpt_dir / "task_compressor_modules.pt"
    if modules_path.exists():
        state = torch.load(modules_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded modules: {len(state)} keys, "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model.to(device)
    return model, tokenizer


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    """Compute BLEU, ROUGE-L, and Exact Match."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        smooth = SmoothingFunction().method1
    except ImportError:
        sentence_bleu = None

    bleu_scores = []
    rouge_scores = []
    exact_matches = 0

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip()
        ref_clean = ref.strip()

        # Exact match
        if pred_clean == ref_clean:
            exact_matches += 1

        # ROUGE-L
        rouge_result = scorer.score(ref_clean, pred_clean)
        rouge_scores.append(rouge_result["rougeL"].fmeasure)

        # BLEU
        if sentence_bleu is not None:
            ref_tokens = ref_clean.split()
            pred_tokens = pred_clean.split()
            if ref_tokens and pred_tokens:
                bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
                bleu_scores.append(bleu)

    n = len(predictions)
    metrics = {
        "n_samples": n,
        "exact_match": exact_matches / max(n, 1),
        "rouge_l": sum(rouge_scores) / max(len(rouge_scores), 1),
    }
    if bleu_scores:
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Task Compressor")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = Config.from_yaml(args.config)
    data_path = args.data or config.data.dev_file

    logger.info(f"Loading model from {args.checkpoint}")
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, config, args.device)

    pipeline = TaskCompressorPipeline(
        model, tokenizer, device=args.device, cache_dir=args.cache_dir
    )

    # ── Benchmark mode ───────────────────────────────────────────────────
    if args.benchmark:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        sample = data[0]
        results = pipeline.benchmark(
            context=sample["context"],
            prompt=sample["question"],
            n_runs=20,
            max_new_tokens=64,
        )
        logger.info("Benchmark results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.2f}")
        return

    # ── Evaluation mode ──────────────────────────────────────────────────
    logger.info(f"Loading data from {data_path}")
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples > 0:
        data = data[: args.max_samples]

    predictions = []
    references = []
    results_detail = []

    for item in tqdm(data, desc="Evaluating"):
        pred = pipeline.generate(
            context=item["context"],
            prompt=item["question"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        ref = item["answer"]
        predictions.append(pred)
        references.append(ref)
        results_detail.append({
            "question": item["question"],
            "reference": ref,
            "prediction": pred,
        })

    metrics = compute_metrics(predictions, references)
    logger.info("Evaluation metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": metrics, "predictions": results_detail},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detailed QA evaluation across all Task Compressor experiments.

Auto-discovers checkpoint directories under outputs/, loads each model,
evaluates on both held-out dev set and a sample from the training set.

- **stdout**: summary statistics (comparison table, bar charts, ranking)
- **markdown report**: full per-sample analysis saved to results/qa_eval_report.md

Usage:
    # Auto-discover all experiments under outputs/
    python scripts/evaluate_qa_detailed.py --config configs/h200_base.yaml

    # Only under a specific subdirectory
    python scripts/evaluate_qa_detailed.py --config configs/h200_base.yaml \
        --outputs_dir outputs/ablation_compression

    # Specific checkpoints
    python scripts/evaluate_qa_detailed.py --config configs/h200_base.yaml \
        --checkpoints outputs/ablation_compression/k512_np128_nc384/best,outputs/ablation_compression/k64_np16_nc48/best

    # Full dev set
    python scripts/evaluate_qa_detailed.py --config configs/h200_base.yaml \
        --max_eval_samples 0

    # Multi-GPU (via accelerate)
    accelerate launch --multi_gpu --num_processes 4 \
        scripts/evaluate_qa_detailed.py --config configs/h200_base.yaml
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import string
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_compressor.config import Config, ModelConfig
from task_compressor.data import QACollator
from task_compressor.models.task_compressor_model import TaskCompressorModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Terminal colors ──────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

W = 100  # terminal output width


def bar(value: float, max_val: float, width: int = 30) -> str:
    if max_val <= 0:
        return ""
    filled = min(int(round(value / max_val * width)), width)
    return "\u2588" * filled + "\u2591" * (width - filled)


# ── Metric functions ─────────────────────────────────────────────────────────

_PUNCT = set(string.punctuation)
_CJK_PUNCT = set("。，、；：？！""''【】（）《》〈〉——…—～·")


def normalize_text(s: str) -> str:
    """Lower-case, strip articles/punctuation, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT and ch not in _CJK_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_for_f1(text: str) -> List[str]:
    """Character-level for CJK, word-level otherwise."""
    tokens = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
        elif ch == " ":
            tokens.append(" ")
        else:
            tokens.append(ch)
    # Split into word-level tokens, then expand CJK chars
    result = []
    for token in normalize_text(text).split():
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in token)
        if has_cjk:
            result.extend(ch for ch in token if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())
        else:
            result.append(token)
    return result


def compute_exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = _tokenize_for_f1(pred)
    gold_tokens = _tokenize_for_f1(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    from collections import Counter
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())

    if common == 0:
        return 0.0
    precision = common / sum(pred_counts.values())
    recall = common / sum(gold_counts.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest common subsequence length (space-optimized DP)."""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[len(b)]


def compute_rouge_l(pred: str, gold: str) -> float:
    pred_tokens = _tokenize_for_f1(pred)
    gold_tokens = _tokenize_for_f1(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Data containers ──────────────────────────────────────────────────────────


@dataclass
class SampleResult:
    idx: int
    question: str
    gold: str
    prediction: str
    context_preview: str
    em: float
    f1: float
    rouge_l: float
    source: str  # "dev" or "train"


@dataclass
class ModelResult:
    name: str
    k_value: int  # total soft prompt count (n_p + n_c)
    config_info: str
    checkpoint_path: str
    # Dev metrics
    dev_loss: float = 0.0
    dev_ppl: float = 0.0
    dev_em: float = 0.0
    dev_f1: float = 0.0
    dev_rouge_l: float = 0.0
    dev_n: int = 0
    dev_empty_preds: int = 0
    # Train metrics
    train_loss: float = 0.0
    train_ppl: float = 0.0
    train_em: float = 0.0
    train_f1: float = 0.0
    train_rouge_l: float = 0.0
    train_n: int = 0
    train_empty_preds: int = 0
    # Per-sample
    samples: List[SampleResult] = field(default_factory=list)


# ── Eval Dataset (returns raw text alongside tokens) ─────────────────────────


class EvalQADataset(Dataset):
    """QA dataset for evaluation: returns both tokenized tensors and raw text."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_context_length: int = 4096,
        max_prompt_length: int = 256,
        max_response_length: int = 512,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        context_enc = self.tokenizer(
            item["context"],
            truncation=True,
            max_length=self.max_context_length,
            add_special_tokens=False,
        )
        prompt_enc = self.tokenizer(
            item["question"],
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )
        response_enc = self.tokenizer(
            item["answer"],
            truncation=True,
            max_length=self.max_response_length,
            add_special_tokens=False,
        )

        return {
            "context_ids": torch.tensor(context_enc["input_ids"], dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_enc["input_ids"], dtype=torch.long),
            "response_ids": torch.tensor(response_enc["input_ids"], dtype=torch.long),
            # Raw text for metrics
            "question_text": item["question"],
            "answer_text": item["answer"],
            "context_preview": item["context"][:300],
        }


class EvalQACollator:
    """Collator for evaluation: pads tensors and collects raw text lists."""

    def __init__(
        self,
        pad_token_id: int,
        max_context_length: int | None = None,
        max_prompt_length: int | None = None,
        max_response_length: int | None = None,
    ):
        self.qa_collator = QACollator(
            pad_token_id=pad_token_id,
            max_context_length=max_context_length,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
        )

    def __call__(self, batch: list[dict]) -> dict:
        # Separate text and tensor fields
        text_fields = {}
        for key in ("question_text", "answer_text", "context_preview"):
            text_fields[key] = [item[key] for item in batch]

        # Delegate tensor padding to QACollator
        result = self.qa_collator(batch)

        # Add text fields
        result.update(text_fields)
        return result


# ── Checkpoint discovery ─────────────────────────────────────────────────────


def infer_config_from_checkpoint(ckpt_dir: Path) -> Tuple[int, int]:
    """Infer n_prompt_tokens and n_context_tokens from checkpoint weights.

    Returns:
        (n_prompt_tokens, n_context_tokens)
    """
    modules_path = ckpt_dir / "task_compressor_modules.pt"
    state = torch.load(modules_path, map_location="cpu", weights_only=True)

    n_context = state["context_tokens"].shape[0]
    n_prompt = 0
    for key in state:
        if "latent_tokens" in key:
            n_prompt = state[key].shape[0]
            break
    return n_prompt, n_context


def discover_checkpoints(
    base_dir: str = "outputs",
) -> List[Tuple[Path, int, int, int]]:
    """Recursively find checkpoint directories under base_dir.

    Returns list of (checkpoint_dir, n_prompt, n_context, k) sorted by k.
    """
    base = Path(base_dir)
    if not base.exists():
        return []

    found = []
    for modules_file in base.rglob("task_compressor_modules.pt"):
        ckpt_dir = modules_file.parent
        # Must also have lora_adapter
        if not (ckpt_dir / "lora_adapter").is_dir():
            continue

        try:
            n_p, n_c = infer_config_from_checkpoint(ckpt_dir)
        except Exception as e:
            logger.warning(f"Cannot infer config from {ckpt_dir}: {e}")
            continue

        k = n_p + n_c
        found.append((ckpt_dir, n_p, n_c, k))

    # Sort by k value, then by path for stable ordering
    found.sort(key=lambda x: (x[3], str(x[0])))
    return found


# ── Model loading ────────────────────────────────────────────────────────────


def load_model(
    checkpoint_dir: Path,
    base_config: Config,
    n_prompt_tokens: int,
    n_context_tokens: int,
    accelerator: Accelerator,
) -> TaskCompressorModel:
    """Load a Task Compressor model from checkpoint."""
    # Clone the base config and override prompt/context tokens
    import copy
    config = copy.deepcopy(base_config)
    config.model.n_prompt_tokens = n_prompt_tokens
    config.model.n_context_tokens = n_context_tokens

    torch_dtype = torch.bfloat16 if config.training.bf16 else torch.float32
    model = TaskCompressorModel(config.model, torch_dtype=torch_dtype)

    # Load task compressor modules
    modules_path = checkpoint_dir / "task_compressor_modules.pt"
    if modules_path.exists():
        state = torch.load(modules_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"  Loaded task_compressor_modules.pt "
            f"({len(state)} keys, {len(missing)} missing, {len(unexpected)} unexpected)"
        )

    # Load LoRA adapter weights
    lora_path = checkpoint_dir / "lora_adapter"
    if lora_path.exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as load_safetensors

        safetensors_path = lora_path / "adapter_model.safetensors"
        bin_path = lora_path / "adapter_model.bin"
        if safetensors_path.exists():
            lora_state = load_safetensors(str(safetensors_path))
        elif bin_path.exists():
            lora_state = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            logger.warning(f"  No adapter weights found in {lora_path}")
            lora_state = None

        if lora_state is not None:
            set_peft_model_state_dict(model.base_model, lora_state)
            logger.info(f"  Loaded LoRA adapter ({len(lora_state)} keys)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = accelerator.prepare(model)
    return model


# ── Core evaluation ──────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_qa_detailed(
    model,
    eval_loader: DataLoader,
    tokenizer,
    accelerator: Accelerator,
    max_new_tokens: int = 64,
    source_label: str = "dev",
) -> Tuple[Dict[str, float], List[SampleResult]]:
    """Evaluate a Task Compressor model on QA data.

    Returns:
        (metrics_dict, list_of_sample_results)
    """
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    all_em, all_f1, all_rouge, all_loss = [], [], [], []
    samples = []
    empty_count = 0
    global_idx = 0

    total_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        if accelerator.is_main_process and batch_idx % 10 == 0:
            pct = 100 * batch_idx / max(total_batches, 1)
            print(
                f"\r  [{source_label}] batch {batch_idx}/{total_batches} "
                f"({pct:.0f}%)",
                end="",
                flush=True,
            )

        # Extract text fields before moving tensors to device
        question_texts = batch.pop("question_text")
        answer_texts = batch.pop("answer_text")
        context_previews = batch.pop("context_preview")

        # Move tensors to device
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        # ── Teacher-forcing loss ──
        outputs = unwrapped(
            context_ids=batch["context_ids"],
            context_mask=batch["context_mask"],
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            response_ids=batch["response_ids"],
            response_mask=batch["response_mask"],
        )
        batch_loss = outputs["qa_loss"].float().item()

        # ── Generation ──
        encoder_hidden = unwrapped.encode(
            batch["context_ids"], batch["context_mask"]
        )
        compressed = unwrapped.compress(
            encoder_hidden,
            batch["context_mask"],
            batch["prompt_ids"],
            batch["prompt_mask"],
        )
        gen_ids = unwrapped.generate(
            compressed,
            batch["prompt_ids"],
            batch["prompt_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy for deterministic eval
            eos_token_id=tokenizer.eos_token_id,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for i, (pred, gold) in enumerate(zip(preds, answer_texts)):
            if not pred.strip():
                empty_count += 1
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            rl = compute_rouge_l(pred, gold)
            all_em.append(em)
            all_f1.append(f1)
            all_rouge.append(rl)
            all_loss.append(batch_loss)

            q_text = question_texts[i] if i < len(question_texts) else ""
            ctx = context_previews[i] if i < len(context_previews) else ""
            samples.append(
                SampleResult(
                    idx=global_idx,
                    question=q_text,
                    gold=gold,
                    prediction=pred,
                    context_preview=ctx,
                    em=em,
                    f1=f1,
                    rouge_l=rl,
                    source=source_label,
                )
            )
            global_idx += 1

    if accelerator.is_main_process:
        print()  # newline after progress

    n = max(len(all_em), 1)
    avg_loss = sum(all_loss) / max(len(all_loss), 1)
    metrics = {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20.0)),
        "exact_match": sum(all_em) / n,
        "f1": sum(all_f1) / n,
        "rouge_l": sum(all_rouge) / n,
        "n_samples": len(all_em),
        "empty_preds": empty_count,
    }
    return metrics, samples


# ── Baseline (raw Qwen) evaluation ───────────────────────────────────────────


def _strip_and_concat(
    fields_ids: List[torch.Tensor],
    fields_masks: List[torch.Tensor],
    sample_idx: int,
) -> torch.Tensor:
    """For one sample, strip padding from each field and concatenate."""
    parts = []
    for ids, mask in zip(fields_ids, fields_masks):
        real_len = int(mask[sample_idx].sum().item())
        parts.append(ids[sample_idx, :real_len])
    return torch.cat(parts)


def _build_left_padded(
    fields_ids: List[torch.Tensor],
    fields_masks: List[torch.Tensor],
    pad_token_id: int,
):
    """Strip per-field padding, concatenate, then left-pad the batch."""
    B = fields_ids[0].shape[0]
    device = fields_ids[0].device
    dtype = fields_ids[0].dtype

    sequences = [
        _strip_and_concat(fields_ids, fields_masks, i) for i in range(B)
    ]
    max_len = max(s.shape[0] for s in sequences)

    out_ids = torch.full((B, max_len), pad_token_id, dtype=dtype, device=device)
    out_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        slen = seq.shape[0]
        out_ids[i, max_len - slen :] = seq
        out_mask[i, max_len - slen :] = 1
    return out_ids, out_mask


def _build_right_padded(
    fields_ids: List[torch.Tensor],
    fields_masks: List[torch.Tensor],
    fields_labels: List[torch.Tensor],
    pad_token_id: int,
    label_pad: int = -100,
):
    """Strip per-field padding, concatenate, then right-pad (for loss)."""
    B = fields_ids[0].shape[0]
    device = fields_ids[0].device
    dtype = fields_ids[0].dtype

    id_seqs, label_seqs = [], []
    for i in range(B):
        id_parts, label_parts = [], []
        for ids, mask, labels in zip(fields_ids, fields_masks, fields_labels):
            real_len = int(mask[i].sum().item())
            id_parts.append(ids[i, :real_len])
            label_parts.append(labels[i, :real_len])
        id_seqs.append(torch.cat(id_parts))
        label_seqs.append(torch.cat(label_parts))

    max_len = max(s.shape[0] for s in id_seqs)

    out_ids = torch.full((B, max_len), pad_token_id, dtype=dtype, device=device)
    out_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
    out_labels = torch.full((B, max_len), label_pad, dtype=dtype, device=device)
    for i in range(B):
        slen = id_seqs[i].shape[0]
        out_ids[i, :slen] = id_seqs[i]
        out_mask[i, :slen] = 1
        out_labels[i, :slen] = label_seqs[i]
    return out_ids, out_mask, out_labels


def discover_baseline_models(models_dir: str = "models") -> List[Tuple[str, str]]:
    """Return list of (model_path, display_name) for raw models under models/."""
    base = Path(models_dir)
    if not base.exists():
        return []
    found = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if (d / "config.json").exists():
            found.append((str(d), d.name))
    return found


@torch.no_grad()
def evaluate_baseline_qa(
    qwen_model,
    eval_loader: DataLoader,
    tokenizer,
    accelerator: Accelerator,
    max_new_tokens: int = 64,
    source_label: str = "dev",
) -> Tuple[Dict[str, float], List[SampleResult]]:
    """Evaluate raw Qwen (no compression) on QA as upper bound."""
    qwen_model.eval()
    unwrapped = accelerator.unwrap_model(qwen_model)

    all_em, all_f1, all_rouge, all_loss = [], [], [], []
    samples = []
    empty_count = 0
    global_idx = 0

    total_batches = len(eval_loader)
    for batch_idx, batch in enumerate(eval_loader):
        if accelerator.is_main_process and batch_idx % 10 == 0:
            pct = 100 * batch_idx / max(total_batches, 1)
            print(
                f"\r  [{source_label}] batch {batch_idx}/{total_batches} "
                f"({pct:.0f}%)",
                end="",
                flush=True,
            )

        # Extract text fields
        question_texts = batch.pop("question_text")
        answer_texts = batch.pop("answer_text")
        context_previews = batch.pop("context_preview")

        batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        doc_ids = batch["context_ids"]
        doc_mask = batch["context_mask"]
        q_ids = batch["prompt_ids"]
        q_mask = batch["prompt_mask"]
        ans_ids = batch["response_ids"]
        ans_mask = batch["response_mask"]

        # Teacher-forcing loss
        ignore_doc = torch.full_like(doc_ids, -100)
        ignore_q = torch.full_like(q_ids, -100)
        ans_labels = ans_ids.clone()
        ans_labels[ans_mask == 0] = -100

        input_ids, attn_mask, labels = _build_right_padded(
            [doc_ids, q_ids, ans_ids],
            [doc_mask, q_mask, ans_mask],
            [ignore_doc, ignore_q, ans_labels],
            tokenizer.pad_token_id,
        )
        outputs = unwrapped(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels,
            use_cache=False,
        )
        batch_loss = outputs.loss.detach().item()

        # Generate
        gen_input, gen_mask = _build_left_padded(
            [doc_ids, q_ids], [doc_mask, q_mask], tokenizer.pad_token_id
        )
        input_len = gen_input.shape[1]
        gen_out = unwrapped.generate(
            input_ids=gen_input,
            attention_mask=gen_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
        gen_only = gen_out[:, input_len:]
        preds = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for i, (pred, gold) in enumerate(zip(preds, answer_texts)):
            if not pred.strip():
                empty_count += 1
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            rl = compute_rouge_l(pred, gold)
            all_em.append(em)
            all_f1.append(f1)
            all_rouge.append(rl)
            all_loss.append(batch_loss)

            q_text = question_texts[i] if i < len(question_texts) else ""
            ctx = context_previews[i] if i < len(context_previews) else ""
            samples.append(
                SampleResult(
                    idx=global_idx,
                    question=q_text,
                    gold=gold,
                    prediction=pred,
                    context_preview=ctx,
                    em=em,
                    f1=f1,
                    rouge_l=rl,
                    source=source_label,
                )
            )
            global_idx += 1

    if accelerator.is_main_process:
        print()

    n = max(len(all_em), 1)
    avg_loss = sum(all_loss) / max(len(all_loss), 1)
    metrics = {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20.0)),
        "exact_match": sum(all_em) / n,
        "f1": sum(all_f1) / n,
        "rouge_l": sum(all_rouge) / n,
        "n_samples": len(all_em),
        "empty_preds": empty_count,
    }
    return metrics, samples


# ── Stdout: summary statistics ───────────────────────────────────────────────


def print_summary(results: List[ModelResult]):
    """Print concise summary to terminal."""

    print(f"\n{'=' * W}")
    print(f"  {BOLD}COMPARISON TABLE{RESET}")
    print(f"{'=' * W}")

    print(
        f"  {'':12}  {'--- Dev Set ---':^33}  "
        f"{'--- Train Peek ---':^33}"
    )
    header = (
        f"  {'Model':<12}  {'Loss':>6} {'PPL':>7} "
        f"{'EM':>7} {'F1':>7} {'RL':>7} {'Empty':>6}  "
        f"{'Loss':>6} {'PPL':>7} "
        f"{'EM':>7} {'F1':>7} {'RL':>7} {'Empty':>6}"
    )
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'-' * (W - 4)}")

    baselines = [r for r in results if r.k_value <= 0]
    compressed = [r for r in results if r.k_value > 0]

    best_f1 = max((r.dev_f1 for r in results), default=0)
    best_em = max((r.dev_em for r in results), default=0)

    def _print_row(r):
        name = r.name if r.k_value <= 0 else f"k={r.k_value}"
        em_s = f"{r.dev_em:>6.2%}"
        f1_s = f"{r.dev_f1:>7.4f}"
        if r.dev_f1 == best_f1 and len(results) > 1:
            f1_s = f"{GREEN}{f1_s}{RESET}"
        if r.dev_em == best_em and len(results) > 1:
            em_s = f"{GREEN}{em_s}{RESET}"
        de = f"{r.dev_empty_preds}" if r.dev_n > 0 else "-"
        te = f"{r.train_empty_preds}" if r.train_n > 0 else "-"
        print(
            f"  {name:<12}  {r.dev_loss:>6.3f} {r.dev_ppl:>7.2f} "
            f"{em_s} {f1_s} {r.dev_rouge_l:>7.4f} {de:>6}  "
            f"{r.train_loss:>6.3f} {r.train_ppl:>7.2f} "
            f"{r.train_em:>6.2%} {r.train_f1:>7.4f} {r.train_rouge_l:>7.4f} {te:>6}"
        )

    if baselines:
        for r in baselines:
            _print_row(r)
        print(f"  {'-' * (W - 4)}")

    for r in compressed:
        _print_row(r)

    print(f"  {'-' * (W - 4)}")

    # Overfitting analysis
    print(f"\n  {BOLD}Overfitting (train_F1 - dev_F1):{RESET}")
    for r in results:
        if r.train_n == 0:
            continue
        label = r.name if r.k_value <= 0 else f"k={r.k_value}"
        gap = r.train_f1 - r.dev_f1
        if gap > 0.10:
            tag = f"{RED}HIGH{RESET}"
        elif gap > 0.05:
            tag = f"{YELLOW}MODERATE{RESET}"
        else:
            tag = f"{GREEN}LOW{RESET}"
        print(f"    {label:<12} gap = {gap:+.4f}  {tag}")

    # Bar charts
    def _label(r):
        return r.name if r.k_value <= 0 else f"k={r.k_value}"

    print(f"\n  {BOLD}Dev F1:{RESET}")
    max_f1 = max((r.dev_f1 for r in results), default=0.001)
    for r in results:
        b = bar(r.dev_f1, max_f1, 35)
        print(f"    {_label(r):<12} {CYAN}{b}{RESET} {r.dev_f1:.4f}")

    print(f"\n  {BOLD}Dev EM:{RESET}")
    max_em = max((r.dev_em for r in results), default=0.001)
    for r in results:
        b = bar(r.dev_em, max_em, 35)
        print(f"    {_label(r):<12} {MAGENTA}{b}{RESET} {r.dev_em:.2%}")

    print(f"\n  {BOLD}Dev PPL (lower=better):{RESET}")
    max_ppl = max((r.dev_ppl for r in results), default=1)
    for r in results:
        b = bar(r.dev_ppl, max_ppl, 35)
        print(f"    {_label(r):<12} {YELLOW}{b}{RESET} {r.dev_ppl:.2f}")

    # Ranking
    print(f"\n{'=' * W}")
    print(f"  {BOLD}RANKING (by Dev F1){RESET}")
    print(f"{'=' * W}")
    ranked = sorted(results, key=lambda r: r.dev_f1, reverse=True)
    for i, r in enumerate(ranked):
        label = r.name if r.k_value <= 0 else f"k={r.k_value}"
        baseline_tag = " (baseline)" if r.k_value <= 0 else ""
        print(
            f"  {i+1}. {label:<12}  "
            f"F1={r.dev_f1:.4f}  EM={r.dev_em:.2%}  "
            f"PPL={r.dev_ppl:.2f}  Loss={r.dev_loss:.4f}  "
            f"({r.dev_n} samples){baseline_tag}"
        )
    print(f"{'=' * W}\n")


# ── Markdown report ──────────────────────────────────────────────────────────


def write_markdown_report(
    results: List[ModelResult],
    report_path: str,
    top_n: int = 10,
    args=None,
):
    """Write full per-sample analysis to a markdown file."""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("# Task Compressor QA Evaluation Report\n\n")
        f.write(f"Generated: {ts}\n\n")

        if args:
            f.write("## Evaluation Config\n\n")
            f.write("| Parameter | Value |\n|---|---|\n")
            f.write(f"| Base config | `{args.config}` |\n")
            f.write(f"| Dev data | `{args.eval_data}` |\n")
            f.write(f"| Train data | `{args.train_data}` |\n")
            f.write(
                f"| Max dev samples | "
                f"{args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'} |\n"
            )
            f.write(f"| Train peek samples | {args.train_peek_samples} |\n")
            f.write(f"| Models evaluated | {len(results)} |\n\n")

        # Summary table
        f.write("## Summary\n\n")

        def _md_name(r):
            return (
                f"**{r.name}** (baseline)"
                if r.k_value <= 0
                else f"k={r.k_value}"
            )

        f.write("### Dev Set\n\n")
        f.write(
            "| Model | Config | Loss | PPL | EM | F1 | ROUGE-L | Empty | N |\n"
        )
        f.write(
            "|-------|--------|------|-----|----|----|---------|-------|---|\n"
        )
        for r in results:
            empty_pct = 100 * r.dev_empty_preds / max(r.dev_n, 1)
            f.write(
                f"| {_md_name(r)} | {r.config_info} | {r.dev_loss:.4f} | "
                f"{r.dev_ppl:.2f} | {r.dev_em:.2%} | {r.dev_f1:.4f} | "
                f"{r.dev_rouge_l:.4f} | {r.dev_empty_preds} ({empty_pct:.1f}%) | "
                f"{r.dev_n} |\n"
            )

        if any(r.train_n > 0 for r in results):
            f.write("\n### Train Peek\n\n")
            f.write(
                "| Model | Loss | PPL | EM | F1 | ROUGE-L | Empty | N |\n"
            )
            f.write(
                "|-------|------|-----|----|----|---------|-------|---|\n"
            )
            for r in results:
                if r.train_n == 0:
                    continue
                empty_pct = 100 * r.train_empty_preds / max(r.train_n, 1)
                f.write(
                    f"| {_md_name(r)} | {r.train_loss:.4f} | "
                    f"{r.train_ppl:.2f} | {r.train_em:.2%} | {r.train_f1:.4f} | "
                    f"{r.train_rouge_l:.4f} | {r.train_empty_preds} ({empty_pct:.1f}%) | "
                    f"{r.train_n} |\n"
                )

        # Overfitting analysis
        if any(r.train_n > 0 for r in results):
            f.write("\n### Overfitting Analysis\n\n")
            f.write("| Model | Dev F1 | Train F1 | Gap | Level |\n")
            f.write("|-------|--------|----------|-----|-------|\n")
            for r in results:
                if r.train_n == 0:
                    continue
                gap = r.train_f1 - r.dev_f1
                level = (
                    "HIGH"
                    if gap > 0.10
                    else ("MODERATE" if gap > 0.05 else "LOW")
                )
                f.write(
                    f"| {_md_name(r)} | {r.dev_f1:.4f} | {r.train_f1:.4f} | "
                    f"{gap:+.4f} | {level} |\n"
                )

        # Per-model sample details
        ranked = sorted(results, key=lambda r: r.dev_f1, reverse=True)
        for r in ranked:
            title = r.name if r.k_value <= 0 else f"k={r.k_value}"
            baseline_note = (
                " (Baseline - no compression)" if r.k_value <= 0 else ""
            )
            f.write(f"\n---\n\n## {title}{baseline_note}\n\n")
            f.write(f"- **Checkpoint**: `{r.checkpoint_path}`\n")
            f.write(f"- **Config**: {r.config_info}\n")
            f.write(
                f"- **Dev**: Loss={r.dev_loss:.4f}  PPL={r.dev_ppl:.2f}  "
                f"EM={r.dev_em:.2%}  F1={r.dev_f1:.4f}  ROUGE-L={r.dev_rouge_l:.4f}\n"
            )
            if r.train_n > 0:
                f.write(
                    f"- **Train**: Loss={r.train_loss:.4f}  PPL={r.train_ppl:.2f}  "
                    f"EM={r.train_em:.2%}  F1={r.train_f1:.4f}  ROUGE-L={r.train_rouge_l:.4f}\n"
                )
            f.write(
                f"- **Empty predictions**: dev={r.dev_empty_preds}/{r.dev_n}"
            )
            if r.train_n > 0:
                f.write(
                    f", train={r.train_empty_preds}/{r.train_n}"
                )
            f.write("\n\n")

            for source_label in ["dev", "train"]:
                source_samples = [
                    s for s in r.samples if s.source == source_label
                ]
                if not source_samples:
                    continue

                tag = (
                    "Dev Set"
                    if source_label == "dev"
                    else "Train Set (peek)"
                )
                sorted_by_f1 = sorted(
                    source_samples, key=lambda s: s.f1, reverse=True
                )

                # Best N
                best = sorted_by_f1[:top_n]
                f.write(f"### Best {len(best)} - {tag}\n\n")
                _write_sample_table(f, best)

                # Worst N
                worst = sorted_by_f1[-top_n:]
                worst.reverse()
                f.write(f"### Worst {len(worst)} - {tag}\n\n")
                _write_sample_table(f, worst)

                # Empty predictions
                empty = [s for s in source_samples if not s.prediction.strip()]
                if empty:
                    shown = empty[:5]
                    f.write(
                        f"### Empty Predictions ({len(empty)} total) - {tag}\n\n"
                    )
                    _write_sample_table(f, shown)

    logger.info(f"Markdown report saved to {report_path}")


def _write_sample_table(f, samples: List[SampleResult]):
    """Write a list of samples as a readable markdown section."""
    for i, s in enumerate(samples, 1):
        em_icon = "V" if s.em == 1.0 else "X"
        f.write(
            f"**#{i}** [{em_icon}] EM={s.em:.0f} | F1={s.f1:.4f} | ROUGE-L={s.rouge_l:.4f}\n\n"
        )
        ctx = s.context_preview[:200].replace("\n", " ")
        f.write(f"> **Context**: {ctx}...\n\n")
        f.write(f"- **Question**: {s.question}\n")
        f.write(f"- **Gold**: {s.gold}\n")
        pred_display = s.prediction if s.prediction.strip() else "*(empty)*"
        f.write(f"- **Prediction**: {pred_display}\n\n")
    f.write("\n")


# ── CSV export ───────────────────────────────────────────────────────────────


def save_csv(results: List[ModelResult], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "k_value",
        "config_info",
        "checkpoint",
        "dev_loss",
        "dev_ppl",
        "dev_em",
        "dev_f1",
        "dev_rouge_l",
        "dev_empty_preds",
        "dev_n",
        "train_loss",
        "train_ppl",
        "train_em",
        "train_f1",
        "train_rouge_l",
        "train_empty_preds",
        "train_n",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x.k_value):
            writer.writerow(
                {
                    "model": r.name,
                    "k_value": r.k_value,
                    "config_info": r.config_info,
                    "checkpoint": r.checkpoint_path,
                    "dev_loss": f"{r.dev_loss:.6f}",
                    "dev_ppl": f"{r.dev_ppl:.4f}",
                    "dev_em": f"{r.dev_em:.6f}",
                    "dev_f1": f"{r.dev_f1:.6f}",
                    "dev_rouge_l": f"{r.dev_rouge_l:.6f}",
                    "dev_empty_preds": r.dev_empty_preds,
                    "dev_n": r.dev_n,
                    "train_loss": f"{r.train_loss:.6f}",
                    "train_ppl": f"{r.train_ppl:.4f}",
                    "train_em": f"{r.train_em:.6f}",
                    "train_f1": f"{r.train_f1:.6f}",
                    "train_rouge_l": f"{r.train_rouge_l:.6f}",
                    "train_empty_preds": r.train_empty_preds,
                    "train_n": r.train_n,
                }
            )
    logger.info(f"CSV saved to {output_path}")


# ── JSON export ──────────────────────────────────────────────────────────────


def save_json(results: List[ModelResult], output_path: str):
    """Save per-model metrics as JSON for programmatic consumption."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = []
    for r in sorted(results, key=lambda x: x.k_value):
        entry = {
            "model": r.name,
            "k_value": r.k_value,
            "config_info": r.config_info,
            "checkpoint": r.checkpoint_path,
            "dev": {
                "loss": r.dev_loss,
                "ppl": r.dev_ppl,
                "em": r.dev_em,
                "f1": r.dev_f1,
                "rouge_l": r.dev_rouge_l,
                "empty_preds": r.dev_empty_preds,
                "n_samples": r.dev_n,
            },
        }
        if r.train_n > 0:
            entry["train"] = {
                "loss": r.train_loss,
                "ppl": r.train_ppl,
                "em": r.train_em,
                "f1": r.train_f1,
                "rouge_l": r.train_rouge_l,
                "empty_preds": r.train_empty_preds,
                "n_samples": r.train_n,
            }
        data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Detailed QA evaluation across all Task Compressor experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/h200_base.yaml",
        help="Base config YAML (for shared params like base_model, data paths)",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Override dev data path (default: from config)",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Override train data path for peek (default: from config)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=500,
        help="Max dev samples (0=all, default=500)",
    )
    parser.add_argument(
        "--train_peek_samples",
        type=int,
        default=100,
        help="Training samples to peek (default=100)",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated checkpoint dirs (default: auto-discover)",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Base outputs dir for auto-discovery (default: outputs)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Dir containing baseline models (default: models)",
    )
    parser.add_argument(
        "--no_baselines",
        action="store_true",
        help="Skip baseline model evaluation",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Best/worst samples per model in report (default=10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size (default=8)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max tokens to generate per sample (default=128)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="results/qa_eval_report.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Also save CSV summary",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Also save JSON summary",
    )
    args = parser.parse_args()

    # ── Load base config ──
    base_config = Config.from_yaml(args.config)
    eval_data = args.eval_data or base_config.data.dev_file
    train_data = args.train_data or base_config.data.train_file

    # Stash resolved data paths on args for report
    args.eval_data = eval_data
    args.train_data = train_data

    # ── Discover checkpoints ──
    if args.checkpoints:
        entries = []
        for p in args.checkpoints.split(","):
            p = p.strip()
            ckpt_dir = Path(p)
            if not (ckpt_dir / "task_compressor_modules.pt").exists():
                logger.warning(f"No task_compressor_modules.pt in {p}, skipping")
                continue
            try:
                n_p, n_c = infer_config_from_checkpoint(ckpt_dir)
            except Exception as e:
                logger.warning(f"Cannot infer config from {ckpt_dir}: {e}")
                continue
            entries.append((ckpt_dir, n_p, n_c, n_p + n_c))
        entries.sort(key=lambda x: (x[3], str(x[0])))
    else:
        entries = discover_checkpoints(args.outputs_dir)

    if not entries:
        logger.error(
            "No checkpoints found. Train some models first, or use --checkpoints."
        )
        return

    # ── Initialize ──
    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        baseline_models = (
            [] if args.no_baselines else discover_baseline_models(args.models_dir)
        )
        print(f"\n{'=' * W}")
        print(f"  {BOLD}TASK COMPRESSOR - QA EVALUATION{RESET}")
        print(f"{'=' * W}")
        print(
            f"  Compressed:  {len(entries)}  "
            f"({', '.join(f'k={k}' for _, _, _, k in entries)})"
        )
        if baseline_models:
            print(
                f"  Baselines:   {len(baseline_models)}  "
                f"({', '.join(n for _, n in baseline_models)})"
            )
        print(f"  Base config: {args.config}")
        print(f"  Dev data:    {eval_data}")
        print(
            f"  Dev samples: "
            f"{args.max_eval_samples if args.max_eval_samples > 0 else 'ALL'}"
        )
        print(f"  Train peek:  {args.train_peek_samples}")
        print(f"  Report:      {args.report}")
        print(f"  Device:      {accelerator.device}")
        print()

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        base_config.model.base_model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Shared data params ──
    max_ctx = base_config.data.max_context_length
    max_prompt = base_config.data.max_prompt_length
    max_resp = base_config.data.max_response_length

    # ── Evaluate each model ──
    all_results: List[ModelResult] = []

    # ── Baseline models (raw Qwen, no compression) ──
    if not args.no_baselines:
        baseline_models = discover_baseline_models(args.models_dir)
        for model_path, display_name in baseline_models:
            if is_main:
                print(f"{'=' * W}")
                print(f"  {BOLD}{display_name} (baseline){RESET}  ({model_path})")
                print(f"  Loading model...")

            qwen = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available()
                    else torch.float32
                ),
            )
            qwen.eval()
            for p in qwen.parameters():
                p.requires_grad = False
            qwen = accelerator.prepare(qwen)

            collator = EvalQACollator(
                pad_token_id=tokenizer.pad_token_id,
                max_context_length=max_ctx,
                max_prompt_length=max_prompt,
                max_response_length=max_resp,
            )
            nw = 0 if accelerator.device.type == "mps" else 2
            pin = accelerator.device.type == "cuda"

            mr = ModelResult(
                name=display_name,
                k_value=0,
                config_info="baseline (no compression)",
                checkpoint_path=model_path,
            )

            # Dev
            if is_main:
                print(f"  Loading dev data...")
            dev_ds = EvalQADataset(
                eval_data,
                tokenizer,
                max_context_length=max_ctx,
                max_prompt_length=max_prompt,
                max_response_length=max_resp,
            )
            if 0 < args.max_eval_samples < len(dev_ds):
                dev_ds = Subset(dev_ds, list(range(args.max_eval_samples)))

            dev_loader = DataLoader(
                dev_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=nw,
                pin_memory=pin,
            )
            dev_loader = accelerator.prepare(dev_loader)

            dev_m, dev_s = evaluate_baseline_qa(
                qwen,
                dev_loader,
                tokenizer,
                accelerator,
                max_new_tokens=args.max_new_tokens,
                source_label="dev",
            )

            mr.dev_loss = dev_m["loss"]
            mr.dev_ppl = dev_m["perplexity"]
            mr.dev_em = dev_m["exact_match"]
            mr.dev_f1 = dev_m["f1"]
            mr.dev_rouge_l = dev_m["rouge_l"]
            mr.dev_n = dev_m["n_samples"]
            mr.dev_empty_preds = dev_m["empty_preds"]
            mr.samples.extend(dev_s)

            if is_main:
                print(
                    f"  {GREEN}Dev:{RESET}   Loss={mr.dev_loss:.4f}  "
                    f"PPL={mr.dev_ppl:.2f}  EM={mr.dev_em:.2%}  "
                    f"F1={mr.dev_f1:.4f}  Empty={mr.dev_empty_preds}/{mr.dev_n}"
                )

            # Train peek
            if args.train_peek_samples > 0 and os.path.exists(train_data):
                train_ds = EvalQADataset(
                    train_data,
                    tokenizer,
                    max_context_length=max_ctx,
                    max_prompt_length=max_prompt,
                    max_response_length=max_resp,
                )
                if args.train_peek_samples < len(train_ds):
                    train_ds = Subset(
                        train_ds, list(range(args.train_peek_samples))
                    )

                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collator,
                    num_workers=nw,
                    pin_memory=pin,
                )
                train_loader = accelerator.prepare(train_loader)

                tr_m, tr_s = evaluate_baseline_qa(
                    qwen,
                    train_loader,
                    tokenizer,
                    accelerator,
                    max_new_tokens=args.max_new_tokens,
                    source_label="train",
                )

                mr.train_loss = tr_m["loss"]
                mr.train_ppl = tr_m["perplexity"]
                mr.train_em = tr_m["exact_match"]
                mr.train_f1 = tr_m["f1"]
                mr.train_rouge_l = tr_m["rouge_l"]
                mr.train_n = tr_m["n_samples"]
                mr.train_empty_preds = tr_m["empty_preds"]
                mr.samples.extend(tr_s)

                if is_main:
                    print(
                        f"  {YELLOW}Train:{RESET} Loss={mr.train_loss:.4f}  "
                        f"PPL={mr.train_ppl:.2f}  EM={mr.train_em:.2%}  "
                        f"F1={mr.train_f1:.4f}  "
                        f"Empty={mr.train_empty_preds}/{mr.train_n}"
                    )

            all_results.append(mr)
            del qwen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Compressed models (Task Compressor) ──
    for ckpt_dir, n_p, n_c, k in entries:
        config_info = f"n_p={n_p}, n_c={n_c}, k={k}"
        # Derive a readable name from the path
        # e.g., outputs/ablation_compression/k512_np128_nc384/best -> k512_np128_nc384
        name_parts = []
        for part in ckpt_dir.parts:
            if re.match(r"k\d+", part):
                name_parts.append(part)
        exp_name = "/".join(name_parts) if name_parts else ckpt_dir.parent.name

        if is_main:
            print(f"{'=' * W}")
            print(f"  {BOLD}k={k}{RESET}  ({config_info})  [{ckpt_dir}]")
            print(f"  Loading model...")

        model = load_model(
            ckpt_dir, base_config, n_p, n_c, accelerator
        )

        collator = EvalQACollator(
            pad_token_id=tokenizer.pad_token_id,
            max_context_length=max_ctx,
            max_prompt_length=max_prompt,
            max_response_length=max_resp,
        )
        nw = 0 if accelerator.device.type == "mps" else 2
        pin = accelerator.device.type == "cuda"

        mr = ModelResult(
            name=exp_name,
            k_value=k,
            config_info=config_info,
            checkpoint_path=str(ckpt_dir),
        )

        # Dev
        if is_main:
            print(f"  Loading dev data...")
        dev_ds = EvalQADataset(
            eval_data,
            tokenizer,
            max_context_length=max_ctx,
            max_prompt_length=max_prompt,
            max_response_length=max_resp,
        )
        if 0 < args.max_eval_samples < len(dev_ds):
            dev_ds = Subset(dev_ds, list(range(args.max_eval_samples)))

        dev_loader = DataLoader(
            dev_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=nw,
            pin_memory=pin,
        )
        dev_loader = accelerator.prepare(dev_loader)

        dev_m, dev_s = evaluate_qa_detailed(
            model,
            dev_loader,
            tokenizer,
            accelerator,
            max_new_tokens=args.max_new_tokens,
            source_label="dev",
        )

        mr.dev_loss = dev_m["loss"]
        mr.dev_ppl = dev_m["perplexity"]
        mr.dev_em = dev_m["exact_match"]
        mr.dev_f1 = dev_m["f1"]
        mr.dev_rouge_l = dev_m["rouge_l"]
        mr.dev_n = dev_m["n_samples"]
        mr.dev_empty_preds = dev_m["empty_preds"]
        mr.samples.extend(dev_s)

        if is_main:
            print(
                f"  {GREEN}Dev:{RESET}   Loss={mr.dev_loss:.4f}  "
                f"PPL={mr.dev_ppl:.2f}  EM={mr.dev_em:.2%}  "
                f"F1={mr.dev_f1:.4f}  Empty={mr.dev_empty_preds}/{mr.dev_n}"
            )

        # Train peek
        if args.train_peek_samples > 0 and os.path.exists(train_data):
            train_ds = EvalQADataset(
                train_data,
                tokenizer,
                max_context_length=max_ctx,
                max_prompt_length=max_prompt,
                max_response_length=max_resp,
            )
            if args.train_peek_samples < len(train_ds):
                train_ds = Subset(
                    train_ds, list(range(args.train_peek_samples))
                )

            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=nw,
                pin_memory=pin,
            )
            train_loader = accelerator.prepare(train_loader)

            tr_m, tr_s = evaluate_qa_detailed(
                model,
                train_loader,
                tokenizer,
                accelerator,
                max_new_tokens=args.max_new_tokens,
                source_label="train",
            )

            mr.train_loss = tr_m["loss"]
            mr.train_ppl = tr_m["perplexity"]
            mr.train_em = tr_m["exact_match"]
            mr.train_f1 = tr_m["f1"]
            mr.train_rouge_l = tr_m["rouge_l"]
            mr.train_n = tr_m["n_samples"]
            mr.train_empty_preds = tr_m["empty_preds"]
            mr.samples.extend(tr_s)

            if is_main:
                print(
                    f"  {YELLOW}Train:{RESET} Loss={mr.train_loss:.4f}  "
                    f"PPL={mr.train_ppl:.2f}  EM={mr.train_em:.2%}  "
                    f"F1={mr.train_f1:.4f}  "
                    f"Empty={mr.train_empty_preds}/{mr.train_n}"
                )

        all_results.append(mr)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Output (main process only) ──
    if not is_main:
        return

    print_summary(all_results)
    write_markdown_report(all_results, args.report, top_n=args.top_n, args=args)
    print(f"  Report saved to: {BOLD}{args.report}{RESET}")

    if args.csv:
        save_csv(all_results, args.csv)
        print(f"  CSV saved to:    {BOLD}{args.csv}{RESET}")

    if args.json:
        save_json(all_results, args.json)
        print(f"  JSON saved to:   {BOLD}{args.json}{RESET}")

    print()


if __name__ == "__main__":
    main()

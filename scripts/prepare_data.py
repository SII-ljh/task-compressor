#!/usr/bin/env python3
"""Prepare training datasets for Deep Compressor.

Default: download complete datasets for recommended training (~1.7B tokens).
--test:       download small subsets for quick MacBook validation.
--make-tiny:  extract tiny subsets from existing local data for hyperparameter tuning.

NTP datasets (Stage 1, target ~1.5B tokens):
  - WikiText-103-raw   (~130M tokens, ~28K docs)
  - SQuAD v1.1 contexts (~2M tokens, ~442 docs)
  - C4 realnewslike     (~700M tokens, English, streamed)
  - CLUECorpusSmall     (~500M tokens, Chinese, streamed)

QA datasets (Stage 2):
  - SQuAD v1.1      87,599 train / 10,570 dev  (English)
  - CMRC2018         10,142 train / 3,219 dev   (Chinese)
  - DuReader-robust  14,520 train / 1,417 dev   (Chinese)
  - TriviaQA RC      ~95K train / ~12K dev       (English)
  - DRCD             ~27K train / ~4K dev        (Traditional Chinese)

For model downloads, use scripts/download_models.py instead.

Estimated download time (full mode): ~1-2 hours (C4 and CLUECorpus streaming).

Usage:
  python scripts/prepare_data.py                    # full download
  python scripts/prepare_data.py --test             # small subset
  python scripts/prepare_data.py --make-tiny        # ~50 samples for smoke tests
  python scripts/prepare_data.py --make-dev         # ~500-2000 for HP tuning
  python scripts/prepare_data.py --make-ablation    # 5% of total for ablation experiments
  python scripts/prepare_data.py --make-all-subsets # all subsets at once
"""

import argparse
import json
import random
import time
import traceback
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ── 1. NTP data ──────────────────────────────────────────────────────
def prepare_ntp_data(test_mode: bool = False):
    """Build NTP pretraining data from multiple sources.

    Full mode target: ~1.5B tokens
      WikiText-103 (~130M) + SQuAD (~2M) + C4 (~700M) + Chinese (~500M)
    Test mode: WikiText-2 + SQuAD, cap 3000 docs
    """
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "ntp_train.jsonl"
    documents = []
    min_chars = 200

    # ── WikiText ──
    wt_config = "wikitext-2-raw-v1" if test_mode else "wikitext-103-raw-v1"
    print(f"[NTP] Loading {wt_config} ...")
    wt = load_dataset("wikitext", wt_config, split="train")

    current_doc = []
    for row in wt:
        line = row["text"].strip()
        if not line:
            if current_doc:
                doc = " ".join(current_doc)
                if len(doc) >= min_chars:
                    documents.append(doc)
                current_doc = []
        else:
            if line.startswith("= ") and line.endswith(" ="):
                continue
            current_doc.append(line)
    if current_doc:
        doc = " ".join(current_doc)
        if len(doc) >= min_chars:
            documents.append(doc)

    print(f"  → WikiText: {len(documents):,} docs")

    # ── SQuAD contexts grouped by article ──
    print("[NTP] Loading SQuAD v1.1 contexts ...")
    squad = load_dataset("rajpurkar/squad", split="train")

    title_contexts = {}
    for row in squad:
        title = row["title"]
        ctx = row["context"].strip()
        if title not in title_contexts:
            title_contexts[title] = []
        if ctx not in title_contexts[title]:
            title_contexts[title].append(ctx)

    before = len(documents)
    for title, contexts in title_contexts.items():
        doc = " ".join(contexts)
        if len(doc) >= min_chars:
            documents.append(doc)

    print(f"  → SQuAD articles: {len(documents) - before} docs, total: {len(documents):,}")

    # ── C4 realnewslike (English, streaming) ──
    if not test_mode:
        # ~700M tokens ≈ 2.8B chars at ~4 chars/token for English
        C4_TARGET_CHARS = 2_800_000_000
        try:
            print("[NTP] Streaming C4 realnewslike (target ~700M tokens) ...")
            print("      This may take 30-60 minutes depending on network speed.")
            c4 = load_dataset("allenai/c4", "realnewslike",
                              split="train", streaming=True)
            c4_chars = 0
            c4_count = 0
            t0 = time.time()
            for row in c4:
                text = row["text"].strip()
                if len(text) < min_chars:
                    continue
                documents.append(text)
                c4_chars += len(text)
                c4_count += 1
                if c4_count % 50_000 == 0:
                    elapsed = time.time() - t0
                    pct = 100 * c4_chars / C4_TARGET_CHARS
                    print(f"    C4: {c4_count:,} docs, ~{c4_chars // 4_000_000}M tok, "
                          f"{pct:.0f}%, {elapsed:.0f}s")
                if c4_chars >= C4_TARGET_CHARS:
                    break
            print(f"  → C4 realnewslike: {c4_count:,} docs, ~{c4_chars // 4_000_000}M tokens")
        except Exception as e:
            print(f"  → C4: skipped ({e})")
            traceback.print_exc()

    # ── Chinese corpus (CLUECorpusSmall → fallback mc4 Chinese) ──
    if not test_mode:
        # ~500M tokens ≈ 1B chars at ~2 chars/token for Chinese
        ZH_TARGET_CHARS = 1_000_000_000
        zh_loaded = False
        sources = [
            ("CLUECorpusSmall",
             lambda: load_dataset("austenjs/ClueCorpusSmallDataset",
                                  split="train", streaming=True)),
            ("mc4 Chinese",
             lambda: load_dataset("mc4", "zh",
                                  split="train", streaming=True)),
        ]
        for source_name, load_fn in sources:
            if zh_loaded:
                break
            try:
                print(f"[NTP] Streaming {source_name} (target ~500M tokens) ...")
                ds = load_fn()
                zh_chars = 0
                zh_count = 0
                t0 = time.time()
                for row in ds:
                    text = row["text"].strip()
                    if len(text) < min_chars:
                        continue
                    documents.append(text)
                    zh_chars += len(text)
                    zh_count += 1
                    if zh_count % 50_000 == 0:
                        elapsed = time.time() - t0
                        pct = 100 * zh_chars / ZH_TARGET_CHARS
                        print(f"    {source_name}: {zh_count:,} docs, "
                              f"~{zh_chars // 2_000_000}M tok, {pct:.0f}%, {elapsed:.0f}s")
                    if zh_chars >= ZH_TARGET_CHARS:
                        break
                print(f"  → {source_name}: {zh_count:,} docs, ~{zh_chars // 2_000_000}M tokens")
                zh_loaded = True
            except Exception as e:
                print(f"  → {source_name}: failed ({e})")
                traceback.print_exc()

        if not zh_loaded:
            print("  → Chinese NTP: all sources failed, skipping")

    # ── Shuffle and save ──
    random.seed(42)
    random.shuffle(documents)
    if test_mode:
        documents = documents[:3000]

    print(f"[NTP] Writing {len(documents):,} docs to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

    size_bytes = out_path.stat().st_size
    if size_bytes > 1e9:
        print(f"  → {len(documents):,} docs, {size_bytes / 1e9:.2f} GB")
    else:
        print(f"  → {len(documents):,} docs, {size_bytes / 1e6:.1f} MB")
    print()


# ── 2. QA data ────────────────────────────────────────────────────────
def _convert_squad(dataset, max_n: int = 0, source: str = "squad"):
    """Convert SQuAD-format dataset to [{context, question, answer, source}]."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answers = row["answers"]["text"]
        if not answers:
            continue
        samples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answers[0],
            "source": source,
        })
    return samples


def _convert_cmrc_or_drcd(dataset, max_n: int = 0, source: str = "cmrc"):
    """Convert CMRC2018 / DRCD dataset (SQuAD-like, answers.text)."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            continue
        samples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answers[0],
            "source": source,
        })
    return samples


def _convert_dureader(dataset, max_n: int = 0, source: str = "dureader"):
    """Convert DuReader-robust dataset."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            answer = row.get("answer", "")
            if not answer:
                continue
            answers = [answer]
        samples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answers[0],
            "source": source,
        })
    return samples


def _convert_triviaqa(dataset, max_n: int = 0, max_context_chars: int = 32000,
                      source: str = "triviaqa"):
    """Convert TriviaQA RC dataset.

    Extracts context from entity_pages.wiki_context or search_results.search_context.
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answer_val = row.get("answer", {}).get("value", "")
        if not answer_val:
            continue
        # Try entity_pages wiki_context first, then search_results
        context = ""
        wiki_ctxs = row.get("entity_pages", {}).get("wiki_context", [])
        if wiki_ctxs:
            context = wiki_ctxs[0].strip()
        if not context or len(context) < 100:
            search_ctxs = row.get("search_results", {}).get("search_context", [])
            if search_ctxs:
                context = search_ctxs[0].strip()
        if not context or len(context) < 100:
            continue
        # Cap context length to save disk
        if len(context) > max_context_chars:
            context = context[:max_context_chars]
        samples.append({
            "context": context,
            "question": row["question"],
            "answer": answer_val,
            "source": source,
        })
    return samples


def prepare_qa_data(test_mode: bool = False):
    """Download and merge QA datasets.

    Full mode:  SQuAD + CMRC2018 + DuReader + TriviaQA + DRCD
    Test mode:  SQuAD (5K train, 1K dev) only
    """
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_all = []
    dev_all = []

    max_train = 5000 if test_mode else 0
    max_dev = 1000 if test_mode else 0

    # ── SQuAD v1.1 ──
    print("[QA] Loading SQuAD v1.1 ...")
    sq_train = _convert_squad(load_dataset("rajpurkar/squad", split="train"), max_train, source="squad")
    sq_dev = _convert_squad(load_dataset("rajpurkar/squad", split="validation"), max_dev, source="squad")
    train_all.extend(sq_train)
    dev_all.extend(sq_dev)
    print(f"  → SQuAD: train={len(sq_train):,}, dev={len(sq_dev):,}")

    # ── CMRC2018 (Chinese) ──
    if not test_mode:
        try:
            print("[QA] Loading CMRC2018 ...")
            cm_train = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="train"), source="cmrc")
            cm_dev = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="validation"), source="cmrc")
            train_all.extend(cm_train)
            dev_all.extend(cm_dev)
            print(f"  → CMRC2018: train={len(cm_train):,}, dev={len(cm_dev):,}")
        except Exception as e:
            print(f"  → CMRC2018: skipped ({e})")
            traceback.print_exc()

    # ── DuReader-robust (Chinese) ──
    if not test_mode:
        try:
            print("[QA] Loading DuReader-robust ...")
            loaded = False
            for repo in ["PaddlePaddle/dureader_robust", "luozhouyang/dureader"]:
                try:
                    d_train = _convert_dureader(load_dataset(repo, split="train"), source="dureader")
                    d_dev = _convert_dureader(load_dataset(repo, split="validation"), source="dureader")
                    train_all.extend(d_train)
                    dev_all.extend(d_dev)
                    print(f"  → DuReader ({repo}): train={len(d_train):,}, dev={len(d_dev):,}")
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                raise RuntimeError("No compatible DuReader repo found")
        except Exception as e:
            print(f"  → DuReader: skipped ({e})")
            traceback.print_exc()

    # ── TriviaQA RC (English) ──
    if not test_mode:
        try:
            print("[QA] Loading TriviaQA (rc) ... (streaming, may take a few minutes)")
            tqa_train_ds = load_dataset("trivia_qa", "rc",
                                        split="train", streaming=True)
            tqa_dev_ds = load_dataset("trivia_qa", "rc",
                                      split="validation", streaming=True)
            tqa_train = _convert_triviaqa(tqa_train_ds, source="triviaqa")
            tqa_dev = _convert_triviaqa(tqa_dev_ds, source="triviaqa")
            train_all.extend(tqa_train)
            dev_all.extend(tqa_dev)
            print(f"  → TriviaQA: train={len(tqa_train):,}, dev={len(tqa_dev):,}")
        except Exception as e:
            print(f"  → TriviaQA: skipped ({e})")
            traceback.print_exc()

    # ── DRCD (Traditional Chinese) ──
    if not test_mode:
        drcd_loaded = False
        drcd_sources = [
            ("clue/drcd",
             lambda: load_dataset("clue", "drcd", split="train"),
             lambda: load_dataset("clue", "drcd", split="validation")),
            ("hfl/drcd",
             lambda: load_dataset("hfl/drcd", split="train"),
             lambda: load_dataset("hfl/drcd", split="validation")),
        ]
        for repo_name, load_train, load_dev in drcd_sources:
            if drcd_loaded:
                break
            try:
                print(f"[QA] Loading DRCD from {repo_name} ...")
                dr_train = _convert_cmrc_or_drcd(load_train(), source="drcd")
                dr_dev = _convert_cmrc_or_drcd(load_dev(), source="drcd")
                train_all.extend(dr_train)
                dev_all.extend(dr_dev)
                print(f"  → DRCD ({repo_name}): train={len(dr_train):,}, dev={len(dr_dev):,}")
                drcd_loaded = True
            except Exception as e:
                print(f"  → {repo_name}: failed ({e})")
                continue

        if not drcd_loaded:
            print("  → DRCD: all sources failed, skipping")

    # ── Shuffle and save ──
    random.seed(42)
    random.shuffle(train_all)

    train_path = DATA_DIR / "qa_train.json"
    dev_path = DATA_DIR / "qa_dev.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_all, f, ensure_ascii=False)
    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_all, f, ensure_ascii=False)

    print(f"  → QA train: {len(train_all):,} samples → {train_path} "
          f"({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  → QA dev:   {len(dev_all):,} samples → {dev_path} "
          f"({dev_path.stat().st_size / 1e6:.1f} MB)")
    print()


# ── helpers ─────────────────────────────────────────────────────────────

def _check_source_files():
    """Check that full data files exist; return list of missing paths."""
    ntp_src = DATA_DIR / "ntp_train.jsonl"
    qa_train_src = DATA_DIR / "qa_train.json"
    qa_dev_src = DATA_DIR / "qa_dev.json"
    missing = [p for p in [ntp_src, qa_train_src, qa_dev_src] if not p.exists()]
    if missing:
        print("ERROR: Source files not found:")
        for p in missing:
            print(f"  {p}")
        print("\nPlease run full data preparation first:")
        print("  python scripts/prepare_data.py")
    return missing


def _reservoir_sample(filepath, n, rng):
    """Reservoir-sample n lines from a JSONL file."""
    reservoir = []
    line_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, line_count - 1)
                if j < n:
                    reservoir[j] = line
    return reservoir, line_count


def _stratified_sample_qa(data, n, rng):
    """Stratified sample from QA data by 'source' field."""
    by_source = {}
    for item in data:
        src = item.get("source", "unknown")
        by_source.setdefault(src, []).append(item)

    result = []
    # Proportional allocation
    remaining = n
    sources = sorted(by_source.keys())
    for i, src in enumerate(sources):
        items = by_source[src]
        if i == len(sources) - 1:
            alloc = remaining
        else:
            alloc = max(1, round(len(items) / len(data) * n))
            alloc = min(alloc, remaining)
        sampled = rng.sample(items, min(alloc, len(items)))
        result.extend(sampled)
        remaining -= len(sampled)

    rng.shuffle(result)
    return result[:n]


def _write_jsonl(data_lines, path):
    """Write list of JSON strings to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for line in data_lines:
            f.write(line + "\n")


def _write_json(data, path):
    """Write a JSON list to file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _print_file_info(path, label=None):
    """Print file size info."""
    name = label or path.name
    size = path.stat().st_size
    if size > 1e6:
        print(f"  → {name}: {size / 1e6:.1f} MB")
    else:
        print(f"  → {name}: {size / 1e3:.1f} KB")


# ── 3. Make tiny subset from existing data ─────────────────────────────
def make_tiny_subset(ntp_n: int = 50, qa_train_n: int = 50,
                     qa_dev_n: int = 20, seed: int = 42):
    """Extract tiny subsets (~50 samples) for pipeline smoke tests.

    NTP: uses reservoir sampling to avoid loading 5.8GB into memory.
    QA:  uses random.sample on the JSON arrays (already fits in memory).
    """
    if _check_source_files():
        return

    rng = random.Random(seed)

    # ── NTP: reservoir sampling ──
    ntp_out = DATA_DIR / "ntp_tiny.jsonl"
    print(f"[Tiny] Sampling {ntp_n} lines from ntp_train.jsonl (reservoir sampling) ...")
    reservoir, line_count = _reservoir_sample(DATA_DIR / "ntp_train.jsonl", ntp_n, rng)
    _write_jsonl(reservoir, ntp_out)
    print(f"  → {ntp_out.name}: {len(reservoir)} lines (from {line_count:,})")

    # ── QA train: random.sample ──
    qa_train_out = DATA_DIR / "qa_tiny_train.json"
    print(f"[Tiny] Sampling {qa_train_n} from qa_train.json ...")
    with open(DATA_DIR / "qa_train.json", "r", encoding="utf-8") as f:
        qa_train_data = json.load(f)
    n = min(qa_train_n, len(qa_train_data))
    _write_json(rng.sample(qa_train_data, n), qa_train_out)
    print(f"  → {qa_train_out.name}: {n} samples (from {len(qa_train_data):,})")

    # ── QA dev: random.sample ──
    qa_dev_out = DATA_DIR / "qa_tiny_dev.json"
    print(f"[Tiny] Sampling {qa_dev_n} from qa_dev.json ...")
    with open(DATA_DIR / "qa_dev.json", "r", encoding="utf-8") as f:
        qa_dev_data = json.load(f)
    n = min(qa_dev_n, len(qa_dev_data))
    _write_json(rng.sample(qa_dev_data, n), qa_dev_out)
    print(f"  → {qa_dev_out.name}: {n} samples (from {len(qa_dev_data):,})")

    print()
    print("Tiny subset files ready:")
    for p in [ntp_out, qa_train_out, qa_dev_out]:
        print(f"  {p}")
    print()


# ── 4. Make dev subset ─────────────────────────────────────────────────
def make_dev_subset(ntp_n: int = 2000, qa_dev_n: int = 500, seed: int = 42):
    """Create representative dev subsets for HP tuning evaluation.

    NTP: stratified by document length (short <1K / mid 1K-10K / long >10K chars).
    QA:  stratified by source field.
    """
    if _check_source_files():
        return

    rng = random.Random(seed)

    # ── NTP: stratified by doc length ──
    ntp_out = DATA_DIR / "ntp_dev.jsonl"
    print(f"[Dev] Building {ntp_n}-line NTP dev set (stratified by doc length) ...")

    # Bucket documents
    buckets = {"short": [], "mid": [], "long": []}
    line_count = 0
    with open(DATA_DIR / "ntp_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                doc_len = len(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                continue
            if doc_len < 1000:
                buckets["short"].append(line)
            elif doc_len < 10000:
                buckets["mid"].append(line)
            else:
                buckets["long"].append(line)

    # Proportional allocation with minimum 1 per bucket
    total_in_buckets = sum(len(v) for v in buckets.values())
    result = []
    remaining = ntp_n
    bucket_names = sorted(buckets.keys())
    for i, bname in enumerate(bucket_names):
        items = buckets[bname]
        if not items:
            continue
        if i == len(bucket_names) - 1:
            alloc = remaining
        else:
            alloc = max(1, round(len(items) / max(total_in_buckets, 1) * ntp_n))
            alloc = min(alloc, remaining)
        sampled = rng.sample(items, min(alloc, len(items)))
        result.extend(sampled)
        remaining -= len(sampled)
        print(f"    {bname}: {len(items):,} total → sampled {len(sampled)}")

    rng.shuffle(result)
    _write_jsonl(result, ntp_out)
    print(f"  → {ntp_out.name}: {len(result)} lines (from {line_count:,})")

    # ── QA dev: stratified by source ──
    qa_out = DATA_DIR / "qa_dev_hp.json"
    print(f"[Dev] Building {qa_dev_n}-sample QA dev set (stratified by source) ...")
    with open(DATA_DIR / "qa_dev.json", "r", encoding="utf-8") as f:
        qa_dev_data = json.load(f)

    sampled = _stratified_sample_qa(qa_dev_data, qa_dev_n, rng)
    _write_json(sampled, qa_out)
    print(f"  → {qa_out.name}: {len(sampled)} samples (from {len(qa_dev_data):,})")
    print()


# ── 5. Make ablation subset ────────────────────────────────────────────
def make_ablation_subset(ratio: float = 0.05, seed: int = 123):
    """Create fixed ablation subsets at a given ratio (default 5%) of total data.

    Uses a different seed from other subsets for independence.
    Writes to data/ablation/ directory.
    """
    if _check_source_files():
        return

    rng = random.Random(seed)
    ablation_dir = DATA_DIR / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation] Target ratio: {ratio:.0%} of total data (seed={seed})")

    # ── NTP: count total lines first, then reservoir sample ──
    ntp_src = DATA_DIR / "ntp_train.jsonl"
    print(f"[Ablation] Counting NTP lines in {ntp_src.name} ...")
    ntp_total = sum(1 for line in open(ntp_src, "r", encoding="utf-8") if line.strip())
    ntp_n = max(1, round(ntp_total * ratio))

    ntp_out = ablation_dir / "ntp_ablation.jsonl"
    print(f"[Ablation] Sampling {ntp_n} NTP lines ({ratio:.0%} of {ntp_total:,}) ...")
    reservoir, _ = _reservoir_sample(ntp_src, ntp_n, rng)
    _write_jsonl(reservoir, ntp_out)
    print(f"  → {ntp_out.name}: {len(reservoir):,} lines")

    # ── QA train: stratified sample ──
    qa_train_out = ablation_dir / "qa_ablation_train.json"
    with open(DATA_DIR / "qa_train.json", "r", encoding="utf-8") as f:
        qa_train_data = json.load(f)
    qa_train_n = max(1, round(len(qa_train_data) * ratio))
    print(f"[Ablation] Sampling {qa_train_n} QA train ({ratio:.0%} of {len(qa_train_data):,}) ...")
    sampled = _stratified_sample_qa(qa_train_data, qa_train_n, rng)
    _write_json(sampled, qa_train_out)
    print(f"  → {qa_train_out.name}: {len(sampled):,} samples")

    # ── QA dev: stratified sample ──
    qa_dev_out = ablation_dir / "qa_ablation_dev.json"
    with open(DATA_DIR / "qa_dev.json", "r", encoding="utf-8") as f:
        qa_dev_data = json.load(f)
    qa_dev_n = max(1, round(len(qa_dev_data) * ratio))
    print(f"[Ablation] Sampling {qa_dev_n} QA dev ({ratio:.0%} of {len(qa_dev_data):,}) ...")
    sampled = _stratified_sample_qa(qa_dev_data, qa_dev_n, rng)
    _write_json(sampled, qa_dev_out)
    print(f"  → {qa_dev_out.name}: {len(sampled):,} samples")
    print()


# ── 6. Make all subsets at once ────────────────────────────────────────
def make_all_subsets():
    """Generate all subsets (tiny, dev, ablation) in one go."""
    print("=" * 60)
    print("Generating all subsets...")
    print("=" * 60)
    print()
    make_tiny_subset()
    make_dev_subset()
    make_ablation_subset()
    print("All subsets created.")
    print()


# ── Summary ─────────────────────────────────────────────────────────────
def print_summary():
    print("=" * 60)
    print("Preparation complete! File summary:")
    print("-" * 60)
    for path in sorted(DATA_DIR.glob("*")):
        if path.is_file():
            size = path.stat().st_size
            if size > 1e9:
                print(f"  {path.name:30s}  {size/1e9:.2f} GB")
            elif size > 1e6:
                print(f"  {path.name:30s}  {size/1e6:.1f} MB")
            else:
                print(f"  {path.name:30s}  {size/1e3:.1f} KB")
    # Print ablation subdirectory
    ablation_dir = DATA_DIR / "ablation"
    if ablation_dir.exists():
        for path in sorted(ablation_dir.glob("*")):
            if path.is_file():
                size = path.stat().st_size
                name = f"ablation/{path.name}"
                if size > 1e6:
                    print(f"  {name:30s}  {size/1e6:.1f} MB")
                else:
                    print(f"  {name:30s}  {size/1e3:.1f} KB")
    print("-" * 60)
    print()


# ── main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: small subsets only")
    parser.add_argument("--make-tiny", action="store_true",
                        help="Extract ~50 sample subsets for pipeline smoke tests")
    parser.add_argument("--make-dev", action="store_true",
                        help="Extract ~500-2000 representative dev subsets for HP tuning")
    parser.add_argument("--make-ablation", action="store_true",
                        help="Extract 5%% of total data as fixed-seed subsets for ablation experiments")
    parser.add_argument("--make-all-subsets", action="store_true",
                        help="Generate all subsets (tiny, dev, ablation) at once")
    args = parser.parse_args()

    # ── subset generation modes (extract from existing data and exit) ──
    subset_mode = args.make_tiny or args.make_dev or args.make_ablation or args.make_all_subsets
    if subset_mode:
        print("=" * 60)
        print("Deep Compressor — Subset Generation")
        print(f"Project root: {PROJECT_ROOT}")
        print("=" * 60)
        print()
        if args.make_all_subsets:
            make_all_subsets()
        else:
            if args.make_tiny:
                make_tiny_subset()
            if args.make_dev:
                make_dev_subset()
            if args.make_ablation:
                make_ablation_subset()
        print_summary()
        return

    mode = "TEST (small subsets)" if args.test else "FULL (~1.7B tokens target)"
    print("=" * 60)
    print(f"Deep Compressor — Data Preparation [{mode}]")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)
    print()

    prepare_ntp_data(test_mode=args.test)
    prepare_qa_data(test_mode=args.test)

    print_summary()


if __name__ == "__main__":
    main()

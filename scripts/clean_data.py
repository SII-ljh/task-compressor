#!/usr/bin/env python3
"""Clean QA datasets by removing trivial / non-informative samples.

Filters applied:
  1. Remove "No answer" (and variants) — model can guess without reading context
  2. Remove empty or whitespace-only answers
  3. Remove duplicate (context, question, answer) triples

Overwrites the original files so existing training scripts work as-is.
A backup is saved alongside each file (e.g. qa_large_train.json.bak).

Usage:
  python scripts/clean_data.py                          # clean default files
  python scripts/clean_data.py --files data/qa_dev.json # clean specific files
  python scripts/clean_data.py --dry-run                # preview without writing
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Default files to clean
DEFAULT_FILES = [
    DATA_DIR / "qa_large_train.json",
    DATA_DIR / "qa_large_dev.json",
    DATA_DIR / "qa_train.json",
    DATA_DIR / "qa_dev.json",
]

# Answers that indicate unanswerable questions (case-insensitive)
NO_ANSWER_PATTERNS = {
    "no answer",
    "no answer.",
    "n/a",
    "unanswerable",
    "not answerable",
    "impossible to answer",
    "cannot be answered",
}


def is_trivial_answer(answer: str) -> str | None:
    """Return the reason if an answer should be filtered, else None."""
    stripped = answer.strip()
    if not stripped:
        return "empty"
    if stripped.lower() in NO_ANSWER_PATTERNS:
        return "no_answer"
    return None


def clean_file(filepath: Path, dry_run: bool = False) -> dict:
    """Clean a single QA JSON file. Returns stats dict."""
    if not filepath.exists():
        return {"skipped": True, "reason": "not found"}

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)
    reasons = {}
    seen = set()
    cleaned = []

    for item in data:
        answer = item.get("answer", "")

        # Filter trivial answers
        reason = is_trivial_answer(answer)
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        # Deduplicate
        key = (item.get("context", ""), item.get("question", ""), answer)
        if key in seen:
            reasons["duplicate"] = reasons.get("duplicate", 0) + 1
            continue
        seen.add(key)

        cleaned.append(item)

    stats = {
        "file": filepath.name,
        "original": original_count,
        "cleaned": len(cleaned),
        "removed": original_count - len(cleaned),
        "reasons": reasons,
    }

    if not dry_run:
        # Backup original
        backup_path = filepath.with_suffix(filepath.suffix + ".bak")
        filepath.rename(backup_path)

        # Write cleaned data
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False)

        stats["backup"] = str(backup_path)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean QA datasets")
    parser.add_argument(
        "--files", nargs="+", type=str,
        help="Specific files to clean (default: qa_large_*.json and qa_*.json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without modifying files",
    )
    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = DEFAULT_FILES

    print("=" * 60)
    print(f"  QA Data Cleaning {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    any_cleaned = False
    for filepath in files:
        print(f"\n  {filepath} ...")
        stats = clean_file(filepath, dry_run=args.dry_run)

        if stats.get("skipped"):
            print(f"    SKIPPED: {stats['reason']}")
            continue

        any_cleaned = True
        pct = 100 * stats["removed"] / max(stats["original"], 1)
        print(f"    {stats['original']:,} → {stats['cleaned']:,}  "
              f"(removed {stats['removed']:,}, {pct:.1f}%)")
        for reason, count in sorted(stats["reasons"].items()):
            print(f"      - {reason}: {count:,}")
        if "backup" in stats:
            print(f"    backup: {stats['backup']}")

    if not any_cleaned:
        print("\n  No files to clean.")
        sys.exit(1)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  Dry run complete. No files modified.")
    else:
        print("  Done. Original files backed up as *.bak")
    print("=" * 60)


if __name__ == "__main__":
    main()

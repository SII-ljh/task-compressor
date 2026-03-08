"""Tests for the data loading and collation pipeline."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from task_compressor.data import NTPCollator, NTPDataset, QACollator, QADataset


class FakeTokenizer:
    """Minimal tokenizer for testing without loading a real model."""

    def __init__(self, vocab_size=256, pad_token_id=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = 1

    def __call__(self, text, truncation=True, max_length=512, add_special_tokens=False, **kwargs):
        # Simple char-level tokenization for testing
        ids = [ord(c) % self.vocab_size for c in text[:max_length]]
        if not ids:
            ids = [0]
        return {"input_ids": ids}


@pytest.fixture
def sample_data_file():
    """Create a temporary QA data file."""
    data = [
        {
            "context": "The quick brown fox jumps over the lazy dog.",
            "question": "What animal jumps?",
            "answer": "fox",
            "source": "test",
        },
        {
            "context": "Paris is the capital of France. It is known for the Eiffel Tower.",
            "question": "What is the capital of France?",
            "answer": "Paris",
            "source": "test",
        },
        {
            "context": "Python is a programming language.",
            "question": "What is Python?",
            "answer": "a programming language",
            "source": "test",
        },
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False)
        return f.name


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


class TestQADataset:
    def test_length(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        assert len(ds) == 3

    def test_item_keys(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        item = ds[0]
        assert "context_ids" in item
        assert "prompt_ids" in item
        assert "response_ids" in item

    def test_item_types(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        item = ds[0]
        assert isinstance(item["context_ids"], torch.Tensor)
        assert item["context_ids"].dtype == torch.long

    def test_truncation(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, max_context_length=10,
                       max_prompt_length=5, max_response_length=3)
        item = ds[0]
        assert len(item["context_ids"]) <= 10
        assert len(item["prompt_ids"]) <= 5
        assert len(item["response_ids"]) <= 3


class TestQACollator:
    def test_batch_shapes(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        collator = QACollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1], ds[2]])

        assert batch["context_ids"].shape[0] == 3
        assert batch["prompt_ids"].shape[0] == 3
        assert batch["response_ids"].shape[0] == 3

    def test_masks_match_ids(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        collator = QACollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1]])

        for key in ["context", "prompt", "response"]:
            ids = batch[f"{key}_ids"]
            mask = batch[f"{key}_mask"]
            assert ids.shape == mask.shape
            # Mask should be 1 where ids != pad, 0 where ids == pad
            expected_mask = (ids != tokenizer.pad_token_id).long()
            assert torch.equal(mask, expected_mask)

    def test_teacher_input_built(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        collator = QACollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1]])

        assert "teacher_input_ids" in batch
        assert "teacher_attention_mask" in batch
        assert "response_starts" in batch
        assert "response_lens" in batch

        # Teacher sequence should be context + prompt + response concatenated
        assert batch["teacher_input_ids"].shape[0] == 2
        assert batch["response_starts"].shape == (2,)
        assert batch["response_lens"].shape == (2,)

    def test_response_starts_correct(self, sample_data_file, tokenizer):
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        collator = QACollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0]])

        # response_start should equal len(context) + len(prompt)
        item = ds[0]
        expected_start = len(item["context_ids"]) + len(item["prompt_ids"])
        assert batch["response_starts"][0].item() == expected_start

    def test_padding_correct(self, sample_data_file, tokenizer):
        """Shorter sequences should be padded to match the longest."""
        ds = QADataset(sample_data_file, tokenizer, 128, 64, 32)
        collator = QACollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1], ds[2]])

        # All context_ids should have the same length (max in batch)
        assert batch["context_ids"].shape[1] > 0
        # Check that padding values exist in shorter sequences
        ctx_lens = [len(ds[i]["context_ids"]) for i in range(3)]
        max_len = max(ctx_lens)
        assert batch["context_ids"].shape[1] == max_len


# ── NTP Dataset / Collator Tests ─────────────────────────────────────────


@pytest.fixture
def ntp_data_file():
    """Create a temporary NTP JSONL data file."""
    lines = [
        json.dumps({"text": "The quick brown fox jumps over the lazy dog. " * 5}),
        json.dumps({"text": "Paris is the capital of France. It is known for art. " * 4}),
        json.dumps({"text": "Python is a programming language used widely. " * 6}),
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        for line in lines:
            f.write(line + "\n")
        return f.name


class TestNTPDataset:
    def test_length(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        assert len(ds) == 3

    def test_item_keys(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        item = ds[0]
        assert "doc_ids" in item
        assert "segment_ids" in item
        assert "segment_labels" in item

    def test_item_types(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        item = ds[0]
        assert isinstance(item["doc_ids"], torch.Tensor)
        assert item["doc_ids"].dtype == torch.long
        assert isinstance(item["segment_ids"], torch.Tensor)

    def test_doc_segment_split(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        item = ds[0]
        assert len(item["doc_ids"]) >= 1
        assert len(item["segment_ids"]) >= 1

    def test_segment_labels_match(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        item = ds[0]
        assert torch.equal(item["segment_ids"], item["segment_labels"])

    def test_segment_length_bounded(self, ntp_data_file, tokenizer):
        seg_len = 16
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=seg_len)
        item = ds[0]
        assert len(item["segment_ids"]) <= seg_len

    def test_lazy_loading_reopens_handle(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        # Access items sequentially (simulates lazy loading)
        for i in range(len(ds)):
            item = ds[i]
            assert "doc_ids" in item


class TestNTPCollator:
    def test_batch_shapes(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        collator = NTPCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1], ds[2]])

        assert batch["doc_ids"].shape[0] == 3
        assert batch["segment_ids"].shape[0] == 3
        assert batch["segment_labels"].shape[0] == 3

    def test_masks_match_ids(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        collator = NTPCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1]])

        for key in ["doc", "segment"]:
            ids = batch[f"{key}_ids"]
            mask = batch[f"{key}_mask"]
            assert ids.shape == mask.shape
            expected_mask = (ids != tokenizer.pad_token_id).long()
            assert torch.equal(mask, expected_mask)

    def test_segment_labels_padding(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        collator = NTPCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0], ds[1]])

        labels = batch["segment_labels"]
        mask = batch["segment_mask"]
        # Padding positions in labels should be -100
        pad_positions = mask == 0
        if pad_positions.any():
            assert (labels[pad_positions] == -100).all()

    def test_all_fields_present(self, ntp_data_file, tokenizer):
        ds = NTPDataset(ntp_data_file, tokenizer, max_context_length=128, ntp_segment_len=32)
        collator = NTPCollator(pad_token_id=tokenizer.pad_token_id)
        batch = collator([ds[0]])

        expected_keys = {"doc_ids", "doc_mask", "segment_ids", "segment_mask", "segment_labels"}
        assert expected_keys == set(batch.keys())

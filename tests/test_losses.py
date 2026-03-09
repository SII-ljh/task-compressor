"""Tests for loss functions."""

import pytest
import torch

from task_compressor.losses import compute_qa_loss


class TestQALoss:
    def test_basic(self):
        B, L, V = 2, 10, 100
        logits = torch.randn(B, L, V)
        labels = torch.randint(0, V, (B, L))
        loss = compute_qa_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_with_ignore_index(self):
        B, L, V = 2, 10, 100
        logits = torch.randn(B, L, V)
        labels = torch.full((B, L), -100, dtype=torch.long)
        labels[:, 5:] = torch.randint(0, V, (B, 5))
        loss = compute_qa_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_all_ignored_returns_nan(self):
        """When all labels are -100, CE returns NaN (no valid targets)."""
        B, L, V = 2, 10, 100
        logits = torch.randn(B, L, V)
        labels = torch.full((B, L), -100, dtype=torch.long)
        loss = compute_qa_loss(logits, labels)
        assert torch.isnan(loss)

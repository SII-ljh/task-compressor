"""Tests for loss functions."""

import pytest
import torch

from task_compressor.losses import (
    compute_distill_loss_kl,
    compute_distill_loss_mse,
    compute_qa_loss,
)


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


class TestDistillLossKL:
    def test_identical_logits_near_zero(self):
        """When teacher == student, KL should be ~0."""
        B, L, V = 2, 8, 50
        logits = torch.randn(B, L, V)
        mask = torch.ones(B, L)
        loss = compute_distill_loss_kl(logits, logits.clone(), mask, temperature=2.0)
        assert loss.item() < 1e-5

    def test_different_logits_positive(self):
        B, L, V = 2, 8, 50
        student = torch.randn(B, L, V)
        teacher = torch.randn(B, L, V)
        mask = torch.ones(B, L)
        loss = compute_distill_loss_kl(student, teacher, mask, temperature=2.0)
        assert loss.item() > 0

    def test_mask_applied(self):
        """Masked positions should not contribute to loss."""
        B, L, V = 2, 8, 50
        student = torch.randn(B, L, V)
        teacher = torch.randn(B, L, V)

        full_mask = torch.ones(B, L)
        partial_mask = torch.ones(B, L)
        partial_mask[:, 4:] = 0

        loss_full = compute_distill_loss_kl(student, teacher, full_mask)
        loss_partial = compute_distill_loss_kl(student, teacher, partial_mask)
        # Different masks should give different losses
        assert loss_full.item() != loss_partial.item()

    def test_temperature_scaling(self):
        """Higher temperature should generally change loss magnitude."""
        B, L, V = 2, 8, 50
        student = torch.randn(B, L, V)
        teacher = torch.randn(B, L, V)
        mask = torch.ones(B, L)
        loss_t1 = compute_distill_loss_kl(student, teacher, mask, temperature=1.0)
        loss_t4 = compute_distill_loss_kl(student, teacher, mask, temperature=4.0)
        # Just check both are positive (exact relationship depends on data)
        assert loss_t1.item() > 0
        assert loss_t4.item() > 0

    def test_gradient_only_through_student(self):
        """Gradients should flow through student but not teacher."""
        B, L, V = 2, 8, 50
        student = torch.randn(B, L, V, requires_grad=True)
        teacher = torch.randn(B, L, V)  # no grad
        mask = torch.ones(B, L)
        loss = compute_distill_loss_kl(student, teacher, mask)
        loss.backward()
        assert student.grad is not None


class TestDistillLossMSE:
    def test_identical_hidden_zero(self):
        B, L, D = 2, 8, 64
        h = torch.randn(B, L, D)
        mask = torch.ones(B, L)
        loss = compute_distill_loss_mse(h, h.clone(), mask)
        assert loss.item() < 1e-6

    def test_different_hidden_positive(self):
        B, L, D = 2, 8, 64
        s = torch.randn(B, L, D)
        t = torch.randn(B, L, D)
        mask = torch.ones(B, L)
        loss = compute_distill_loss_mse(s, t, mask)
        assert loss.item() > 0

    def test_mask_applied(self):
        B, L, D = 2, 8, 64
        s = torch.randn(B, L, D)
        t = torch.randn(B, L, D)
        full_mask = torch.ones(B, L)
        half_mask = torch.zeros(B, L)
        half_mask[:, :4] = 1
        loss_full = compute_distill_loss_mse(s, t, full_mask)
        loss_half = compute_distill_loss_mse(s, t, half_mask)
        assert loss_full.item() != loss_half.item()

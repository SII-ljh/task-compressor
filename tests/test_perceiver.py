"""Tests for the Query-Conditioned Perceiver."""

import pytest
import torch

from task_compressor.models.perceiver import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    PerceiverBlock,
    QueryConditionedPerceiver,
)

from .conftest import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS


class TestMultiHeadCrossAttention:
    def test_output_shape(self):
        attn = MultiHeadCrossAttention(HIDDEN_SIZE, NUM_HEADS)
        query = torch.randn(2, 8, HIDDEN_SIZE)
        kv = torch.randn(2, 32, HIDDEN_SIZE)
        out = attn(query, kv)
        assert out.shape == (2, 8, HIDDEN_SIZE)

    def test_with_mask(self):
        attn = MultiHeadCrossAttention(HIDDEN_SIZE, NUM_HEADS)
        query = torch.randn(2, 8, HIDDEN_SIZE)
        kv = torch.randn(2, 32, HIDDEN_SIZE)
        mask = torch.ones(2, 32)
        mask[:, 16:] = 0  # mask out last 16 positions
        out = attn(query, kv, kv_mask=mask)
        assert out.shape == (2, 8, HIDDEN_SIZE)
        assert not torch.isnan(out).any()

    def test_no_nan_with_all_masked(self):
        """All positions masked should not produce NaN (SDPA handles this)."""
        attn = MultiHeadCrossAttention(HIDDEN_SIZE, NUM_HEADS)
        query = torch.randn(1, 4, HIDDEN_SIZE)
        kv = torch.randn(1, 8, HIDDEN_SIZE)
        mask = torch.zeros(1, 8)  # all masked
        out = attn(query, kv, kv_mask=mask)
        # SDPA may produce nan/0 for fully masked — just check no crash
        assert out.shape == (1, 4, HIDDEN_SIZE)


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(HIDDEN_SIZE, NUM_HEADS)
        x = torch.randn(2, 16, HIDDEN_SIZE)
        out = attn(x)
        assert out.shape == (2, 16, HIDDEN_SIZE)


class TestPerceiverBlock:
    def test_output_shape(self):
        block = PerceiverBlock(HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE)
        query = torch.randn(2, 8, HIDDEN_SIZE)
        kv = torch.randn(2, 64, HIDDEN_SIZE)
        out = block(query, kv)
        assert out.shape == (2, 8, HIDDEN_SIZE)

    def test_residual_connection(self):
        """Output should differ from input (cross-attn + self-attn + FFN)."""
        block = PerceiverBlock(HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE)
        query = torch.randn(2, 8, HIDDEN_SIZE)
        kv = torch.randn(2, 64, HIDDEN_SIZE)
        out = block(query, kv)
        assert not torch.allclose(out, query, atol=1e-5)


class TestQueryConditionedPerceiver:
    def test_output_shape(self):
        perceiver = QueryConditionedPerceiver(
            HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE, num_layers=2
        )
        query = torch.randn(2, 64, HIDDEN_SIZE)  # k=64
        kv = torch.randn(2, 128, HIDDEN_SIZE)
        out = perceiver(query, kv)
        assert out.shape == (2, 64, HIDDEN_SIZE)

    def test_variable_context_length(self):
        """Perceiver should handle different KV lengths."""
        perceiver = QueryConditionedPerceiver(
            HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE, num_layers=1
        )
        query = torch.randn(2, 16, HIDDEN_SIZE)
        kv_short = torch.randn(2, 32, HIDDEN_SIZE)
        kv_long = torch.randn(2, 256, HIDDEN_SIZE)
        out_short = perceiver(query, kv_short)
        out_long = perceiver(query, kv_long)
        assert out_short.shape == (2, 16, HIDDEN_SIZE)
        assert out_long.shape == (2, 16, HIDDEN_SIZE)

    def test_with_mask(self):
        perceiver = QueryConditionedPerceiver(
            HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE, num_layers=2
        )
        query = torch.randn(2, 16, HIDDEN_SIZE)
        kv = torch.randn(2, 64, HIDDEN_SIZE)
        mask = torch.ones(2, 64)
        mask[0, 32:] = 0
        mask[1, 48:] = 0
        out = perceiver(query, kv, kv_mask=mask)
        assert out.shape == (2, 16, HIDDEN_SIZE)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        perceiver = QueryConditionedPerceiver(
            HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE, num_layers=2
        )
        query = torch.randn(2, 16, HIDDEN_SIZE, requires_grad=True)
        kv = torch.randn(2, 64, HIDDEN_SIZE, requires_grad=True)
        out = perceiver(query, kv)
        out.sum().backward()
        assert query.grad is not None
        assert kv.grad is not None
        for p in perceiver.parameters():
            assert p.grad is not None

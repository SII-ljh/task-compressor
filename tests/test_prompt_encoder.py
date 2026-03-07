"""Tests for the Prompt Encoder."""

import pytest
import torch
import torch.nn as nn

from task_compressor.models.prompt_encoder import PromptEncoder

from .conftest import HIDDEN_SIZE, NUM_HEADS


@pytest.fixture
def fake_embedding():
    return nn.Embedding(256, HIDDEN_SIZE)


class TestPromptEncoder:
    def test_output_shape(self, fake_embedding):
        n_p = 16
        encoder = PromptEncoder(fake_embedding, HIDDEN_SIZE, NUM_HEADS, n_p)
        prompt_ids = torch.randint(0, 256, (2, 20))
        prompt_mask = torch.ones(2, 20)
        out = encoder(prompt_ids, prompt_mask)
        assert out.shape == (2, n_p, HIDDEN_SIZE)

    def test_different_n_p(self, fake_embedding):
        for n_p in [4, 8, 16, 32]:
            encoder = PromptEncoder(fake_embedding, HIDDEN_SIZE, NUM_HEADS, n_p)
            prompt_ids = torch.randint(0, 256, (1, 10))
            prompt_mask = torch.ones(1, 10)
            out = encoder(prompt_ids, prompt_mask)
            assert out.shape == (1, n_p, HIDDEN_SIZE)

    def test_frozen_embedding_no_grad(self, fake_embedding):
        """Embedding weights should not receive gradients."""
        encoder = PromptEncoder(fake_embedding, HIDDEN_SIZE, NUM_HEADS, 16)
        prompt_ids = torch.randint(0, 256, (2, 10))
        prompt_mask = torch.ones(2, 10)
        out = encoder(prompt_ids, prompt_mask)
        out.sum().backward()
        # Embedding should have no grad (used inside torch.no_grad())
        assert fake_embedding.weight.grad is None

    def test_latent_tokens_have_grad(self, fake_embedding):
        encoder = PromptEncoder(fake_embedding, HIDDEN_SIZE, NUM_HEADS, 16)
        prompt_ids = torch.randint(0, 256, (2, 10))
        prompt_mask = torch.ones(2, 10)
        out = encoder(prompt_ids, prompt_mask)
        out.sum().backward()
        assert encoder.latent_tokens.grad is not None

    def test_variable_prompt_length(self, fake_embedding):
        encoder = PromptEncoder(fake_embedding, HIDDEN_SIZE, NUM_HEADS, 16)
        # Short prompt
        out_short = encoder(
            torch.randint(0, 256, (1, 5)),
            torch.ones(1, 5),
        )
        # Long prompt
        out_long = encoder(
            torch.randint(0, 256, (1, 100)),
            torch.ones(1, 100),
        )
        assert out_short.shape == out_long.shape == (1, 16, HIDDEN_SIZE)

    def test_with_padding_mask(self, fake_embedding):
        encoder = PromptEncoder(fake_embedding, HIDDEN_SIZE, NUM_HEADS, 16)
        prompt_ids = torch.randint(0, 256, (2, 20))
        prompt_mask = torch.ones(2, 20)
        prompt_mask[0, 10:] = 0  # first sample has only 10 real tokens
        out = encoder(prompt_ids, prompt_mask)
        assert out.shape == (2, 16, HIDDEN_SIZE)
        assert not torch.isnan(out).any()

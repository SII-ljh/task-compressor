"""Shared test fixtures for Task Compressor.

All tests use small random models to avoid downloading real weights.
"""

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# Small model dimensions for fast testing
HIDDEN_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2
INTERMEDIATE_SIZE = 128
VOCAB_SIZE = 256
MAX_SEQ_LEN = 32


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dtype():
    return torch.float32


@pytest.fixture
def small_config():
    """Config matching a tiny Qwen-like model."""
    return AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_SEQ_LEN * 4,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )


@pytest.fixture
def small_model(small_config):
    """Tiny randomly-initialized Qwen model for testing."""
    model = AutoModelForCausalLM.from_config(small_config).float()
    model.eval()
    return model


@pytest.fixture
def embedding_layer(small_model):
    """Embedding layer from the small model."""
    return small_model.get_input_embeddings()

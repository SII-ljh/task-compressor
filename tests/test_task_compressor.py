"""Tests for the full TaskCompressorModel.

Uses a tiny randomly-initialized model to avoid downloading real weights.
"""

import pytest
import torch

from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from task_compressor.config import ModelConfig
from task_compressor.models.task_compressor_model import TaskCompressorModel

from .conftest import HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_HEADS, NUM_LAYERS, VOCAB_SIZE


def _make_tiny_config() -> ModelConfig:
    """Create a ModelConfig that uses a tiny random model."""
    return ModelConfig(
        base_model="__tiny__",  # sentinel, we'll override the model
        lora_rank=8,
        lora_alpha=8,
        lora_target_modules=["q_proj", "v_proj"],
        n_prompt_tokens=4,
        n_context_tokens=4,
        num_perceiver_layers=1,
        perceiver_ffn_mult=2,
        gradient_checkpointing=False,
    )


def _build_tiny_model() -> TaskCompressorModel:
    """Build a TaskCompressorModel with tiny random weights."""
    config = _make_tiny_config()

    # Create tiny Qwen model directly from config
    model_config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-0.6B",
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        num_hidden_layers=NUM_LAYERS,
        intermediate_size=INTERMEDIATE_SIZE,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=512,
        torch_dtype=torch.float32,
    )
    base_model = AutoModelForCausalLM.from_config(model_config).float()

    # Build TaskCompressorModel manually (bypass from_pretrained)
    model = TaskCompressorModel.__new__(TaskCompressorModel)
    torch.nn.Module.__init__(model)
    model.config = config
    model._torch_dtype = torch.float32

    # Freeze base model
    for p in base_model.parameters():
        p.requires_grad = False

    # Add LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.base_model = get_peft_model(base_model, lora_config)

    hidden_size = HIDDEN_SIZE
    num_heads = NUM_HEADS
    intermediate_size = hidden_size * config.perceiver_ffn_mult

    from task_compressor.models.perceiver import QueryConditionedPerceiver
    from task_compressor.models.prompt_encoder import PromptEncoder

    model.prompt_encoder = PromptEncoder(
        model.base_model.get_input_embeddings(),
        hidden_size, num_heads, config.n_prompt_tokens,
    )
    model.context_tokens = torch.nn.Parameter(
        torch.randn(config.n_context_tokens, hidden_size) * 0.02
    )
    model.perceiver = QueryConditionedPerceiver(
        hidden_size, num_heads, intermediate_size, config.num_perceiver_layers,
    )
    model.separator = torch.nn.Parameter(
        torch.randn(1, hidden_size) * 0.02
    )

    return model


class TestTaskCompressorModel:
    @pytest.fixture
    def model(self):
        return _build_tiny_model()

    def test_encode_output_shape(self, model):
        ctx_ids = torch.randint(0, VOCAB_SIZE, (2, 16))
        ctx_mask = torch.ones(2, 16, dtype=torch.long)
        hidden = model.encode(ctx_ids, ctx_mask)
        assert hidden.shape == (2, 16, HIDDEN_SIZE)

    def test_compress_output_shape(self, model):
        k = model.config.n_prompt_tokens + model.config.n_context_tokens
        encoder_hidden = torch.randn(2, 32, HIDDEN_SIZE)
        encoder_mask = torch.ones(2, 32)
        prompt_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
        prompt_mask = torch.ones(2, 8, dtype=torch.long)
        compressed = model.compress(
            encoder_hidden, encoder_mask, prompt_ids, prompt_mask
        )
        assert compressed.shape == (2, k, HIDDEN_SIZE)

    def test_forward_returns_losses(self, model):
        B = 2
        ctx_ids = torch.randint(0, VOCAB_SIZE, (B, 16))
        ctx_mask = torch.ones(B, 16, dtype=torch.long)
        prompt_ids = torch.randint(0, VOCAB_SIZE, (B, 8))
        prompt_mask = torch.ones(B, 8, dtype=torch.long)
        resp_ids = torch.randint(0, VOCAB_SIZE, (B, 6))
        resp_mask = torch.ones(B, 6, dtype=torch.long)

        outputs = model(
            context_ids=ctx_ids,
            context_mask=ctx_mask,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            response_ids=resp_ids,
            response_mask=resp_mask,
            distill_alpha=0.0,
        )
        assert "loss" in outputs
        assert "qa_loss" in outputs
        assert outputs["loss"].item() > 0

    def test_forward_with_distillation(self, model):
        B = 2
        ctx_ids = torch.randint(0, VOCAB_SIZE, (B, 16))
        ctx_mask = torch.ones(B, 16, dtype=torch.long)
        prompt_ids = torch.randint(0, VOCAB_SIZE, (B, 8))
        prompt_mask = torch.ones(B, 8, dtype=torch.long)
        resp_ids = torch.randint(0, VOCAB_SIZE, (B, 6))
        resp_mask = torch.ones(B, 6, dtype=torch.long)

        # Build teacher input
        teacher_ids = torch.cat([ctx_ids, prompt_ids, resp_ids], dim=1)
        teacher_mask = torch.ones(B, teacher_ids.shape[1], dtype=torch.long)
        resp_starts = torch.tensor([16 + 8, 16 + 8], dtype=torch.long)
        resp_lens = torch.tensor([6, 6], dtype=torch.long)

        outputs = model(
            context_ids=ctx_ids,
            context_mask=ctx_mask,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            response_ids=resp_ids,
            response_mask=resp_mask,
            teacher_input_ids=teacher_ids,
            teacher_attention_mask=teacher_mask,
            response_starts=resp_starts,
            response_lens=resp_lens,
            distill_alpha=0.5,
            distill_temperature=2.0,
        )
        assert outputs["distill_loss"].item() >= 0

    def test_backward_updates_lora_and_perceiver(self, model):
        B = 2
        ctx_ids = torch.randint(0, VOCAB_SIZE, (B, 16))
        ctx_mask = torch.ones(B, 16, dtype=torch.long)
        prompt_ids = torch.randint(0, VOCAB_SIZE, (B, 8))
        prompt_mask = torch.ones(B, 8, dtype=torch.long)
        resp_ids = torch.randint(0, VOCAB_SIZE, (B, 6))
        resp_mask = torch.ones(B, 6, dtype=torch.long)

        outputs = model(
            context_ids=ctx_ids, context_mask=ctx_mask,
            prompt_ids=prompt_ids, prompt_mask=prompt_mask,
            response_ids=resp_ids, response_mask=resp_mask,
            distill_alpha=0.0,
        )
        outputs["loss"].backward()

        # Check LoRA parameters have gradients
        lora_has_grad = False
        for name, p in model.named_parameters():
            if "lora" in name.lower() and p.requires_grad:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    lora_has_grad = True
                    break
        assert lora_has_grad, "LoRA parameters should have gradients"

        # Check perceiver parameters have gradients
        perceiver_has_grad = False
        for name, p in model.named_parameters():
            if "perceiver" in name and p.requires_grad:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    perceiver_has_grad = True
                    break
        assert perceiver_has_grad, "Perceiver parameters should have gradients"

    def test_encoder_decoder_weight_sharing(self, model):
        """Encoder and decoder should share the same base weights."""
        # Get embedding layer (shared)
        embed = model.get_embedding_layer()
        weight_before = embed.weight.data.clone()

        # Encode (enables adapter)
        ctx_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        ctx_mask = torch.ones(1, 8, dtype=torch.long)
        model.encode(ctx_ids, ctx_mask)

        # The embedding should be the same object
        assert embed is model.get_embedding_layer()
        assert torch.equal(weight_before, embed.weight.data)

    def test_generate_produces_tokens(self, model):
        k = model.config.n_prompt_tokens + model.config.n_context_tokens
        compressed = torch.randn(1, k, HIDDEN_SIZE)
        prompt_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        prompt_mask = torch.ones(1, 8, dtype=torch.long)

        generated = model.generate(
            compressed, prompt_ids, prompt_mask,
            max_new_tokens=5, temperature=0,
        )
        assert generated.shape[0] == 1
        assert generated.shape[1] > 0

    def test_lora_parameter_count(self, model):
        """LoRA should add a small fraction of total parameters."""
        trainable = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.base_model.parameters())
        ratio = trainable / total
        # LoRA should be a small fraction (< 10% for rank=8 on tiny model)
        assert ratio < 0.15, f"LoRA param ratio too high: {ratio:.2%}"

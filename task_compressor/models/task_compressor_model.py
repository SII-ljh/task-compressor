"""Task Compressor — full model assembling encoder, perceiver, and decoder.

Architecture overview:
  1. LoRA Encoder:  Qwen + LoRA → context hidden states
  2. Prompt Encoder: learnable latents attend to frozen prompt embeddings
  3. Perceiver:     query (prompt latents + context tokens) × KV (encoder hidden)
                    → k soft prompt vectors
  4. Decoder:       frozen Qwen (LoRA off) takes [soft prompts | sep | prompt] → logits

Encoder and Decoder share the same Qwen weights; LoRA adapters are toggled
via peft's enable/disable_adapter_layers().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from ..config import ModelConfig
from .perceiver import QueryConditionedPerceiver
from .prompt_encoder import PromptEncoder


class TaskCompressorModel(nn.Module):
    """Full Task Compressor model."""

    def __init__(self, config: ModelConfig, torch_dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self._torch_dtype = torch_dtype

        # ── 1. Load base Qwen model ──────────────────────────────────────
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch_dtype,
        )
        # Freeze all base model parameters first
        for p in self.base_model.parameters():
            p.requires_grad = False

        # ── 2. Add LoRA adapters (skip if rank=0 → frozen encoder) ──────
        self._has_lora = config.lora_rank > 0
        if self._has_lora:
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.base_model = get_peft_model(self.base_model, lora_config)

        # ── Read model dimensions from config ────────────────────────────
        model_cfg = self.base_model.config
        hidden_size = model_cfg.hidden_size
        num_heads = model_cfg.num_attention_heads
        if config.perceiver_ffn_mult > 0:
            intermediate_size = hidden_size * config.perceiver_ffn_mult
        else:
            intermediate_size = model_cfg.intermediate_size

        # ── 3. Prompt Encoder ────────────────────────────────────────────
        self.prompt_encoder = PromptEncoder(
            embedding_layer=self.get_embedding_layer(),
            hidden_size=hidden_size,
            num_heads=num_heads,
            n_prompt_tokens=config.n_prompt_tokens,
        ).to(torch_dtype)

        # ── 4. Learnable context tokens ──────────────────────────────────
        self.context_tokens = nn.Parameter(
            torch.randn(config.n_context_tokens, hidden_size, dtype=torch_dtype)
            * 0.02
        )

        # ── 5. Perceiver ────────────────────────────────────────────────
        self.perceiver = QueryConditionedPerceiver(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=config.num_perceiver_layers,
        ).to(torch_dtype)

        # ── 5b. Match Perceiver output scale to Qwen embedding scale ───
        # final_ln outputs unit-variance (~1.0) by default, but the Qwen
        # decoder expects input embeddings at std≈0.03.  A ~34x scale
        # mismatch causes the decoder to produce extreme logits → large
        # gradients → bf16 overflow → NaN.  Fix: set final_ln gamma to
        # the embedding std so the Perceiver output starts at the right
        # scale.  The gamma is learnable and will adapt during training.
        with torch.no_grad():
            embed_std = (
                self.get_embedding_layer()
                .weight.float()
                .std()
                .item()
            )
        nn.init.constant_(self.perceiver.final_ln.weight, embed_std)

        # ── 6. Separator token ───────────────────────────────────────────
        self.separator = nn.Parameter(
            torch.randn(1, hidden_size, dtype=torch_dtype) * 0.02
        )

        # ── 7. Upcast trainable params to fp32 for optimizer stability ──
        # AdamW stores exp_avg / exp_avg_sq in the parameter's dtype.
        # With bf16 params these running averages lose precision after
        # ~300 steps, causing divergent updates → NaN gradients.
        # Keeping trainable params fp32 gives fp32 optimizer states while
        # autocast(bf16) still handles the forward pass in bf16.
        if torch_dtype != torch.float32:
            for p in self.parameters():
                if p.requires_grad:
                    p.data = p.data.float()

        # ── Gradient checkpointing ───────────────────────────────────────
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_inner_model(self):
        """Get the underlying CausalLM, unwrapping PEFT wrapper if present."""
        if self._has_lora:
            return self.base_model.get_base_model()
        return self.base_model

    def get_embedding_layer(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

    @property
    def hidden_size(self) -> int:
        return self.base_model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.base_model.config.vocab_size

    # ── Core methods ─────────────────────────────────────────────────────

    def encode(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode context through Qwen (LoRA-augmented if rank > 0, frozen if rank = 0).

        Returns:
            (B, L_c, D) last hidden states.
        """
        if self._has_lora:
            self.base_model.enable_adapter_layers()
        # Call the inner transformer (QwenModel) directly to get
        # last_hidden_state.  Going through the CausalLM wrapper requires
        # output_hidden_states=True which forces ALL intermediate layer
        # outputs to persist in memory — defeating gradient checkpointing
        # and wasting ~7 GB for a 0.6B model (B=32, L=4096).
        causal_lm = self._get_inner_model()
        outputs = causal_lm.model(
            input_ids=context_ids,
            attention_mask=context_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def compress(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compress encoder hidden states conditioned on the prompt.

        Returns:
            (B, k, D) compressed soft prompts.
        """
        B = encoder_hidden.shape[0]

        # Prompt latents: (B, n_p, D)
        prompt_latents = self.prompt_encoder(prompt_ids, prompt_mask)

        # Learnable context tokens: (B, n_c, D)
        ctx_tokens = self.context_tokens.unsqueeze(0).expand(B, -1, -1)

        # Concatenate query: (B, k, D)  where k = n_p + n_c
        query = torch.cat([prompt_latents, ctx_tokens], dim=1)

        # Perceiver: (B, k, D)
        return self.perceiver(query, encoder_hidden, encoder_mask)

    def _disable_adapter_forward_only(self):
        """Disable LoRA forward contribution without changing requires_grad.

        PEFT's ``disable_adapter_layers()`` also sets ``requires_grad=False``
        on LoRA params, which blocks gradient flow during backward.  Here we
        set ``_disable_adapters`` directly (the private backing field) to skip
        the property setter and the ``enable_adapters()`` method.
        """
        for module in self.base_model.modules():
            if hasattr(module, "_disable_adapters"):
                module._disable_adapters = True

    def _enable_adapter_forward_only(self):
        """Re-enable LoRA forward contribution without changing requires_grad."""
        for module in self.base_model.modules():
            if hasattr(module, "_disable_adapters"):
                module._disable_adapters = False

    @property
    def _gradient_checkpointing_enabled(self) -> bool:
        """Check if gradient checkpointing is currently enabled."""
        return getattr(self.base_model, "is_gradient_checkpointing", False)

    def _pause_gradient_checkpointing(self):
        """Temporarily disable gradient checkpointing, return whether it was on."""
        was_enabled = self._gradient_checkpointing_enabled
        if was_enabled:
            self.base_model.gradient_checkpointing_disable()
        return was_enabled

    def _resume_gradient_checkpointing(self, was_enabled: bool):
        """Re-enable gradient checkpointing if it was on before."""
        if was_enabled:
            # gradient_checkpointing_disable() sets config.use_cache=True;
            # reset before re-enabling to suppress spurious warning
            if hasattr(self.base_model, "config"):
                self.base_model.config.use_cache = False
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def decode_train(
        self,
        compressed: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict:
        """Student decode with teacher-forcing. Returns logits and labels.

        Input sequence: [compressed (k) | sep (1) | prompt (L_p) | response (L_t)]

        To save VRAM, we call the inner transformer to get hidden states
        then apply lm_head **only** to response-relevant positions, avoiding
        the full (B, total_len, V) logit tensor.

        Caller (``forward()``) is responsible for toggling LoRA OFF before
        calling this method.  Gradient checkpointing stays ON; a backward
        hook on ``encoder_hidden`` ensures encode layers recompute with
        LoRA ON while decode layers recompute with LoRA OFF.
        """
        B = compressed.shape[0]
        device = compressed.device
        k = compressed.shape[1]

        embed_layer = self.get_embedding_layer()
        prompt_embeds = embed_layer(prompt_ids)     # (B, L_p, D)
        target_embeds = embed_layer(response_ids)   # (B, L_t, D)
        sep = self.separator.unsqueeze(0).expand(B, -1, -1)  # (B, 1, D)

        # Concatenate embeddings
        inputs_embeds = torch.cat(
            [compressed, sep, prompt_embeds, target_embeds], dim=1
        )

        # Build attention mask
        prefix_mask = torch.ones(B, k + 1, device=device, dtype=prompt_mask.dtype)
        attn_mask = torch.cat(
            [prefix_mask, prompt_mask, response_mask], dim=1
        )

        L_p = prompt_ids.shape[1]
        L_t = response_ids.shape[1]
        prefix_len = k + 1 + L_p

        # Use inner transformer to get hidden states, then apply lm_head
        # only to response-relevant positions.  This avoids materializing
        # the full (B, total_len, V) logit tensor — saving ~50% decode VRAM.
        causal_lm = self._get_inner_model()
        outputs = causal_lm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # (B, total_len, D)

        # Logits at positions [prefix_len-1, prefix_len+L_t-1) predict
        # tokens at positions [prefix_len, prefix_len+L_t), i.e. response_ids.
        # This pre-applies the causal shift so no shift is needed in loss.
        relevant_hidden = last_hidden[:, prefix_len - 1 : prefix_len + L_t - 1, :]
        logits = causal_lm.lm_head(relevant_hidden)  # (B, L_t, V)

        # Labels: response_ids with padding masked to -100
        labels = response_ids.clone()
        labels[response_mask == 0] = -100

        return {
            "logits": logits,   # (B, L_t, V) — response positions only
            "labels": labels,   # (B, L_t)
        }

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict:
        """Full QA training forward pass.

        Returns dict with ``loss`` and ``qa_loss``.
        """
        # 1. Encode context
        encoder_hidden = self.encode(context_ids, context_mask)

        # 2. Compress
        compressed = self.compress(
            encoder_hidden, context_mask, prompt_ids, prompt_mask
        )

        # 3. Disable LoRA for frozen-decoder pass (grad-ckpt stays ON).
        #    A backward hook on encoder_hidden re-enables LoRA before
        #    encode layers are recomputed, keeping each direction correct.
        if self._has_lora:
            self._disable_adapter_forward_only()
            if encoder_hidden.requires_grad:
                encoder_hidden.register_hook(
                    lambda grad: self._enable_adapter_forward_only() or grad
                )

        # 4. Student decode (returns response-only logits & labels)
        dec_out = self.decode_train(
            compressed, prompt_ids, prompt_mask, response_ids, response_mask
        )
        logits = dec_out["logits"]    # (B, L_t, V) — response positions only
        labels = dec_out["labels"]    # (B, L_t)

        # 5. QA loss (CE on response tokens, fp32 for numerical safety)
        # logits[:, i, :] predicts labels[:, i] — already aligned, no shift
        qa_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        return {
            "loss": qa_loss,
            "qa_loss": qa_loss,
        }

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        compressed: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation from compressed representation.

        Args:
            compressed:  (B, k, D) soft prompts from Perceiver
            prompt_ids:  (B, L_p) prompt token ids
            prompt_mask: (B, L_p) prompt attention mask
        Returns:
            (B, max_generated) generated token ids
        """
        gc_was_on = False
        if self._has_lora:
            gc_was_on = self._pause_gradient_checkpointing()
            self._disable_adapter_forward_only()

        B = compressed.shape[0]
        device = compressed.device
        k = compressed.shape[1]

        embed_layer = self.get_embedding_layer()
        prompt_embeds = embed_layer(prompt_ids)
        sep = self.separator.unsqueeze(0).expand(B, -1, -1)

        # Initial input
        inputs_embeds = torch.cat([compressed, sep, prompt_embeds], dim=1)
        prefix_mask = torch.ones(B, k + 1, device=device, dtype=prompt_mask.dtype)
        attn_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

        generated_ids = []
        past_key_values = None

        for step in range(max_new_tokens):
            if step == 0:
                outputs = self.base_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attn_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = self.base_model(
                    inputs_embeds=next_embeds,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]  # (B, V)

            # Sampling
            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = (cumsum - sorted_probs) > top_p
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, 1)
                next_token = sorted_idx.gather(-1, sampled)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token)

            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Prepare next step
            next_embeds = embed_layer(next_token)  # (B, 1, D)
            attn_mask = torch.cat(
                [attn_mask, torch.ones(B, 1, device=device, dtype=attn_mask.dtype)],
                dim=1,
            )

        if self._has_lora:
            self._enable_adapter_forward_only()
            self._resume_gradient_checkpointing(gc_was_on)
        return torch.cat(generated_ids, dim=-1)  # (B, num_generated)

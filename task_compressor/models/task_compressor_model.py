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

        # ── 2. Add LoRA adapters ─────────────────────────────────────────
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

        # ── 6. Separator token ───────────────────────────────────────────
        self.separator = nn.Parameter(
            torch.randn(1, hidden_size, dtype=torch_dtype) * 0.02
        )

        # ── Gradient checkpointing ───────────────────────────────────────
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    # ── Helpers ──────────────────────────────────────────────────────────

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
        """Encode context through LoRA-augmented Qwen.

        Returns:
            (B, L_c, D) last hidden states.
        """
        self.base_model.enable_adapter_layers()
        outputs = self.base_model(
            input_ids=context_ids,
            attention_mask=context_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

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

        Gradient checkpointing is temporarily disabled during decode because
        the LoRA adapter toggle changes the computation graph.  Checkpointing
        recomputation would see the wrong adapter state, producing shape
        mismatches.  The decode sequence is typically much shorter than the
        encode context, so the memory cost is acceptable.
        """
        gc_was_on = self._pause_gradient_checkpointing()
        self._disable_adapter_forward_only()
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

        # Forward through frozen decoder
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            return_dict=True,
        )
        logits = outputs.logits  # (B, total_len, V)

        # Build labels: only compute loss on response tokens
        L_p = prompt_ids.shape[1]
        L_t = response_ids.shape[1]
        prefix_len = k + 1 + L_p
        total_len = inputs_embeds.shape[1]

        labels = torch.full(
            (B, total_len), -100, dtype=torch.long, device=device
        )
        labels[:, prefix_len: prefix_len + L_t] = response_ids
        # Mask padding positions in response
        pad_positions = response_mask == 0
        labels[:, prefix_len: prefix_len + L_t][pad_positions] = -100

        # Re-enable adapter forward so subsequent calls (e.g. encode) work
        self._enable_adapter_forward_only()
        self._resume_gradient_checkpointing(gc_was_on)

        return {
            "logits": logits,
            "labels": labels,
            "prefix_len": prefix_len,
        }

    def teacher_forward(
        self,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Frozen teacher forward (LoRA disabled, no gradients).

        Args:
            teacher_input_ids:      (B, L_total) — concat of context + prompt + response
            teacher_attention_mask: (B, L_total)
        Returns:
            (B, L_total, V) teacher logits.
        """
        gc_was_on = self._pause_gradient_checkpointing()
        self._disable_adapter_forward_only()
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                return_dict=True,
            )
        self._enable_adapter_forward_only()
        self._resume_gradient_checkpointing(gc_was_on)
        return outputs.logits

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        teacher_input_ids: torch.Tensor | None = None,
        teacher_attention_mask: torch.Tensor | None = None,
        response_starts: torch.Tensor | None = None,
        response_lens: torch.Tensor | None = None,
        distill_alpha: float = 0.0,
        distill_temperature: float = 2.0,
        distill_method: str = "kl",
    ) -> dict:
        """Full training forward pass.

        Returns dict with ``loss``, ``qa_loss``, ``distill_loss``.
        """
        B = context_ids.shape[0]
        device = context_ids.device

        # 1. Encode context
        encoder_hidden = self.encode(context_ids, context_mask)

        # 2. Compress
        compressed = self.compress(
            encoder_hidden, context_mask, prompt_ids, prompt_mask
        )

        # 3. Student decode
        dec_out = self.decode_train(
            compressed, prompt_ids, prompt_mask, response_ids, response_mask
        )
        student_logits = dec_out["logits"]
        labels = dec_out["labels"]
        prefix_len = dec_out["prefix_len"]

        # 4. QA loss (CE on response tokens)
        shift_logits = student_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        qa_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # 5. Distillation loss (optional)
        distill_loss = torch.tensor(0.0, device=device)
        if (
            distill_alpha > 0
            and teacher_input_ids is not None
            and response_starts is not None
        ):
            teacher_logits = self.teacher_forward(
                teacher_input_ids, teacher_attention_mask
            )

            # Extract response-predicting logits
            L_t = response_ids.shape[1]
            V = student_logits.shape[-1]

            # Student: logits at positions [prefix_len-1, prefix_len-1+L_t)
            student_resp = student_logits[:, prefix_len - 1: prefix_len - 1 + L_t, :]

            # Teacher: per-sample extraction (response_starts vary)
            teacher_resp = torch.zeros(B, L_t, V, device=device, dtype=student_resp.dtype)
            for i in range(B):
                start = response_starts[i].item() - 1
                length = min(response_lens[i].item(), L_t)
                teacher_resp[i, :length] = teacher_logits[i, start: start + length].to(student_resp.dtype)

            if distill_method == "kl":
                student_log_p = F.log_softmax(student_resp / distill_temperature, dim=-1)
                teacher_p = F.softmax(teacher_resp / distill_temperature, dim=-1)
                kl = F.kl_div(student_log_p, teacher_p, reduction="none").sum(-1)
                kl = kl * response_mask.float()
                distill_loss = kl.sum() / response_mask.float().sum().clamp(min=1)
                distill_loss = distill_loss * (distill_temperature ** 2)
            else:  # mse
                # Extract hidden states instead — simplified: use logits MSE
                mse = (student_resp - teacher_resp).pow(2).mean(-1)
                mse = mse * response_mask.float()
                distill_loss = mse.sum() / response_mask.float().sum().clamp(min=1)

        total_loss = qa_loss + distill_alpha * distill_loss
        return {
            "loss": total_loss,
            "qa_loss": qa_loss,
            "distill_loss": distill_loss,
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

        self._enable_adapter_forward_only()
        self._resume_gradient_checkpointing(gc_was_on)
        return torch.cat(generated_ids, dim=-1)  # (B, num_generated)

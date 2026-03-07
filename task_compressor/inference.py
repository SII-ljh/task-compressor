"""Inference pipeline for Task Compressor.

Provides:
  - EncoderCacheManager: cache encoded context hidden states (put/get/evict)
  - TaskCompressorPipeline: end-to-end encode → compress → generate
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .models.task_compressor_model import TaskCompressorModel


class EncoderCacheManager:
    """Manage cached encoder hidden states on disk (safetensors format)."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}

        # Load existing cache index
        for p in self.cache_dir.glob("*.safetensors"):
            key = p.stem
            self._index[key] = p

    def _make_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def put(
        self,
        key: str,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        path = self.cache_dir / f"{key}.safetensors"
        save_file(
            {"hidden_states": hidden_states.cpu(), "attention_mask": attention_mask.cpu()},
            str(path),
        )
        self._index[key] = path

    def get(self, key: str) -> dict[str, torch.Tensor] | None:
        if key not in self._index:
            return None
        path = self._index[key]
        if not path.exists():
            del self._index[key]
            return None
        data = load_file(str(path))
        return {
            "hidden_states": data["hidden_states"],
            "attention_mask": data["attention_mask"],
        }

    def evict(self, key: str) -> bool:
        if key not in self._index:
            return False
        path = self._index.pop(key)
        path.unlink(missing_ok=True)
        return True

    def keys(self) -> list[str]:
        return list(self._index.keys())

    def __len__(self) -> int:
        return len(self._index)


class TaskCompressorPipeline:
    """End-to-end inference pipeline: encode → cache → compress → generate."""

    def __init__(
        self,
        model: TaskCompressorModel,
        tokenizer: AutoTokenizer,
        device: torch.device | str = "cuda",
        cache_dir: str | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.cache = EncoderCacheManager(cache_dir) if cache_dir else None

    @torch.no_grad()
    def encode_context(
        self,
        context: str,
        cache_key: str | None = None,
        max_length: int = 4096,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode context text and optionally cache the result.

        Returns:
            (hidden_states, attention_mask) both on self.device
        """
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attn_mask = inputs["attention_mask"].to(self.device)

        hidden_states = self.model.encode(input_ids, attn_mask)

        if cache_key and self.cache:
            self.cache.put(cache_key, hidden_states, attn_mask)

        return hidden_states, attn_mask

    @torch.no_grad()
    def generate(
        self,
        context: str | None = None,
        prompt: str = "",
        cache_key: str | None = None,
        max_context_length: int = 4096,
        max_prompt_length: int = 256,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response given context and prompt.

        If ``cache_key`` is provided and found in cache, uses cached hidden
        states instead of re-encoding.
        """
        # 1. Get encoder hidden states
        cached = None
        if cache_key and self.cache:
            cached = self.cache.get(cache_key)

        if cached is not None:
            hidden_states = cached["hidden_states"].to(self.device)
            encoder_mask = cached["attention_mask"].to(self.device)
        elif context is not None:
            hidden_states, encoder_mask = self.encode_context(
                context, cache_key=cache_key, max_length=max_context_length
            )
        else:
            raise ValueError("Must provide either context or a valid cache_key")

        # 2. Tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )
        prompt_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_mask = prompt_inputs["attention_mask"].to(self.device)

        # 3. Compress
        compressed = self.model.compress(
            hidden_states, encoder_mask, prompt_ids, prompt_mask
        )

        # 4. Generate
        eos_id = self.tokenizer.eos_token_id
        generated_ids = self.model.generate(
            compressed,
            prompt_ids,
            prompt_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
        )

        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def benchmark(
        self,
        context: str,
        prompt: str,
        n_runs: int = 10,
        max_new_tokens: int = 64,
    ) -> dict:
        """Benchmark encode, cache-hit, compress, and generate latency."""
        results = {}

        # Encode latency
        t0 = time.perf_counter()
        hidden, mask = self.encode_context(context, cache_key="__bench__")
        results["encode_ms"] = (time.perf_counter() - t0) * 1000

        # Cache hit latency
        t0 = time.perf_counter()
        cached = self.cache.get("__bench__") if self.cache else None
        if cached is not None:
            _ = cached["hidden_states"].to(self.device)
        results["cache_hit_ms"] = (time.perf_counter() - t0) * 1000

        # Tokenize prompt
        prompt_inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=256, add_special_tokens=False,
        )
        prompt_ids = prompt_inputs["input_ids"].to(self.device)
        prompt_mask = prompt_inputs["attention_mask"].to(self.device)

        # Compress latency
        t0 = time.perf_counter()
        compressed = self.model.compress(hidden, mask, prompt_ids, prompt_mask)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        results["compress_ms"] = (time.perf_counter() - t0) * 1000

        # Generate latency (averaged)
        gen_times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.model.generate(
                compressed, prompt_ids, prompt_mask,
                max_new_tokens=max_new_tokens, temperature=0,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            gen_times.append((time.perf_counter() - t0) * 1000)

        results["generate_ms_mean"] = sum(gen_times) / len(gen_times)
        results["generate_ms_p50"] = sorted(gen_times)[len(gen_times) // 2]

        # Cleanup
        if self.cache:
            self.cache.evict("__bench__")

        return results

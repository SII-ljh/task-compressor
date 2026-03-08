"""Dataset and data collation for Task Compressor training.

QA mode: Each sample is a (context, prompt/question, response/answer) triple.
NTP mode: Each sample is a document split into (doc, segment) for next-token
prediction pretraining.

The collators pad each component separately and build the required masks.
"""

import json
import os
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class QADataset(Dataset):
    """QA dataset: each item has context, question, answer."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_context_length: int = 4096,
        max_prompt_length: int = 256,
        max_response_length: int = 512,
    ):
        data_path = Path(data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        context_enc = self.tokenizer(
            item["context"],
            truncation=True,
            max_length=self.max_context_length,
            add_special_tokens=False,
        )
        prompt_enc = self.tokenizer(
            item["question"],
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        )
        response_enc = self.tokenizer(
            item["answer"],
            truncation=True,
            max_length=self.max_response_length,
            add_special_tokens=False,
        )

        return {
            "context_ids": torch.tensor(context_enc["input_ids"], dtype=torch.long),
            "prompt_ids": torch.tensor(prompt_enc["input_ids"], dtype=torch.long),
            "response_ids": torch.tensor(response_enc["input_ids"], dtype=torch.long),
        }


class QACollator:
    """Pads each component and builds teacher input sequences.

    Returns a dict with:
        - context_ids, context_mask    (B, max_L_c)
        - prompt_ids, prompt_mask      (B, max_L_p)
        - response_ids, response_mask  (B, max_L_t)
        - teacher_input_ids, teacher_attention_mask  (B, max_L_teacher)
        - response_starts  (B,) — position where response begins in teacher sequence
        - response_lens    (B,) — unpadded response length per sample
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        context_ids_list = [item["context_ids"] for item in batch]
        prompt_ids_list = [item["prompt_ids"] for item in batch]
        response_ids_list = [item["response_ids"] for item in batch]

        # Pad each component separately
        context_ids = pad_sequence(
            context_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        prompt_ids = pad_sequence(
            prompt_ids_list, batch_first=True, padding_value=self.pad_token_id
        )
        response_ids = pad_sequence(
            response_ids_list, batch_first=True, padding_value=self.pad_token_id
        )

        context_mask = (context_ids != self.pad_token_id).long()
        prompt_mask = (prompt_ids != self.pad_token_id).long()
        response_mask = (response_ids != self.pad_token_id).long()

        # Build teacher sequences: concatenate unpadded per-sample, then pad
        teacher_seqs = []
        response_starts = []
        response_lens = []
        for ctx, prm, resp in zip(
            context_ids_list, prompt_ids_list, response_ids_list
        ):
            teacher_seqs.append(torch.cat([ctx, prm, resp]))
            response_starts.append(len(ctx) + len(prm))
            response_lens.append(len(resp))

        teacher_input_ids = pad_sequence(
            teacher_seqs, batch_first=True, padding_value=self.pad_token_id
        )
        teacher_attention_mask = (teacher_input_ids != self.pad_token_id).long()

        return {
            "context_ids": context_ids,
            "context_mask": context_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "response_ids": response_ids,
            "response_mask": response_mask,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "response_starts": torch.tensor(response_starts, dtype=torch.long),
            "response_lens": torch.tensor(response_lens, dtype=torch.long),
        }


# ── NTP (Next-Token Prediction) pretraining data ─────────────────────────


class NTPDataset(Dataset):
    """NTP dataset: lazy-loading JSONL ``{"text": "..."}`` with byte-offset index.

    Each item is a document split at a random point into ``doc_ids`` (context)
    and ``segment_ids`` (continuation to predict).  Per-worker file handles are
    reopened after fork to be safe with :class:`DataLoader` workers.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_context_length: int = 4096,
        ntp_segment_len: int = 256,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.ntp_segment_len = ntp_segment_len

        # Build byte-offset index (one pass through the file)
        self._offsets: list[int] = []
        with open(self.data_path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self._offsets.append(offset)

        # Per-worker file handle (initialised lazily after fork)
        self._fh = None
        self._worker_id: int | None = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _get_file_handle(self):
        """Return a file handle, reopening if we're in a new worker."""
        worker_info = torch.utils.data.get_worker_info()
        current_id = worker_info.id if worker_info is not None else -1
        if self._fh is None or self._worker_id != current_id:
            if self._fh is not None:
                self._fh.close()
            self._fh = open(self.data_path, "r", encoding="utf-8")
            self._worker_id = current_id
        return self._fh

    def __getitem__(self, idx: int) -> dict:
        fh = self._get_file_handle()
        fh.seek(self._offsets[idx])
        line = fh.readline()
        item = json.loads(line)

        tokens = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_context_length + self.ntp_segment_len,
            add_special_tokens=False,
        )["input_ids"]

        # Need at least 2 tokens to split into doc + segment
        if len(tokens) < 2:
            tokens = tokens + [self.tokenizer.pad_token_id or 0] * (2 - len(tokens))

        # Split: ensure at least 1 token in each part
        max_doc_len = min(len(tokens) - 1, self.max_context_length)
        split_point = random.randint(1, max_doc_len)
        doc_ids = tokens[:split_point]
        segment_ids = tokens[split_point: split_point + self.ntp_segment_len]
        if not segment_ids:
            segment_ids = tokens[-1:]

        doc_ids = torch.tensor(doc_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return {
            "doc_ids": doc_ids,
            "segment_ids": segment_ids,
            "segment_labels": segment_ids.clone(),
        }


class NTPCollator:
    """Pad NTP batch fields and generate masks.

    Returns a dict with:
        - doc_ids, doc_mask       (B, max_L_doc)
        - segment_ids, segment_mask, segment_labels  (B, max_L_seg)
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        doc_ids = pad_sequence(
            [item["doc_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        segment_ids = pad_sequence(
            [item["segment_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        segment_labels = pad_sequence(
            [item["segment_labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        doc_mask = (doc_ids != self.pad_token_id).long()
        segment_mask = (segment_ids != self.pad_token_id).long()

        return {
            "doc_ids": doc_ids,
            "doc_mask": doc_mask,
            "segment_ids": segment_ids,
            "segment_mask": segment_mask,
            "segment_labels": segment_labels,
        }

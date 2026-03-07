"""Dataset and data collation for Task Compressor training.

Each sample is a (context, prompt/question, response/answer) triple.
The collator pads each component separately and also builds the concatenated
teacher input sequence (context + prompt + response without intermediate padding).
"""

import json
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

"""Dataset and data collation for Task Compressor training.

Each sample is a (context, prompt/question, response/answer) triple.
The collator pads each component separately and builds the required masks.
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
    """Pads each component and builds attention masks.

    Returns a dict with:
        - context_ids, context_mask    (B, max_L_c)
        - prompt_ids, prompt_mask      (B, max_L_p)
        - response_ids, response_mask  (B, max_L_t)
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

        return {
            "context_ids": context_ids,
            "context_mask": context_mask,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "response_ids": response_ids,
            "response_mask": response_mask,
        }

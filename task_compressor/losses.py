"""Loss functions for Task Compressor training.

L_QA: Cross-entropy on response tokens.
"""

import torch
import torch.nn.functional as F


def compute_qa_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss on response tokens only.

    Uses the standard shift-by-one approach: logits[i] predicts labels[i+1].

    Args:
        logits: (B, seq_len, V)
        labels: (B, seq_len) with -100 for non-response positions
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )

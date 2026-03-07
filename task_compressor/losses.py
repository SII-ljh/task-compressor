"""Loss functions for Task Compressor training.

L = L_QA + alpha * L_distill

- L_QA:     Cross-entropy on response tokens.
- L_distill: KL divergence (default) or MSE between student and teacher logits
             at response token positions.
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


def compute_distill_loss_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL-divergence distillation loss.

    KL(teacher || student) scaled by T^2.

    Args:
        student_logits: (B, L, V) student logits at response positions
        teacher_logits: (B, L, V) teacher logits at response positions
        mask:           (B, L) 1 for real response tokens, 0 for padding
        temperature:    softmax temperature
    """
    student_log_p = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits / temperature, dim=-1)

    # kl_div with log_target=False: KL(teacher || student)
    kl = F.kl_div(student_log_p, teacher_p, reduction="none").sum(dim=-1)  # (B, L)

    kl = kl * mask.float()
    loss = kl.sum() / mask.float().sum().clamp(min=1)
    return loss * (temperature ** 2)


def compute_distill_loss_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE distillation loss on hidden states.

    Args:
        student_hidden: (B, L, D) student hidden states at response positions
        teacher_hidden: (B, L, D) teacher hidden states at response positions
        mask:           (B, L) 1 for real response tokens, 0 for padding
    """
    mse = (student_hidden - teacher_hidden).pow(2).mean(dim=-1)  # (B, L)
    mse = mse * mask.float()
    return mse.sum() / mask.float().sum().clamp(min=1)

"""Symmetric InfoNCE (CLIP-style) loss."""

import torch
import torch.nn.functional as F


def info_nce(z_eeg: torch.Tensor, z_img: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy over the cosine-similarity matrix of a batch.

    z_eeg, z_img: (B, D) embeddings, row i of each is a positive pair.
    logit_scale: scalar, log of the inverse temperature.
    """
    z_eeg = F.normalize(z_eeg, dim=-1)
    z_img = F.normalize(z_img, dim=-1)
    logits = logit_scale.exp() * z_eeg @ z_img.t()
    target = torch.arange(z_eeg.shape[0], device=z_eeg.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))

"""Retrieval metrics and a binomial test against chance."""

from math import comb

import numpy as np


def similarity(eeg_emb: np.ndarray, img_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (n_eeg, n_img)."""
    a = eeg_emb / np.linalg.norm(eeg_emb, axis=1, keepdims=True)
    b = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    return a @ b.T


def topk_accuracy(sim: np.ndarray, ks=(1, 5)) -> dict:
    """Row i's correct candidate is column i. Returns {k: accuracy}."""
    n = sim.shape[0]
    correct = sim[np.arange(n), np.arange(n)][:, None]
    # rank = number of candidates strictly more similar than the correct one
    rank = (sim > correct).sum(axis=1)
    return {k: float((rank < k).mean()) for k in ks}


def binomial_p(successes: int, n: int, p: float) -> float:
    """One-sided P(X >= successes) for X ~ Binomial(n, p)."""
    return float(sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(successes, n + 1)))

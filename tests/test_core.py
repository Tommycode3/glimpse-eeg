"""Fast tests on synthetic data. No downloads."""

import numpy as np
import torch

from eeg2img.losses import info_nce
from eeg2img.metrics import binomial_p, similarity, topk_accuracy
from eeg2img.model import EEGEncoder
from eeg2img.train import Config, fit


def synthetic(n, n_ch=17, n_t=40, dim=32, noise=0.5, seed=0):
    """EEG that is a fixed linear mixing of the image embedding plus noise."""
    rng = np.random.default_rng(seed)
    mix = np.random.default_rng(99).standard_normal((dim, n_ch * n_t)) / np.sqrt(dim)
    y = rng.standard_normal((n, dim)).astype(np.float32)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = (y @ mix * np.sqrt(dim) + noise * rng.standard_normal((n, n_ch * n_t))).astype(np.float32)
    return x.reshape(n, n_ch, n_t), y


def test_shapes():
    for t in (5, 10, 40, 100):
        m = EEGEncoder(n_channels=17, n_times=t, emb_dim=512)
        assert m(torch.zeros(3, 17, t)).shape == (3, 512)
    m = EEGEncoder(n_channels=8, n_times=80, emb_dim=64)
    assert m(torch.zeros(2, 8, 80)).shape == (2, 64)


def test_topk_and_chance():
    assert topk_accuracy(np.eye(200)) == {1: 1.0, 5: 1.0}
    rng = np.random.default_rng(0)
    acc = np.mean([topk_accuracy(rng.standard_normal((200, 200)))[1] for _ in range(50)])
    assert abs(acc - 1 / 200) < 0.004
    assert abs(binomial_p(0, 200, 0.005) - 1.0) < 1e-9
    assert binomial_p(30, 200, 0.005) < 1e-20


def test_info_nce_prefers_aligned_pairs():
    z = torch.randn(64, 32)
    s = torch.tensor(np.log(10.0))
    assert info_nce(z, z, s) < info_nce(z, torch.randn(64, 32), s)


def _cfg(**kw):
    return Config(epochs=40, patience=40, batch=256, lr=1e-3, channels=[f"c{i}" for i in range(17)], **kw)


def test_info_nce_training_reaches_high_retrieval():
    x, y = synthetic(2000)
    xv, yv = synthetic(200, seed=1)
    model, _, _ = fit(_cfg(), x[:1800], y[:1800], x[1800:], y[1800:])
    from eeg2img.train import encode
    acc = topk_accuracy(similarity(encode(model, xv), yv))
    assert acc[1] > 0.5, acc
    assert acc[5] > 0.8, acc


def test_shuffled_labels_stay_at_chance():
    x, y = synthetic(2000)
    xv, yv = synthetic(200, seed=1)
    model, _, _ = fit(_cfg(shuffle_labels=True), x[:1800], y[:1800], x[1800:], y[1800:])
    from eeg2img.train import encode
    acc = topk_accuracy(similarity(encode(model, xv), yv))
    assert acc[5] < 0.08, acc  # chance is 0.025

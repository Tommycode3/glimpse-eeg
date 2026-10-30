"""Train an EEG encoder for one subject and evaluate 200-way zero-shot retrieval."""

import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .data import CH_NAMES, load_subject
from .embed import emb_path
from .losses import info_nce
from .metrics import similarity, topk_accuracy
from .model import EEGEncoder

N_VAL_CONCEPTS = 74  # 740 training images held out by concept for model selection


@dataclass
class Config:
    subject: int = 1
    seed: int = 0
    channels: list = field(default_factory=lambda: list(CH_NAMES))
    t0: int = 20          # sample index, 20 = 0 ms
    t1: int = 100         # exclusive, 100 = 800 ms
    epochs: int = 100
    patience: int = 20
    batch: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-2
    shuffle_labels: bool = False


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def split_indices(n_train: int, seed: int = 1234):
    """Hold out whole concepts (10 images each) for validation; fixed across runs."""
    concepts = np.arange(n_train) // 10
    rng = np.random.default_rng(seed)
    val_c = rng.choice(concepts.max() + 1, N_VAL_CONCEPTS, replace=False)
    val = np.isin(concepts, val_c)
    return np.where(~val)[0], np.where(val)[0]


def prepare_arrays(cfg: Config, train_eeg, test_eeg):
    ch = [CH_NAMES.index(c) for c in cfg.channels]
    tr = np.asarray(train_eeg[:, ch, cfg.t0:cfg.t1], dtype=np.float32)
    te = np.asarray(test_eeg[:, ch, cfg.t0:cfg.t1], dtype=np.float32)
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True) + 1e-6  # fit on training data only
    return (tr - mu) / sd, (te - mu) / sd


def fit(cfg: Config, x_tr, y_tr, x_val, y_val, verbose=False):
    """Train with InfoNCE; keep the epoch with the best validation top-5."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = device()
    if cfg.shuffle_labels:  # control: break the EEG-image pairing
        y_tr = y_tr[rng.permutation(len(y_tr))]
    xt, yt = torch.from_numpy(x_tr).to(dev), torch.from_numpy(y_tr).to(dev)
    model = EEGEncoder(x_tr.shape[1], x_tr.shape[2], y_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, since = -1.0, None, 0
    for ep in range(cfg.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(xt))).to(dev)
        for i in range(0, len(perm) - cfg.batch // 2, cfg.batch):  # drop a short last batch
            idx = perm[i:i + cfg.batch]
            loss = info_nce(model(xt[idx]), yt[idx], model.logit_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, np.log(100))
        val = topk_accuracy(similarity(encode(model, x_val), y_val))[5]
        if verbose:
            print(f"ep {ep:3d} loss {loss.item():.3f} val top5 {val:.3f}", flush=True)
        if val > best:
            best, since = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, best, ep + 1


@torch.no_grad()
def encode(model, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = [model(torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(dev)).cpu().numpy()
           for i in range(0, len(x), batch)]
    return np.concatenate(out)


def run(cfg: Config, verbose=False, return_model=False):
    train_eeg, test_eeg = load_subject(cfg.subject)
    img_tr = np.load(emb_path("train"), mmap_mode="r")
    img_te = np.load(emb_path("test"))
    x_tr, x_te = prepare_arrays(cfg, train_eeg, test_eeg)
    tr_idx, val_idx = split_indices(len(x_tr))
    model, val_top5, epochs = fit(cfg, x_tr[tr_idx], np.asarray(img_tr[tr_idx]),
                                  x_tr[val_idx], np.asarray(img_tr[val_idx]), verbose)
    sim = similarity(encode(model, x_te), img_te)
    acc = topk_accuracy(sim)
    res = {"config": asdict(cfg), "top1": acc[1], "top5": acc[5], "val_top5_740way": val_top5, "epochs": epochs}
    if return_model:
        return res, model, x_te, img_te, sim
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--shuffle-labels", action="store_true")
    a = ap.parse_args()
    from .memory import wait_for_memory
    wait_for_memory(25)
    res = run(Config(subject=a.subject, seed=a.seed, epochs=a.epochs, shuffle_labels=a.shuffle_labels), verbose=True)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

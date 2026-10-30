"""All experiments in the README. Results are appended to results/runs.jsonl.

Runs are sequential (one subject, one model at a time) and resumable: a run whose
key is already in the results file is skipped.
"""

import argparse
import json
from dataclasses import replace

import numpy as np

from .data import CH_NAMES, ROOT, load_subject
from .embed import emb_path
from .memory import wait_for_memory
from .metrics import similarity, topk_accuracy
from .train import Config, encode, prepare_arrays, run, split_indices

RES = ROOT / "results"
RUNS = RES / "runs.jsonl"
OCCIPITAL = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"]
PARIETAL = [c for c in CH_NAMES if c not in OCCIPITAL]
WIN = 10    # 100 ms windows
STEP = 5    # 50 ms steps


def done_keys():
    if not RUNS.exists():
        return set()
    return {json.loads(line)["key"] for line in RUNS.read_text().splitlines() if line.strip()}


def log(key, res):
    RES.mkdir(exist_ok=True)
    res = {"key": key, **res}
    with RUNS.open("a") as f:
        f.write(json.dumps(res) + "\n")
    print(key, f"top1={res['top1']:.3f} top5={res['top5']:.3f}", flush=True)


def ridge(cfg: Config):
    """Closed-form ridge regression from flattened EEG to CLIP, lambda chosen on validation."""
    train_eeg, test_eeg = load_subject(cfg.subject)
    y = np.load(emb_path("train")).astype(np.float64)
    x_tr, x_te = prepare_arrays(cfg, train_eeg, test_eeg)
    x_tr, x_te = x_tr.reshape(len(x_tr), -1).astype(np.float64), x_te.reshape(len(x_te), -1).astype(np.float64)
    tr, va = split_indices(len(x_tr))
    xtx, xty = x_tr[tr].T @ x_tr[tr], x_tr[tr].T @ y[tr]
    best = None
    for lam in 10.0 ** np.arange(1, 7):
        w = np.linalg.solve(xtx + lam * np.eye(len(xtx)), xty)
        v = topk_accuracy(similarity(x_tr[va] @ w, y[va]))[5]
        if best is None or v > best[0]:
            best = (v, lam, w)
    acc = topk_accuracy(similarity(x_te @ best[2], np.load(emb_path("test"))))
    return {"top1": acc[1], "top5": acc[5], "lambda": best[1], "val_top5_740way": best[0]}


def occlusion(model, x_te, img_te):
    """Drop in test accuracy when one channel, or one 50 ms bin, is set to its training mean (0)."""
    base = topk_accuracy(similarity(encode(model, x_te), img_te))
    out = {"base": base, "channels": {}, "bins": []}
    for c, name in enumerate(CH_NAMES):
        x = x_te.copy()
        x[:, c] = 0
        a = topk_accuracy(similarity(encode(model, x), img_te))
        out["channels"][name] = {k: base[k] - a[k] for k in a}
    for t in range(0, x_te.shape[2], 5):
        x = x_te.copy()
        x[:, :, t:t + 5] = 0
        a = topk_accuracy(similarity(encode(model, x), img_te))
        out["bins"].append({"t_ms": (t + 20) * 10 - 200, **{str(k): base[k] - a[k] for k in a}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: main ridge shuffle electrodes windows")
    a = ap.parse_args()
    parts = a.only or ["main", "ridge", "shuffle", "electrodes", "windows"]
    done = done_keys()

    def go(key, cfg, fn=None):
        if key in done:
            return
        wait_for_memory(25)
        if fn is not None:
            log(key, {"config": cfg.__dict__, **fn(cfg)})
            return
        if key.startswith("main") and cfg.seed == 0:
            res, model, x_te, img_te, sim = run(cfg, return_model=True)
            (RES / "similarity").mkdir(parents=True, exist_ok=True)
            np.save(RES / "similarity" / f"sub-{cfg.subject:02d}.npy", sim.astype(np.float32))
            (RES / "occlusion").mkdir(parents=True, exist_ok=True)
            occ = occlusion(model, x_te, img_te)
            (RES / "occlusion" / f"sub-{cfg.subject:02d}.json").write_text(json.dumps(occ, indent=1))
            log(key, res)
        else:
            log(key, run(cfg))

    for s in a.subjects:
        full = Config(subject=s, epochs=150, patience=20)
        if "main" in parts:
            for seed in range(a.seeds):
                go(f"main/sub-{s:02d}/seed-{seed}", replace(full, seed=seed))
        if "ridge" in parts:
            go(f"ridge/sub-{s:02d}", full, ridge)
        if "shuffle" in parts:
            for seed in range(a.seeds):
                go(f"shuffle/sub-{s:02d}/seed-{seed}", replace(full, seed=seed, shuffle_labels=True, epochs=60))
        if "electrodes" in parts:
            for name, chs in (("occipital", OCCIPITAL), ("parietal", PARIETAL)):
                go(f"electrodes/{name}/sub-{s:02d}", replace(full, channels=chs))
        if "windows" in parts:
            for t0 in range(0, 100 - WIN + 1, STEP):
                go(f"windows/sub-{s:02d}/t0-{t0 * 10 - 200}", replace(full, t0=t0, t1=t0 + WIN, epochs=60, patience=10))


if __name__ == "__main__":
    main()

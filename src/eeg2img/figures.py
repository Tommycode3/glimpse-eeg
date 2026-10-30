"""Summary table and figures from results/. Run after experiments."""

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CH_NAMES, ROOT, concept_names
from .experiments import RES, RUNS
from .metrics import binomial_p

FIG = ROOT / "figures"
SUBJ_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "lines.linewidth": 2,
})


def load_runs():
    return [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]


def summarize(runs):
    g = defaultdict(list)
    for r in runs:
        parts = r["key"].split("/")
        name = "/".join(p for p in parts if not p.startswith("seed-"))
        g[name].append((r["top1"], r["top5"]))
    rows = {}
    for name, v in sorted(g.items()):
        a = np.array(v) * 100
        rows[name] = {"n": len(v), "top1": a[:, 0].mean(), "top1_sd": a[:, 0].std(), "top5": a[:, 1].mean(),
                      "top5_sd": a[:, 1].std()}
    # binomial p for main seed-0 runs (200 test trials)
    for r in runs:
        if r["key"].startswith("main/") and r["key"].endswith("seed-0"):
            name = r["key"].rsplit("/", 1)[0]
            rows[name]["p_top1_seed0"] = binomial_p(round(r["top1"] * 200), 200, 1 / 200)
    (RES / "summary.json").write_text(json.dumps(rows, indent=1))
    lines = ["| run | n | top-1 % | top-5 % |", "|---|---|---|---|"]
    for name, r in rows.items():
        if name.startswith("windows"):
            continue
        lines.append(f"| {name} | {r['n']} | {r['top1']:.1f} ± {r['top1_sd']:.1f} | {r['top5']:.1f} ± {r['top5_sd']:.1f} |")
    (RES / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return rows


def fig_windows(runs):
    by = defaultdict(dict)
    for r in runs:
        if r["key"].startswith("windows/"):
            _, sub, t0 = r["key"].split("/")
            by[sub][int(t0[3:]) + 50] = r["top5"] * 100  # window center, ms
    if not by:
        return
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, sub in enumerate(sorted(by)):
        t = sorted(by[sub])
        ax.plot(t, [by[sub][x] for x in t], color=SUBJ_COLORS[i], marker="o", ms=4, label=sub)
    ax.axhline(2.5, color=INK2, ls="--", lw=1)
    ax.text(-145, 3.3, "chance 2.5%", color=INK2, fontsize=9)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("center of 100 ms EEG window (ms after image onset)")
    ax.set_ylabel("top-5 accuracy, 200-way (%)")
    ax.set_title("Which latencies carry information", color=INK, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "time_windows.png", dpi=150)


def fig_occlusion():
    occ = sorted((RES / "occlusion").glob("sub-*.json"))
    if not occ:
        return
    data = [json.loads(p.read_text()) for p in occ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, d in enumerate(data):
        t = [b["t_ms"] + 25 for b in d["bins"]]
        a1.plot(t, [100 * b["5"] for b in d["bins"]], color=SUBJ_COLORS[i], marker="o", ms=4, label=occ[i].stem)
    a1.axhline(0, color=INK2, lw=0.8)
    a1.set_xlabel("center of occluded 50 ms bin (ms)")
    a1.set_ylabel("drop in top-5 accuracy (points)")
    a1.set_title("Temporal importance (occlusion)", color=INK, loc="left")
    a1.legend(frameon=False)
    drops = np.array([[100 * d["channels"][c]["5"] for c in CH_NAMES] for d in data])
    order = np.argsort(-drops.mean(0))
    a2.bar(range(len(CH_NAMES)), drops.mean(0)[order], color=SUBJ_COLORS[0], width=0.7)
    a2.set_xticks(range(len(CH_NAMES)), [CH_NAMES[i] for i in order], rotation=60, fontsize=8)
    a2.set_ylabel("drop in top-5 (points)")
    a2.set_title(f"Spatial importance, mean of {len(data)} subjects", color=INK, loc="left")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "occlusion.png", dpi=150)


def retrieval_rows(sub=1, n_rows=8, k=5):
    sim = np.load(RES / "similarity" / f"sub-{sub:02d}.npy")
    rows = np.linspace(0, len(sim) - 1, n_rows).astype(int)  # evenly spaced, not hand-picked
    return rows, np.argsort(-sim[rows], axis=1)[:, :k]


def fig_retrieval_names(sub=1):
    """Concept names only, so it can be committed (THINGS images may not be redistributed)."""
    if not (RES / "similarity" / f"sub-{sub:02d}.npy").exists():
        return
    names = concept_names("test")
    rows, top = retrieval_rows(sub)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 0.9))
    ax.axis("off")
    ax.text(0, len(rows) + 0.2, "image seen", weight="bold", color=INK)
    for j in range(top.shape[1]):
        ax.text(1.4 + j, len(rows) + 0.2, f"rank {j + 1}", color=INK2)
    for i, (r, tk) in enumerate(zip(rows, top)):
        y = len(rows) - 1 - i
        ax.text(0, y, names[r], color=INK)
        for j, c in enumerate(tk):
            hit = c == r
            ax.text(1.4 + j, y, names[c], color="#ffffff" if hit else INK2,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SUBJ_COLORS[2], ec="none") if hit else None)
    ax.set_xlim(0, 1.4 + top.shape[1])
    ax.set_ylim(-0.6, len(rows) + 0.7)
    fig.tight_layout()
    fig.savefig(FIG / "retrieval_names.png", dpi=150)


def fig_retrieval_images(sub=1):
    """Test image vs top-5 retrieved images. Written to figures/local/ (git-ignored)."""
    from PIL import Image

    from .data import image_paths

    paths, names = image_paths("test"), concept_names("test")
    rows, top = retrieval_rows(sub)
    fig, axes = plt.subplots(len(rows), top.shape[1] + 1, figsize=(1.6 * (top.shape[1] + 1), 1.7 * len(rows)))
    for i, (r, tk) in enumerate(zip(rows, top)):
        for j, c in enumerate([r, *tk]):
            ax = axes[i, j]
            ax.imshow(Image.open(paths[c]).resize((160, 160)))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(j > 0 and c == r)
                s.set_color(SUBJ_COLORS[2])
                s.set_linewidth(4)
            ax.set_title(names[c], fontsize=7, color=INK)
    axes[0, 0].set_title(f"seen: {names[rows[0]]}", fontsize=7, color=INK)
    fig.suptitle(f"sub-{sub:02d}: image seen (left) and top-5 retrieved from 200 unseen test images",
                 fontsize=9, color=INK)
    fig.tight_layout()
    (FIG / "local").mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "local" / f"retrieval_images_sub-{sub:02d}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", action="store_true", help="also render the local-only image figure")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    runs = load_runs()
    summarize(runs)
    fig_windows(runs)
    fig_occlusion()
    fig_retrieval_names()
    if a.images:
        fig_retrieval_images()


if __name__ == "__main__":
    main()

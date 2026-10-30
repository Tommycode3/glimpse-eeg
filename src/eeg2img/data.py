"""THINGS-EEG2 preprocessed data: download, convert to float32 memmaps, load.

Source: Gifford et al. (2022), OSF project 3jk45 (CC BY 4.0).
Preprocessed EEG: 17 posterior channels, 100 Hz, -200 to 790 ms.
"""

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"

OSF_EEG = {  # OSF file ids, node anp5v ("Preprocessed EEG data")
    1: "nb8wr", 2: "u43p5", 3: "fy4n6", 4: "py8jz", 5: "2cn7v",
    6: "bd83f", 7: "mw8hr", 8: "n9bzm", 9: "fv6jz", 10: "8fyu9",
}
OSF_IMAGES = {"image_metadata.npy": "qkgtf", "test_images.zip": "znu7b", "training_images.zip": "3v527"}

CH_NAMES = ["Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1", "P5",
            "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"]
TIMES = np.round(np.arange(-0.2, 0.8, 0.01), 2)  # seconds, 100 samples


def _download(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"downloading {dest.name}", flush=True)
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(dest), f"https://osf.io/download/{file_id}/"], check=True)


def _unzip(zpath: Path, into: Path) -> None:
    with zipfile.ZipFile(zpath) as z:
        z.extractall(into)
    zpath.unlink()


def download_images() -> None:
    for name, fid in OSF_IMAGES.items():
        stem = name.replace(".zip", "")
        if (RAW / stem).exists() or (RAW / name).exists() and not name.endswith(".zip"):
            continue
        _download(fid, RAW / name)
        if name.endswith(".zip"):
            _unzip(RAW / name, RAW)


def download_subject(sub: int) -> None:
    if (RAW / f"sub-{sub:02d}").exists() or (PROC / f"sub-{sub:02d}").exists():
        return
    z = RAW / f"sub-{sub:02d}.zip"
    _download(OSF_EEG[sub], z)
    _unzip(z, RAW)


def prepare_subject(sub: int, keep_raw: bool = False) -> Path:
    """Convert pickled float64 arrays to float32 .npy files that can be memory-mapped.

    train_avg.npy: (16540, 17, 100) mean over the 4 repetitions
    test_avg.npy: (200, 17, 100) mean over the 80 repetitions
    """
    out = PROC / f"sub-{sub:02d}"
    if (out / "test_avg.npy").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    src = RAW / f"sub-{sub:02d}"
    for split in ("training", "test"):
        d = np.load(src / f"preprocessed_eeg_{split}.npy", allow_pickle=True).item()
        assert list(d["ch_names"]) == CH_NAMES
        assert np.allclose(d["times"], TIMES)
        x = d.pop("preprocessed_eeg_data")
        if split == "training":
            np.save(out / "train_avg.npy", x.mean(axis=1, dtype=np.float64).astype(np.float32))
        else:
            np.save(out / "test_avg.npy", x.mean(axis=1, dtype=np.float64).astype(np.float32))
        del x, d
    (out / "info.json").write_text(json.dumps({"ch_names": CH_NAMES, "times": TIMES.tolist()}))
    if not keep_raw:
        shutil.rmtree(src)
    return out


def load_subject(sub: int):
    """Memory-mapped (train, test) arrays for one subject."""
    d = PROC / f"sub-{sub:02d}"
    train = np.load(d / "train_avg.npy", mmap_mode="r")
    test = np.load(d / "test_avg.npy", mmap_mode="r")
    return train, test


def metadata() -> dict:
    return np.load(RAW / "image_metadata.npy", allow_pickle=True).item()


def image_paths(split: str) -> list[Path]:
    m = metadata()
    if split == "train":
        return [RAW / "training_images" / c / f for c, f in zip(m["train_img_concepts"], m["train_img_files"])]
    return [RAW / "test_images" / c / f for c, f in zip(m["test_img_concepts"], m["test_img_files"])]


def concept_names(split: str) -> list[str]:
    m = metadata()
    key = "train_img_concepts" if split == "train" else "test_img_concepts"
    return [c.split("_", 1)[1].replace("_", " ") for c in m[key]]


def main():
    ap = argparse.ArgumentParser(description="Download and prepare THINGS-EEG2 preprocessed data")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--images", action="store_true", help="also download the THINGS-EEG2 image set (~0.66 GB)")
    a = ap.parse_args()
    if a.images:
        download_images()
    for s in a.subjects:  # one subject at a time to bound memory
        download_subject(s)
        print("prepared", prepare_subject(s), flush=True)


if __name__ == "__main__":
    main()

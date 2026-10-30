"""Compute CLIP image embeddings for the THINGS-EEG2 images, once."""

import argparse

import numpy as np
import torch
from PIL import Image

from .data import DATA, image_paths

EMB = DATA / "embeddings"


def emb_path(split: str, model: str = "ViT-B-32", pretrained: str = "openai"):
    return EMB / f"clip_{model}_{pretrained}_{split}.npy"


def main():
    import open_clip

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    model = model.eval().to(dev)
    EMB.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        out = emb_path(split, a.model, a.pretrained)
        if out.exists():
            continue
        paths, feats = image_paths(split), []
        with torch.no_grad():
            for i in range(0, len(paths), a.batch):
                x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + a.batch]]).to(dev)
                f = model.encode_image(x).float()
                feats.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
                if i % (a.batch * 40) == 0:
                    print(split, i, len(paths), flush=True)
        np.save(out, np.concatenate(feats).astype(np.float32))
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()

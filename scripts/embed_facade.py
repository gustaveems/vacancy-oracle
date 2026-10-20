#!/usr/bin/env python3
"""Embed facade-aimed imagery with the same local SigLIP model as embed_zero_shot.py.
Writes data/embeddings_facade.npz (ids = image stems: <vbo>, <vbo>_l, <vbo>_r).
Separate from data/embeddings.npz so the 4-heading baselines stay reproducible.
Resumable: images already in the file are skipped."""
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/embeddings_facade.npz"
MODEL = "google/siglip-base-patch16-224"

have = {}
if OUT.exists():
    old = np.load(OUT, allow_pickle=True)
    have = {str(i): f for i, f in zip(old["ids"], old["feats"])}
files = [f for f in sorted(glob.glob(str(ROOT / "data/imagery_facade/*.jpg"))) if Path(f).stem not in have]
print(f"{len(files)} to embed, {len(have)} cached", flush=True)
if files:
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    for i in range(0, len(files), 32):
        batch = files[i:i + 32]
        x = proc(images=[Image.open(f).convert("RGB") for f in batch], return_tensors="pt")
        with torch.no_grad():
            im = model.get_image_features(**x)
        im = (im / im.norm(dim=-1, keepdim=True)).numpy().astype(np.float16)
        for f, v in zip(batch, im):
            have[Path(f).stem] = v
        if (i // 32) % 10 == 0:
            print(f"  {min(i + 32, len(files))}/{len(files)}", flush=True)
    np.savez_compressed(OUT, ids=np.array(list(have.keys())), feats=np.array(list(have.values())))
print(f"embeddings_facade: {len(have)} images → {OUT.relative_to(ROOT)}")

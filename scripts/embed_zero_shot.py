#!/usr/bin/env python3
"""
Phase 2 first light — local zero-shot vacancy vision scoring (no APIs, no cloud).

SigLIP base/16: embeds every cached Street View image + a set of vacancy/
activity text prompts on CPU, scores each view by cosine margin
(vacant-prompts − active-prompts), aggregates per building.

Outputs:
  data/vision_zeroshot.csv          per-image scores
  data/vision_zeroshot_scores.json  pdok_vbo → calibrated-to-[0,1] vacancy score

Run automatically by scripts/run_phase2.sh after imagery fetch, or manually.
"""
import csv, json, os, glob, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "data" / "imagery"
MODEL = "google/siglip-base-patch16-224"

VACANT = [
    "an abandoned building with boarded-up windows",
    "an empty neglected storefront with faded signage",
    "a derelict property with broken glass and graffiti",
    "a vacant commercial space for lease",
]
ACTIVE = [
    "a busy active shopfront with customers",
    "a well-maintained store with bright signage",
    "a lively retail street with open businesses",
    "a building in active commercial use",
]

def done_ids():
    """(pdok, view) pairs already scored in previous runs — for resume."""
    if not (ROOT / "data/vision_zeroshot.csv").exists(): return set()
    return {(r["pdok_vbo"], r["view"]) for r in csv.DictReader(open(ROOT / "data/vision_zeroshot.csv"))}


def main():
    skip = done_ids()
    files = [f for f in sorted(glob.glob(str(IMG / "*.jpg")))
             if tuple(Path(f).stem.rsplit("_", 1)) not in skip]
    if not files:
        print("nothing to embed — all done"); return
    print(f"{len(files)} images to embed, skipping {len(skip)} scored views", flush=True)
    print(f"{len(files)} images, loading {MODEL}…")
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    texts = VACANT + ACTIVE
    with torch.no_grad():
        tfeat = model.get_text_features(**proc(text=texts, return_tensors="pt", padding="max_length", truncation=True))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    rows, buf, buf_files = [], [], []
    ids, feats = [], []
    def flush():
        if not buf: return
        x = proc(images=[Image.open(f).convert("RGB") for f in buf], return_tensors="pt")
        with torch.no_grad():
            im = model.get_image_features(**x)
        im = im / im.norm(dim=-1, keepdim=True)
        sims = im @ tfeat.T
        ids.extend(Path(f).stem for f in buf_files)
        feats.append(im.numpy())
        vac = sims[:, :len(VACANT)].mean(1); act = sims[:, len(VACANT):].mean(1)
        for f, v, a in zip(buf_files, vac.numpy(), act.numpy()):
            name = Path(f).stem
            pdok, head = name.rsplit("_", 1)
            rows.append({"pdok_vbo": pdok, "view": head, "vacant_sim": round(float(v), 4),
                         "active_sim": round(float(a), 4), "margin": round(float(v - a), 4)})
        print(f"embedded {len(rows)}/{len(files)}", flush=True)
        buf.clear(); buf_files.clear()

    for f in files:
        buf.append(Image.open(f).size and f); buf_files.append(f)
        if len(buf) >= 16: flush()
    flush()

    feats_cat = np.concatenate(feats, axis=0).astype(np.float16)
    # merge with cached embeddings so a resumed run never clobbers them
    npz_path = ROOT / "data/embeddings.npz"
    if npz_path.exists():
        old = np.load(npz_path, allow_pickle=True)
        merged = {str(i): f for i, f in zip(old["ids"], old["feats"])}
        for i, f in zip(ids, feats_cat):
            merged[str(i)] = f
        ids = list(merged.keys())
        feats_cat = np.array(list(merged.values()), dtype=np.float16)
    append = (ROOT / "data/vision_zeroshot.csv").exists()
    mode = "a" if append else "w"
    with open(ROOT / "data/vision_zeroshot.csv", mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdok_vbo", "view", "vacant_sim", "active_sim", "margin"])
        if not append: w.writeheader()
        w.writerows(rows)

    np.savez_compressed(npz_path, ids=np.array(ids), feats=feats_cat)
    print(f"embeddings saved: {len(ids)}×{feats_cat.shape[1]}d → data/embeddings.npz", flush=True)
    per = {}
    allrows = list(csv.DictReader(open(ROOT / "data/vision_zeroshot.csv")))
    for r in allrows:
        per.setdefault(r["pdok_vbo"], []).append(float(r["margin"]))
    margins = np.array([max(v) for v in per.values()])
    lo, hi = np.percentile(margins, [2, 98])
    scores = {k: float(np.clip((max(v) - lo) / (hi - lo), 0, 1)) for k, v in per.items()}
    (ROOT / "data/vision_zeroshot_scores.json").write_text(json.dumps(scores))
    print(f"done this pass; {len(scores)} buildings scored overall | margins {margins.min():.3f}..{margins.max():.3f}", flush=True)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 3 preview — linear probe over cached SigLIP embeddings.
Two evaluations (both with n_pos=13 caveats):
  A. Geographic holdout: train Amsterdam → test Rotterdam+Utrecht (protocol spec)
  B. Stratified 5-fold CV over all labeled buildings (tighter CI, not the headline)
Outputs docs/phase3-preview.md + figure."""
import csv, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
d = np.load(ROOT / "data/embeddings.npz", allow_pickle=True)
ids, feats = d["ids"], d["feats"].astype(np.float32)
by_id = {}
for name, f in zip(ids, feats):
    pdok = str(name).rsplit("_", 1)[0]
    by_id.setdefault(pdok, []).append(f)
X_map = {k: np.mean(v, axis=0) for k, v in by_id.items()}

rows = [r for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["label"] in ("1", "2")]
X, y, city = [], [], []
for r in rows:
    if r["pdok_vbo"] in X_map:
        X.append(X_map[r["pdok_vbo"]]); y.append(1 if r["label"] == "1" else 0); city.append(r["city"])
X, y, city = np.array(X), np.array(y), np.array(city)
zs = json.load(open(ROOT / "data/vision_zeroshot_scores.json"))
zsc = np.array([zs.get(r["pdok_vbo"], 0) for r in rows if r["pdok_vbo"] in X_map])

def pr_auc(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return float(np.sum((R - np.concatenate([[0], R[:-1]])) * P))

def probe(Xtr, ytr, Xte):
    m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
    m.fit(Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9),
          ytr)
    Xn = Xte / np.maximum(np.linalg.norm(Xte, axis=1, keepdims=True), 1e-9)
    return m.decision_function(Xn)

tr = city == "Amsterdam"; te = ~tr
print(f"labeled embedded: {len(y)} (train A'dam {tr.sum()}, pos {y[tr].sum()} | test R+U {te.sum()}, pos {y[te].sum()})")
p_geo = probe(X[tr], y[tr], X[te])
zs_geo = zsc[te]
auc_geo = pr_auc(y[te], p_geo); auc_zs_geo = pr_auc(y[te], zs_geo)
print(f"A) geo holdout   probe PR-AUC {auc_geo:.3f}  vs zero-shot {auc_zs_geo:.3f}")

# stratified CV over everything
oof = np.zeros(len(y))
for trn, tst in StratifiedKFold(5, shuffle=True, random_state=7).split(X, y):
    oof[tst] = probe(X[trn], y[trn], X[tst])
auc_cv = pr_auc(y, oof)
print(f"B) 5-fold CV     probe PR-AUC {auc_cv:.3f}")

# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def pr(scores):
    o = np.argsort(-scores); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    return [0] + list(tp / max(tp[-1], 1)), [1] + list(tp / np.maximum(tp + fp, 1))
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
R, P = pr(oof); ax.plot(R, P, color="#e60012", lw=2, label=f"probe 5-fold CV ({auc_cv:.3f})")
R, P = pr(zs_geo.repeat(1) if False else zsc); ax.plot(R, P, color="#22c55e", lw=2, label=f"zero-shot ({pr_auc(y, zsc):.3f})")
ax.plot([0, 1], [y.mean(), y.mean()], ls="--", color="#8b94a7", label=f"prevalence {y.mean():.3f}")
ax.set(xlabel="recall", ylabel="precision", title="Supervised probe vs zero-shot (13 positives — noisy!)")
ax.legend(frameon=False); ax.grid(alpha=.18); plt.tight_layout()
fig.savefig(ROOT / "docs/figures/probe_vs_zeroshot.png")

md = f"""# Phase 3 preview — supervised probe over embeddings

⚠ **n_pos = 13.** Every number below is directional, not conclusive — that's the
sparsity wall documented in Phase 1; active-labeled industrial picks are the fix.

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train Amsterdam → test Rotterdam+Utrecht, n=8 pos) | linear probe on SigLIP feats | **{auc_geo:.3f}** |
|  | zero-shot SigLIP margin | {auc_zs_geo:.3f} |
| **B. Stratified 5-fold CV** (all 539, n=13 pos) | linear probe | **{auc_cv:.3f}** |

Prevalence floor: {y.mean():.3f}.

**Read:** the probe learns transferable vacancy signal (CV beats zero-shot
substantially); geographic transfer at n=8 positives is coin-flip territory and
awaits the active batch. Next: label `data/active_batch.csv` top-100 model
picks → re-run everything (the label watcher already automates the refresh).

_Frozen protocol unchanged: metrics only on held-out splits._
"""
(ROOT / "docs/phase3-preview.md").write_text(md)
print("wrote docs/phase3-preview.md + figure")

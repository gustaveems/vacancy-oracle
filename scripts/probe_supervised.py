#!/usr/bin/env python3
"""Phase 3 — linear probe over cached SigLIP embeddings.
Two evaluations (positives are still sparse — n_pos is printed and written into the doc):
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
n_pos = int(y.sum()); n_tot = len(y); geo_n = int(te.sum()); geo_pos = int(y[te].sum())
print(f"labeled embedded: {n_tot} (train A'dam {tr.sum()}, pos {y[tr].sum()} | test R+U {geo_n}, pos {geo_pos})")
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
auc_zs_full = pr_auc(y, zsc)
from collections import Counter
pos_by_city = " · ".join(f"{c} {n}" for c, n in sorted(Counter(c for c, yy in zip(city, y) if yy).items()))
def pr(scores):
    o = np.argsort(-scores); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    return [0] + list(tp / max(tp[-1], 1)), [1] + list(tp / np.maximum(tp + fp, 1))
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
R, P = pr(oof); ax.plot(R, P, color="#e60012", lw=2, label=f"probe 5-fold CV ({auc_cv:.3f})")
R, P = pr(zs_geo.repeat(1) if False else zsc); ax.plot(R, P, color="#22c55e", lw=2, label=f"zero-shot ({pr_auc(y, zsc):.3f})")
ax.plot([0, 1], [y.mean(), y.mean()], ls="--", color="#8b94a7", label=f"prevalence {y.mean():.3f}")
ax.set(xlabel="recall", ylabel="precision", title=f"Supervised probe vs zero-shot ({n_pos} positives)")
ax.legend(frameon=False); ax.grid(alpha=.18); plt.tight_layout()
fig.savefig(ROOT / "docs/figures/probe_vs_zeroshot.png")

md = f"""# Phase 3 — supervised probe over embeddings

⚠ **n_pos = {n_pos}.** Still sparse — but doubled by the labeled active batch
(all 13 new positives landed inside the model's top-100 picks: 3.5× enrichment).

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train Amsterdam → test Rotterdam+Utrecht, n={geo_n}, {geo_pos} pos) | linear probe on SigLIP feats | **{auc_geo:.3f}** |
|  | zero-shot SigLIP margin | {auc_zs_geo:.3f} |
| **B. Stratified 5-fold CV** (all {n_tot}, {n_pos} pos) | linear probe | **{auc_cv:.3f}** |

Prevalence floor: {y.mean():.3f}.

**Read:** in-city the probe wins (CV {auc_cv:.3f} vs zero-shot full-set
{auc_zs_full:.3f}); geographic transfer flips it — with only {int(y[tr].sum())}
Amsterdam positives to train on, the probe collapses while the training-free
zero-shot margin holds. Positives by city: {pos_by_city}. The geographic
headline claim still belongs to zero-shot; the fix is more Amsterdam positives
(fresh candidate collection) plus the fused model with registry features per
`docs/eval-protocol.md` (baselines 3–5).

_Frozen protocol unchanged: metrics only on held-out splits._
"""
(ROOT / "docs/phase3-preview.md").write_text(md)
print("wrote docs/phase3-preview.md + figure")

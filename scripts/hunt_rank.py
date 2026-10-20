#!/usr/bin/env python3
"""Amsterdam positive hunt — step 2: rank new candidates for labeling.
1. Facade probe vs 4-heading probe on the same labeled buildings, building-grouped
   5-fold CV over 5 seeds + geographic holdout (post-hoc; protocol of baseline 7).
2. Probe refit on all labels, applied to unlabeled Amsterdam candidates whose
   facade photo is official Google imagery captured 2022 or later (photo-age
   control from baseline 7) → data/active_batch.csv (top 100, studio format)
   and data/hunt_scores.csv (all scored).
Writes data/hunt_results.json."""
import csv, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent

def per_building(npz, key):
    d = np.load(ROOT / npz, allow_pickle=True); out = {}
    for name, f in zip(d["ids"], d["feats"].astype(np.float32)):
        out.setdefault(key(str(name)), []).append(f)
    return {k: np.mean(v, axis=0) for k, v in out.items()}

FAC = per_building("data/embeddings_facade.npz", lambda s: s.split("_")[0])
OLD = per_building("data/embeddings.npz", lambda s: s.rsplit("_", 1)[0])
reg = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/registry_features.csv"))}
meta = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/imagery_facade/manifest.csv"))}
labels = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/labels.csv"))}
cands = list(csv.DictReader(open(ROOT / "data/candidates.csv")))

rows = [r for r in labels.values() if r["label"] in ("1", "2") and r["pdok_vbo"] in FAC and r["pdok_vbo"] in OLD]
vbos = [r["pdok_vbo"] for r in rows]
y = np.array([1 if r["label"] == "1" else 0 for r in rows])
city = np.array([r["city"] for r in rows])
groups = np.array([reg.get(v, {}).get("pand_id") or f"vbo:{v}" for v in vbos])
Xf = np.array([FAC[v] for v in vbos]); Xo = np.array([OLD[v] for v in vbos])

nrm = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
LR = lambda: LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)

TIE = np.random.default_rng(11)
def pr_auc(y, s):
    perm = TIE.permutation(len(s)); o = perm[np.argsort(-s[perm], kind="stable")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return float(np.sum((R - np.concatenate([[0], R[:-1]])) * P))

def grouped_cv(X):
    vals = []
    for seed in (7, 11, 23, 42, 99):
        s = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            m = LR(); m.fit(nrm(X[tr]), y[tr]); s[te] = m.decision_function(nrm(X[te]))
        vals.append(pr_auc(y, s))
    return {"mean": round(float(np.mean(vals)), 3), "min": round(float(np.min(vals)), 3), "max": round(float(np.max(vals)), 3)}

def geo(X):
    a = city == "Amsterdam"
    m = LR(); m.fit(nrm(X[a]), y[a]); g = pr_auc(y[~a], m.decision_function(nrm(X[~a])))
    m = LR(); m.fit(nrm(X[~a]), y[~a]); r = pr_auc(y[a], m.decision_function(nrm(X[a])))
    return {"geo_test_RU": round(g, 3), "rotated_test_Adam": round(r, 3)}

res = {"labeled_with_both_views": {"units": int(len(y)), "pos": int(y.sum()),
                                   "buildings": int(len(np.unique(groups))), "pos_buildings": int(len(np.unique(groups[y == 1])))},
       "probe_facade": {"grouped_cv": grouped_cv(Xf), **geo(Xf)},
       "probe_4heading": {"grouped_cv": grouped_cv(Xo), **geo(Xo)},
       "prevalence": round(float(y.mean()), 3)}

# ---- rank unlabeled Amsterdam candidates ----
use = "probe_facade" if res["probe_facade"]["grouped_cv"]["mean"] >= res["probe_4heading"]["grouped_cv"]["mean"] else "probe_4heading"
X_all = Xf if use == "probe_facade" else Xo
EMB = FAC if use == "probe_facade" else OLD
final = LR(); final.fit(nrm(X_all), y)

pool = []
for c in cands:
    v = c["pdok_vbo"]; m = meta.get(v, {})
    if c["city"] != "Amsterdam" or v in labels or v not in EMB: continue
    if m.get("status") != "ok": continue
    year = int(m["pano_date"][:4]) if m.get("pano_date", "")[:4].isdigit() else 0
    score = float(final.decision_function(nrm(EMB[v][None]))[0])
    pool.append({"pdok_vbo": v, "address": c["address"], "pano_date": m.get("pano_date", ""),
                 "recent": year >= 2022, "score": score})
pool.sort(key=lambda r: -r["score"])
with open(ROOT / "data/hunt_scores.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pdok_vbo", "address", "pano_date", "recent_photo", "probe_score"])
    for r in pool: w.writerow([r["pdok_vbo"], r["address"], r["pano_date"], int(r["recent"]), round(r["score"], 4)])
batch = [r for r in pool if r["recent"]][:100]
lo, hi = (min(r["score"] for r in pool), max(r["score"] for r in pool)) if pool else (0, 1)
with open(ROOT / "data/active_batch.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rank", "pdok_vbo", "vision_vacancy", "city", "address"])
    for i, r in enumerate(batch, 1):
        w.writerow([i, r["pdok_vbo"], round((r["score"] - lo) / max(hi - lo, 1e-9), 3), "Amsterdam", r["address"]])
res["hunt"] = {"ranker": use, "scored_unlabeled_amsterdam": len(pool),
               "with_recent_photo": sum(r["recent"] for r in pool), "batch": len(batch)}
(ROOT / "data/hunt_results.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))

#!/usr/bin/env python3
"""Phase 4 — Oracle v2 demo scores for every candidate.
Refits the baseline-5 fused stack (probe + HistGradientBoosting over registry
features | zero-shot margin | probe score) on ALL usable labels and scores
all 1,050 candidates for the demo. These ranks are deployment-style
(in-sample for the 843 trained rows, out-of-sample for unclear re-passes);
protocol metrics live only in docs/baseline5-fusion.md held-out splits.
Output: data/oracle_scores.csv (pdok_vbo, oracle, rules, zs, label)."""
import csv, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent

d = np.load(ROOT / "data/embeddings.npz", allow_pickle=True)
by_id = {}
for name, f in zip(d["ids"], d["feats"].astype(np.float32)):
    by_id.setdefault(str(name).rsplit("_", 1)[0], []).append(f)
X_map = {k: np.mean(v, axis=0) for k, v in by_id.items()}
zs = json.load(open(ROOT / "data/vision_zeroshot_scores.json"))
rules = json.load(open(ROOT / "data/scores_rules_full.json"))
reg = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/registry_features.csv"))}

USES = ["woonfunctie", "kantoorfunctie", "industriefunctie", "winkelfunctie", "bijeenkomstfunctie",
        "overige gebruiksfunctie", "onderwijsfunctie", "sportfunctie", "logiesfunctie", "gezondheidszorgfunctie"]
FLAGS = USES + ["ligplaats", "standplaats"]

def regfeats(vbo):
    g = reg.get(vbo, {})
    parts = set((g.get("use") or "").split(";"))
    year = float(g["build_year"]) if g.get("build_year") else np.nan
    area = float(g["area"]) if g.get("area") else np.nan
    st = g.get("vbo_status") or ""
    ps = g.get("pand_status") or ""
    return [float(u in parts) for u in FLAGS] + [
        year, area, 2026.0 - year if year == year else np.nan,
        float("Verbouwing" in st), float("gevormd" in st),
        float("Bouwvergunning" in ps), float("Verbouwing" in ps),
    ]

# all candidates with embeddings (the demo universe)
cands = [r["pdok_vbo"] for r in csv.DictReader(open(ROOT / "data/candidates.csv")) if r["pdok_vbo"] in X_map]
labels = {r["pdok_vbo"]: r["label"] for r in csv.DictReader(open(ROOT / "data/labels.csv"))}
Xemb = np.array([X_map[v] for v in cands])
norm = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)

usable = [i for i, v in enumerate(cands) if labels.get(v) in ("1", "2")]
yu = np.array([1 if labels[cands[i]] == "1" else 0 for i in usable])

probe = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
probe.fit(norm(Xemb[usable]), yu)
p_all = probe.decision_function(norm(Xemb))

Xall = np.array([regfeats(v) for v in cands], dtype=float)
zsc = np.array([zs.get(v, 0) for v in cands])
Xfuse = np.column_stack([Xall, zsc, p_all])

gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                     min_samples_leaf=25, l2_regularization=1.0, random_state=7)
gbm.fit(Xfuse[usable], yu)
oracle = gbm.predict_proba(Xfuse)[:, 1]

with open(ROOT / "data/oracle_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["pdok_vbo", "oracle", "rules", "zs", "label"])
    for i, v in enumerate(cands):
        w.writerow([v, round(float(oracle[i]), 4), rules.get(v, ""), round(zsc[i], 4), labels.get(v, "")])
print(f"scored {len(cands)} candidates ({len(usable)} trained rows) → data/oracle_scores.csv")

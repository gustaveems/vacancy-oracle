#!/usr/bin/env python3
"""Phase 3b — protocol baseline 3: registry-only GBM.
Features from data/registry_features.csv (BAG): use indicators (gebruiksdoel),
build year, area, age, VBO/pand status flags. Ownership form is BRK-only
(key-required) and unavailable — documented in the README, frozen protocol
untouched. No image features, no city feature (that is the point of the
geographic holdout). Fixed a-priori hyperparameters, no tuning — at this
positive count tuning would be overfitting theater.
Two evaluations:
  A. Geographic holdout: train Amsterdam → test Rotterdam+Utrecht (protocol spec)
  B. Stratified 5-fold CV over all labeled buildings
Outputs docs/baseline3-registry.md + figure."""
import csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- data ----
reg = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/registry_features.csv"))}
rows = [(r["pdok_vbo"], r["city"], int(r["label"] == "1")) for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["label"] in ("1", "2")]
y = np.array([b for _, _, b in rows])
city = np.array([c for _, c, _ in rows])

USES = ["woonfunctie", "kantoorfunctie", "industriefunctie", "winkelfunctie", "bijeenkomstfunctie",
        "overige gebruiksfunctie", "onderwijsfunctie", "sportfunctie", "logiesfunctie", "gezondheidszorgfunctie"]
FLAGS = USES + ["ligplaats", "standplaats"]

def feats(vbo):
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

X = np.array([feats(v) for v, _, _ in rows], dtype=float)
n_pos = int(y.sum()); n_tot = len(y)

from sklearn.ensemble import HistGradientBoostingClassifier
gbm = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                     min_samples_leaf=25, l2_regularization=1.0, random_state=7)

def pr_auc(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return float(np.sum((R - np.concatenate([[0], R[:-1]])) * P))

def p_at_r(y, s, r_min):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    ok = P[R >= r_min]
    return float(ok.max()) if len(ok) else 0.0

# ---- A. geographic holdout ----
tr = city == "Amsterdam"; te = ~tr
gbm.fit(X[tr], y[tr])
auc_geo = pr_auc(y[te], gbm.predict_proba(X[te])[:, 1])

# ---- B. stratified 5-fold CV ----
from sklearn.model_selection import StratifiedKFold, cross_val_predict
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
pcv = cross_val_predict(gbm, X, y, cv=cv, method="predict_proba")[:, 1]
auc_cv = pr_auc(y, pcv)
p50 = p_at_r(y, pcv, 0.5); p70 = p_at_r(y, pcv, 0.7)
brier = float(np.mean((pcv - y) ** 2))
brier_maj = float(y.mean() * (1 - y.mean()))

n_full_pos = int(sum(1 for v, _, b in rows if b and (reg.get(v, {}).get("use") not in ("", None)) and reg[v]["use"] not in ("ligplaats", "standplaats")))
tr_n, te_n = int(tr.sum()), int(te.sum())
tr_pos, te_pos = int(y[tr].sum()), int(y[te].sum())
print(f"n={n_tot} ({n_pos} pos, {n_full_pos} with full VBO features) | CV PR-AUC {auc_cv:.3f} | geo PR-AUC {auc_geo:.3f} (train A'dam {tr_pos} pos → test R+U {te_pos} pos)")

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def pr_curve(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return R, P

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
R, P = pr_curve(y, pcv); ax.plot(R, P, label=f"5-fold CV (PR-AUC {auc_cv:.3f})")
R, P = pr_curve(y[te], gbm.predict_proba(X[te])[:, 1]); ax.plot(R, P, label=f"geo holdout (PR-AUC {auc_geo:.3f})")
ax.axhline(y.mean(), ls="--", c="gray", label=f"majority floor {y.mean():.3f}")
ax.set(xlabel="recall", ylabel="precision", title=f"Baseline 3 — registry-only GBM ({n_pos} positives)")
ax.legend(); fig.tight_layout()
fig.savefig(ROOT / "docs/figures/baseline3_pr.png")

# ---- doc ----
md = f"""# Baseline 3 — registry-only GBM (BAG features)

⚠ **n_pos = {n_pos}** ({n_full_pos} with full VBO features; the rest are
ligplaats/standplaats or ended registrations). Sparse-positive regime — the
same caveat as every baseline so far.

Features: use indicators (gebruiksdoel, {len(FLAGS)} flags), build year, area,
age, VBO/pand status flags (verbouwing / gevormd / bouwvergunning).
Ownership form is BRK-only and key-required → not included (README documents
this; frozen `docs/eval-protocol.md` untouched). No image features, no city
feature. Fixed a-priori hyperparameters, no tuning (HistGradientBoosting:
150 trees, lr 0.08, ≤8 leaves, min 25/leaf, L2 1.0).

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train Amsterdam n={tr_n}, {tr_pos} pos → test R+U n={te_n}, {te_pos} pos) | registry GBM | **{auc_geo:.3f}** |
| **B. Stratified 5-fold CV** (all {n_tot}, {n_pos} pos) | registry GBM | **{auc_cv:.3f}** |

Reference on the same {n_tot}-label set: zero-shot full 0.082 / geo 0.126,
probe CV 0.390 / geo 0.024 (`docs/phase3-preview.md`). Majority floor
{y.mean():.3f}.

Secondary (CV): precision@recall 0.5 = {p50:.2f}, 0.7 = {p70:.2f}; Brier
{brier:.3f} (majority no-skill {brier_maj:.3f}). Permutation importances
omitted — at {n_pos} positives they are noise, not signal.

**Read:** in-city, BAG metadata alone is a real signal — CV {auc_cv:.3f} is
on par with the image probe (0.390) and well above zero-shot (0.082); at
{n_pos} positives the gap to the probe is noise, call it parity. Geographic
transfer is the same wall as every supervised model here: {auc_geo:.3f}
with {tr_pos} Amsterdam training positives, below the training-free
zero-shot (0.126). The registry's role is fusion input (baseline 5), not
standalone detection — statuses like *verbouwing*/*gevormd*/*bouwvergunning*
mark transition states that images see directly, so the two views should
complement rather than compete. Calibration and FP-cost framing deferred to
the fused model per protocol.

_Frozen protocol unchanged: metrics only on held-out splits._
"""
(ROOT / "docs/baseline3-registry.md").write_text(md)
print(f"wrote docs/baseline3-registry.md + docs/figures/baseline3_pr.png")

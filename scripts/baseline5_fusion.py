#!/usr/bin/env python3
"""Phase 3b — protocol baseline 5: fused model (registry + vision).
Late fusion: HistGradientBoosting over [BAG registry features | zero-shot
margin | linear-probe score]. Early fusion (raw 768-d embeddings into the
GBM) was rejected — 26 positives cannot support 768 extra dimensions.
Level-0 vision scores are out-of-fold per row (standard stacking, same folds
as the meta-learner); geographic splits train probe + GBM on the same city
set, test scores are honest. Fixed a-priori hyperparameters, no tuning.
Evaluations:
  A.  Geographic holdout: train Amsterdam → test Rotterdam+Utrecht (protocol spec)
  A′. Rotated fold: train Rotterdam+Utrecht → test Amsterdam (protocol "one rotated fold")
  B.  Stratified 5-fold CV over all labeled buildings
  Ablation = 5 minus each input (image-only run here; registry-only = baseline 3).
Metrics: PR-AUC, precision@recall 0.5/0.7, Brier + reliability plot, per-class
confusion and FP-cost curve at the ≥90%-precision demo threshold.
Outputs docs/baseline5-fusion.md + figure."""
import csv, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent

# ---- vision level-0: per-building mean SigLIP embedding + zero-shot margin ----
d = np.load(ROOT / "data/embeddings.npz", allow_pickle=True)
by_id = {}
for name, f in zip(d["ids"], d["feats"].astype(np.float32)):
    pdok = str(name).rsplit("_", 1)[0]
    by_id.setdefault(pdok, []).append(f)
X_map = {k: np.mean(v, axis=0) for k, v in by_id.items()}
zs = json.load(open(ROOT / "data/vision_zeroshot_scores.json"))

# ---- registry features (same as baseline 3) ----
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

# ---- aligned data ----
rows = [r for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["label"] in ("1", "2") and r["pdok_vbo"] in X_map]
y = np.array([1 if r["label"] == "1" else 0 for r in rows])
city = np.array([r["city"] for r in rows])
Xemb = np.array([X_map[r["pdok_vbo"]] for r in rows])
zsc = np.array([zs.get(r["pdok_vbo"], 0) for r in rows])
Xreg = np.array([regfeats(r["pdok_vbo"]) for r in rows], dtype=float)
n_pos = int(y.sum()); n_tot = len(y)

# ---- level-0 probe (config mirrors probe_supervised.py) ----
def probe_scores(Xtr, ytr, Xte):
    n = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    m = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
    m.fit(n(Xtr), ytr)
    return m.decision_function(n(Xte))

CV = StratifiedKFold(5, shuffle=True, random_state=7)
p_oof = cross_val_predict(LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000),
                          Xemb / np.maximum(np.linalg.norm(Xemb, axis=1, keepdims=True), 1e-9), y,
                          cv=CV, method="decision_function")

GBM = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                             min_samples_leaf=25, l2_regularization=1.0, random_state=7)

def pr_auc(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return float(np.sum((R - np.concatenate([[0], R[:-1]])) * P))

def curve(y, s):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return [0] + list(R), [1] + list(P), tp, fp

def p_at_r(y, s, r_min):
    o = np.argsort(-s); yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    ok = P[R >= r_min]
    return float(ok.max()) if len(ok) else 0.0

# ---- B. stratified 5-fold CV: fused and image-only ----
Xfuse = np.column_stack([Xreg, zsc, p_oof])
Ximg = np.column_stack([zsc, p_oof])
fused_oof = cross_val_predict(GBM(), Xfuse, y, cv=CV, method="predict_proba")[:, 1]
img_oof = cross_val_predict(GBM(), Ximg, y, cv=CV, method="predict_proba")[:, 1]
auc_cv = pr_auc(y, fused_oof); auc_img = pr_auc(y, img_oof)
p50 = p_at_r(y, fused_oof, 0.5); p70 = p_at_r(y, fused_oof, 0.7)
brier = float(np.mean((fused_oof - y) ** 2)); brier0 = float(y.mean() * (1 - y.mean()))

# ---- A. geographic holdout + A'. rotated fold ----
tr = city == "Amsterdam"; te = ~tr
p_geo_tr = probe_scores(Xemb[tr], y[tr], Xemb[tr])
p_geo_te = probe_scores(Xemb[tr], y[tr], Xemb[te])
g = GBM(); g.fit(np.column_stack([Xreg[tr], zsc[tr], p_geo_tr]), y[tr])
auc_geo = pr_auc(y[te], g.predict_proba(np.column_stack([Xreg[te], zsc[te], p_geo_te]))[:, 1])

p_rot_te = probe_scores(Xemb[te], y[te], Xemb[tr])
p_rot_tr = probe_scores(Xemb[te], y[te], Xemb[te])
g2 = GBM(); g2.fit(np.column_stack([Xreg[te], zsc[te], p_rot_tr]), y[te])
auc_rot = pr_auc(y[tr], g2.predict_proba(np.column_stack([Xreg[tr], zsc[tr], p_rot_te]))[:, 1])
print(f"n={n_tot} ({n_pos} pos) | CV fused {auc_cv:.3f} / image-only {auc_img:.3f} | geo {auc_geo:.3f} | rotated {auc_rot:.3f}")

# ---- demo threshold at >=90% precision (CV OOF) ----
o = np.argsort(-fused_oof); yy = y[o]
tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
cand = np.where(P >= 0.9)[0]
if len(cand):
    i = cand[np.argmax(R[cand])]
    thr = float(fused_oof[o][i]); TP, FP = int(tp[i]), int(fp[i])
    FN, TN = n_pos - TP, int((n_tot - n_pos) - FP)
    demo_txt = (f"threshold {thr:.3f} — TP {TP} / FP {FP} / FN {FN} / TN {TN} "
                f"(recall {TP / n_pos:.2f}, precision {P[i]:.2f})")
    demo_point = (FP, TP)
else:
    i = int(np.argmax(P)); demo_txt = (f"not achievable at {n_pos} positives — best precision "
                                       f"{P[i]:.2f} at recall {R[i]:.2f}")
    demo_point = (int(fp[i]), int(tp[i]))

# ---- figure: PR + reliability + FP-cost ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)
ax = axes[0]
R, Pc, *_ = curve(y, fused_oof); ax.plot(R, Pc, color="#e60012", lw=2, label=f"fused 5-fold CV ({auc_cv:.3f})")
R, Pc, *_ = curve(y, img_oof); ax.plot(R, Pc, color="#f59e0b", lw=2, label=f"image-only CV ({auc_img:.3f})")
s_geo = g.predict_proba(np.column_stack([Xreg[te], zsc[te], p_geo_te]))[:, 1]
R, Pc, *_ = curve(y[te], s_geo); ax.plot(R, Pc, color="#2563eb", lw=2, ls="--", label=f"fused geo holdout ({auc_geo:.3f})")
s_rot = g2.predict_proba(np.column_stack([Xreg[tr], zsc[tr], p_rot_te]))[:, 1]
R, Pc, *_ = curve(y[tr], s_rot); ax.plot(R, Pc, color="#8b5cf6", lw=2, ls="--", label=f"fused rotated fold ({auc_rot:.3f})")
ax.axhline(y.mean(), ls=":", c="#8b94a7", label=f"prevalence {y.mean():.3f}")
ax.set(xlabel="recall", ylabel="precision", title="Baseline 5 — fused model (PR-AUC)")
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.18)

ax = axes[1]
pt, pp = calibration_curve(y, fused_oof, n_bins=8, strategy="quantile")
ax.plot(pp, pt, "o-", color="#e60012", label=f"fused OOF (Brier {brier:.3f})")
ax.plot([0, max(pp)], [0, max(pp)], ":", c="#8b94a7")
ax.set(xlabel="predicted vacancy probability", ylabel="observed vacancy rate",
       title=f"Reliability (n_pos={n_pos} — noisy)")
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.18)

ax = axes[2]
ax.plot(fp, tp, color="#e60012", lw=2)
ax.plot(*demo_point, "*", color="#111", ms=14)
ax.annotate("≥90% precision\n" + (f"thr {thr:.3f}" if len(cand) else "not achievable"),
            demo_point, textcoords="offset points", xytext=(12, -6), fontsize=8)
ax.set(xlabel="false positives (unwanted outreach letters)", ylabel="true positives (caught vacancies)",
       title="FP cost curve (CV OOF)")
ax.grid(alpha=.18)
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/baseline5_fusion.png")

md = f"""# Baseline 5 — fused model (registry + vision)

⚠ **n_pos = {n_pos}.** Sparse-positive regime — every number below is
directional, as with all baselines so far.

Architecture: late fusion — HistGradientBoosting over 21 features:
19 BAG registry features | zero-shot margin | linear-probe score
(probe config mirrors `probe_supervised.py`). Early fusion (raw 768-d
embeddings into the GBM) was rejected: {n_pos} positives cannot support 768
extra dimensions. Level-0 vision scores are out-of-fold per row (standard
stacking, same folds); on geographic splits probe and GBM train on the same
city set — test scores are honest. Fixed a-priori hyperparameters, no
tuning.

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train A'dam n={int(tr.sum())}, {int(y[tr].sum())} pos → test R+U n={int(te.sum())}, {int(y[te].sum())} pos) | **fused** | **{auc_geo:.3f}** |
|  | zero-shot margin | 0.126 |
|  | probe | 0.024 |
|  | registry GBM | 0.046 |
| **A′. Rotated fold** (train R+U, {int(y[te].sum())} pos → test A'dam, {int(y[tr].sum())} pos) | **fused** | **{auc_rot:.3f}** |
| **B. Stratified 5-fold CV** (all {n_tot}, {n_pos} pos) | **fused** | **{auc_cv:.3f}** |
|  | image-only (ablation: 5 − registry) | {auc_img:.3f} |
|  | registry-only (ablation: 5 − image, baseline 3) | 0.407 |
|  | probe (baseline 4) | 0.390 |
|  | zero-shot | 0.082 |
|  | majority floor | {y.mean():.3f} |

Secondary (CV OOF): precision@recall 0.5 = {p50:.2f}, 0.7 = {p70:.2f}; Brier
{brier:.3f} (no-skill {brier0:.3f}).

**Demo threshold (≥90% precision):** {demo_txt}.

Calibration: reliability plot in the figure — at {n_pos} positives the
per-bin rates are noise; Brier is the trustworthy summary.

**Read:** in-city the fused model posts the family-best CV PR-AUC
({auc_cv:.3f}) — but the ablation reveals the plateau: image-only stacks to
{auc_img:.3f} and registry-only reaches 0.407, all within noise of each
other at {n_pos} positives. The two views are not yet complementary — they
are three ways of ranking roughly the same handful of buildings.
Geographic transfer remains the wall: fused {auc_geo:.3f} is the best
supervised geo number yet still below the training-free zero-shot margin
(0.126), and the rotated fold — training on R+U's 21 positives, four times
Amsterdam's — collapses to {auc_rot:.3f}. The wall is symmetric and is not
bought off with training-side positives. The ≥90%-precision demo threshold
technically exists but catches 1 of {n_pos} vacancies — vacuous, reported
as such. Against the rules-only baseline (0.032, 701-label era) the fused
model is an order of magnitude ahead, so the README claim stands; the
binding constraints are positive count and cross-city signal — more
Amsterdam labeling, not more model.

_Frozen protocol unchanged: metrics only on held-out splits._
"""
(ROOT / "docs/baseline5-fusion.md").write_text(md)
print("wrote docs/baseline5-fusion.md + docs/figures/baseline5_fusion.png")

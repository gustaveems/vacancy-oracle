#!/usr/bin/env python3
"""Baseline 7 — audit of the in-city numbers, prompted by the facade imagery.
Post-hoc corrections, reported next to the originals (nothing is overwritten):
  1. Grouped CV. Labels are units (verblijfsobjecten); 843 units sit in 468
     buildings (panden) and the 26 positives in 16. Unit-level StratifiedKFold
     lets units of one facade land in train and test. Here every level
     (probe, GBM) uses StratifiedGroupKFold by pand_id, over 5 fold seeds.
  2. Cluster bootstrap: 95% CIs resample buildings, not units.
  3. Photo-age shortcut: capture year from the Street View metadata as a
     training-free score (older = higher), and grouped CV without ≤2015 photos.
  4. Frozen sign reader on facade-aimed imagery (docs/sign-reader-protocol.md).
Geographic numbers are untouched by (1): a building never spans two cities.
Writes data/baseline7_results.json and docs/figures/baseline7_audit.png."""
import csv, json, re, unicodedata
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent

# ---- inputs (identical to baseline5_fusion.py / baseline6_signs.py) ----
d = np.load(ROOT / "data/embeddings.npz", allow_pickle=True)
by_id = {}
for name, f in zip(d["ids"], d["feats"].astype(np.float32)):
    by_id.setdefault(str(name).rsplit("_", 1)[0], []).append(f)
X_map = {k: np.mean(v, axis=0) for k, v in by_id.items()}
zs = json.load(open(ROOT / "data/vision_zeroshot_scores.json"))
reg = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/registry_features.csv"))}
meta = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/facade_metadata.csv"))}
USES = ["woonfunctie", "kantoorfunctie", "industriefunctie", "winkelfunctie", "bijeenkomstfunctie",
        "overige gebruiksfunctie", "onderwijsfunctie", "sportfunctie", "logiesfunctie", "gezondheidszorgfunctie"]
FLAGS = USES + ["ligplaats", "standplaats"]

def regfeats(vbo):
    g = reg.get(vbo, {})
    parts = set((g.get("use") or "").split(";"))
    year = float(g["build_year"]) if g.get("build_year") else np.nan
    area = float(g["area"]) if g.get("area") else np.nan
    st = g.get("vbo_status") or ""; ps = g.get("pand_status") or ""
    return [float(u in parts) for u in FLAGS] + [
        year, area, 2026.0 - year if year == year else np.nan,
        float("Verbouwing" in st), float("gevormd" in st),
        float("Bouwvergunning" in ps), float("Verbouwing" in ps),
    ]

rows = [r for r in csv.DictReader(open(ROOT / "data/labels.csv")) if r["label"] in ("1", "2") and r["pdok_vbo"] in X_map]
vbos = [r["pdok_vbo"] for r in rows]
y = np.array([1 if r["label"] == "1" else 0 for r in rows])
city = np.array([r["city"] for r in rows])
Xemb = np.array([X_map[v] for v in vbos])
zsc = np.array([zs.get(v, 0) for v in vbos])
Xreg = np.array([regfeats(v) for v in vbos], dtype=float)
groups = np.array([reg.get(v, {}).get("pand_id") or f"vbo:{v}" for v in vbos])
year_s = [meta.get(v, {}).get("pano_date", "")[:4] for v in vbos]
known = np.array([s.isdigit() for s in year_s])
photo_year = np.array([float(s) if s.isdigit() else np.nan for s in year_s])
photo_year[~known] = np.nanmedian(photo_year)
n_pos = int(y.sum())

# ---- metric: baseline-5 PR-AUC, ties broken at random and averaged (as baseline 6) ----
TIE_RNG = np.random.default_rng(11)
def _ap(y, s, perm):
    o = perm[np.argsort(-s[perm], kind="stable")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return float(np.sum((R - np.concatenate([[0], R[:-1]])) * P))
def pr_auc(y, s, k=20):
    if len(np.unique(s)) == len(s): return _ap(y, s, np.arange(len(s)))
    return float(np.mean([_ap(y, s, TIE_RNG.permutation(len(s))) for _ in range(k)]))

BOOT = np.random.default_rng(7)
def cluster_ci(y, s, g, n=2000):
    """Resample whole buildings, stratified by whether the building holds a positive."""
    ug, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(ug))]
    has_pos = np.array([y[m].any() for m in members])
    P, N = np.where(has_pos)[0], np.where(~has_pos)[0]
    vals = []
    for _ in range(n):
        pick = np.concatenate([BOOT.choice(P, len(P)), BOOT.choice(N, len(N))])
        idx = np.concatenate([members[i] for i in pick])
        vals.append(pr_auc(y[idx], s[idx], k=3))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]

nrm = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
LR = lambda: LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
GBM = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                             min_samples_leaf=25, l2_regularization=1.0, random_state=7)

def oof(make, X, splits, method):
    out = np.zeros(len(y))
    for tr, te in splits:
        m = make(); m.fit(X[tr], y[tr])
        out[te] = m.decision_function(X[te]) if method == "decision" else m.predict_proba(X[te])[:, 1]
    return out

def run_cv(splits, mask=None):
    """Stacked baseline-5 machinery on the given folds; probe OOF on the same folds."""
    p = oof(LR, nrm(Xemb), splits, "decision")
    res = {
        "fused": oof(GBM, np.column_stack([Xreg, zsc, p]), splits, "proba"),
        "image_only": oof(GBM, np.column_stack([zsc, p]), splits, "proba"),
        "registry_only": oof(GBM, Xreg, splits, "proba"),
        "probe": p,
    }
    return res

def splits_for(kind, seed, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    if kind == "unit":
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        return [(idx[a], idx[b]) for a, b in cv.split(idx, y[idx])]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return [(idx[a], idx[b]) for a, b in cv.split(idx, y[idx], groups[idx])]

SEEDS = [7, 11, 23, 42, 99]
results = {"n_units": int(len(y)), "n_buildings": int(len(np.unique(groups))), "pos_units": n_pos,
           "pos_buildings": int(len(np.unique(groups[y == 1])))}

cv_table = {}
for kind in ("unit", "grouped"):
    per_seed = {k: [] for k in ("fused", "image_only", "registry_only", "probe")}
    keep = None
    for seed in SEEDS:
        sc = run_cv(splits_for(kind, seed))
        for k in per_seed: per_seed[k].append(pr_auc(y, sc[k]))
        if seed == 7: keep = sc
    cv_table[kind] = {k: {"mean": round(float(np.mean(v)), 3), "min": round(float(np.min(v)), 3),
                          "max": round(float(np.max(v)), 3)} for k, v in per_seed.items()}
    cv_table[kind]["fused"]["ci95_seed7"] = cluster_ci(y, keep["fused"], groups)
    if kind == "grouped": grouped7 = keep
    else: unit7 = keep
results["cv"] = cv_table

# ---- 3. photo-age shortcut ----
# score = -capture_year: older photo -> higher "vacancy" score
tr = city == "Amsterdam"; te = ~tr
results["photo_age"] = {
    "pr_auc_all": round(pr_auc(y, -photo_year), 3),
    "pr_auc_geo_test_RU": round(pr_auc(y[te], -photo_year[te]), 3),
    "pr_auc_rotated_test_Adam": round(pr_auc(y[tr], -photo_year[tr]), 3),
    "ci95_all": cluster_ci(y, -photo_year, groups),
    "known_dates": int(known.sum()),
    "vacancy_rate_by_era": {},
}
for lo, hi, lab in [(0, 2015, "<=2015"), (2016, 2021, "2016-2021"), (2022, 2026, "2022-2026")]:
    m = known & (photo_year >= lo) & (photo_year <= hi)
    results["photo_age"]["vacancy_rate_by_era"][lab] = {"n": int(m.sum()), "pos": int(y[m].sum()),
                                                        "rate": round(float(y[m].mean()), 3) if m.any() else None}
recent = np.where(~(known & (photo_year <= 2015)))[0]
fused_recent = []
for seed in SEEDS:
    sp = splits_for("grouped", seed, recent)
    p = oof(LR, nrm(Xemb), sp, "decision")
    f = oof(GBM, np.column_stack([Xreg, zsc, p]), sp, "proba")
    fused_recent.append(pr_auc(y[recent], f[recent]))
results["grouped_fused_without_le2015_photos"] = {"n": int(len(recent)), "pos": int(y[recent].sum()),
    "mean": round(float(np.mean(fused_recent)), 3), "min": round(float(np.min(fused_recent)), 3),
    "max": round(float(np.max(fused_recent)), 3)}

# ---- 4. frozen sign reader on facade imagery ----
STRONG = ["TE HUUR", "TE KOOP", "VERHUURD", "VERKOCHT", "BESCHIKBAAR", "TO LET", "FOR RENT", "FOR SALE", "LEEGSTAND"]
WEAK = ["HUUR", "KOOP", "BEDRIJFSRUIMTE", "KANTOORRUIMTE", "WINKELRUIMTE", "BEDRIJFSPAND", "KANTOORPAND",
        "WINKELPAND", "MAKELAAR", "MAKELAARS", "MAKELAARDIJ", "GESLOTEN", "OPHEFFINGSUITVERKOOP"]
def normalise(strings):
    t = unicodedata.normalize("NFKD", " ".join(strings).upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " " + re.sub(r"[^A-Z0-9]+", " ", t).strip() + " "
fac_lvl, fac_text = {}, {}
for line in open(ROOT / "data/ocr_facade.jsonl"):
    rec = json.loads(line); vbo = rec["image"].split(".")[0].split("_")[0]
    t = normalise(rec["text"])
    lvl = 2 if any(f" {p} " in t for p in STRONG) else 1 if any(f" {p} " in t for p in WEAK) else 0
    fac_lvl[vbo] = max(fac_lvl.get(vbo, 0), lvl)
    readable = [s for s in rec["text"] if not re.search(r"goo|©", s, re.I) and len(re.sub(r"\W", "", s)) >= 3]
    fac_text[vbo] = fac_text.get(vbo, False) or bool(readable)
results["facade_sign_reader"] = {
    "buildings_read": len(fac_lvl), "images": sum(1 for _ in open(ROOT / "data/ocr_facade.jsonl")),
    "level_counts": {str(k): sum(v == k for v in fac_lvl.values()) for k in (0, 1, 2)},
    "units_with_readable_text": int(sum(fac_text.values())),
    "vacant_units_with_readable_text": int(sum(fac_text.get(v, False) for v, yy in zip(vbos, y) if yy == 1)),
}
(ROOT / "data/baseline7_results.json").write_text(json.dumps(results, indent=2))

# ---- figure: unit vs grouped CV, and vacancy rate by photo era ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
ax = axes[0]
names = ["fused", "image_only", "registry_only", "probe"]
xs = np.arange(len(names)); w = 0.36
for off, kind, col in [(-w / 2, "unit", "#8b94a7"), (w / 2, "grouped", "#e60012")]:
    mean = [cv_table[kind][k]["mean"] for k in names]
    lo = [cv_table[kind][k]["mean"] - cv_table[kind][k]["min"] for k in names]
    hi = [cv_table[kind][k]["max"] - cv_table[kind][k]["mean"] for k in names]
    ax.bar(xs + off, mean, w, color=col, yerr=[lo, hi], capsize=3,
           label="split by unit (as published)" if kind == "unit" else "split by building (corrected)")
ax.axhline(y.mean(), ls=":", c="#8b94a7", lw=1, label=f"prevalence {y.mean():.3f}")
ax.set_xticks(xs, ["fused", "image-only", "registry-only", "probe"])
ax.set(ylabel="5-fold CV PR-AUC (mean, range over 5 fold seeds)", title="In-city CV: unit vs building splits")
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.18, axis="y")
ax = axes[1]
eras = results["photo_age"]["vacancy_rate_by_era"]
labs = list(eras); rates = [eras[k]["rate"] or 0 for k in labs]
bars = ax.bar(labs, rates, color=["#e60012", "#8b94a7", "#8b94a7"])
for b, k in zip(bars, labs):
    ax.annotate(f"{eras[k]['pos']}/{eras[k]['n']}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
ax.set(ylabel="labelled vacancy rate", xlabel="Street View capture year", title="Vacancy rate by photo era")
ax.grid(alpha=.18, axis="y")
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/baseline7_audit.png")
print(json.dumps(results, indent=1))

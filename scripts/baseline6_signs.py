#!/usr/bin/env python3
"""Sign reader, step 2 — keyword labeling function + baseline 6 (fusion + sign).
Implements docs/sign-reader-protocol.md exactly (frozen before OCR was read):
  1. sign-only training-free score         10·sign_level + sign_views
  2. sign → zero-shot training-free score  sign_level + rank(zs)/n
  3. labeling-function quality per city    coverage / precision / recall
  4. baseline 6 = baseline 5 + [sign_level, sign_views], same GBM, folds, stacking
  5. 95% stratified bootstrap CIs (2,000) on headline PR-AUCs, baseline 5 included
Side output: data/repass_batch.csv (unclear labels ordered by sign evidence).
Writes data/sign_features.csv, data/baseline6_results.json, docs/figures/baseline6_signs.png."""
import csv, json, unicodedata, re
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent

# ---- 1. keyword levels (verbatim from the protocol) ----
STRONG = ["TE HUUR", "TE KOOP", "VERHUURD", "VERKOCHT", "BESCHIKBAAR", "TO LET", "FOR RENT", "FOR SALE", "LEEGSTAND"]
WEAK = ["HUUR", "KOOP", "BEDRIJFSRUIMTE", "KANTOORRUIMTE", "WINKELRUIMTE", "BEDRIJFSPAND", "KANTOORPAND",
        "WINKELPAND", "MAKELAAR", "MAKELAARS", "MAKELAARDIJ", "GESLOTEN", "OPHEFFINGSUITVERKOOP"]

def normalise(strings):
    t = unicodedata.normalize("NFKD", " ".join(strings).upper())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " " + re.sub(r"[^A-Z0-9]+", " ", t).strip() + " "

def image_level(strings):
    t = normalise(strings)
    hits = [p for p in STRONG if f" {p} " in t]
    if hits: return 2, hits
    hits = [p for p in WEAK if f" {p} " in t]
    return (1, hits) if hits else (0, [])

# Imagery audit (post-hoc, found while validating the null result): the OCR also
# reveals what each image *is*. Google serves "Sorry, we have no imagery here." as
# a normal JPEG, and user-contributed photos carry a personal © credit.
def image_kind(strings):
    j = " ".join(strings)
    if "no imagery here" in j.lower(): return "placeholder"
    if re.search(r"©\s*(?!\d{4}\s*Goo|Goo|Go\b)[A-Z][a-z]+ [A-Z]", j): return "user_photo"
    return "official"

sign = {}  # vbo -> {"level", "views", "keywords"}
kinds = {}  # vbo -> Counter of image kinds
for line in open(ROOT / "data/ocr_text.jsonl"):
    rec = json.loads(line)
    vbo = rec["image"].rsplit("_", 1)[0]
    kinds.setdefault(vbo, {"placeholder": 0, "user_photo": 0, "official": 0})[image_kind(rec["text"])] += 1
    lvl, kws = image_level(rec["text"])
    s = sign.setdefault(vbo, {"level": 0, "views": 0, "keywords": set()})
    s["level"] = max(s["level"], lvl)
    s["views"] += int(lvl >= 1)
    s["keywords"].update(kws)

with open(ROOT / "data/sign_features.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pdok_vbo", "sign_level", "sign_views", "keywords"])
    for vbo in sorted(sign):
        s = sign[vbo]; w.writerow([vbo, s["level"], s["views"], ";".join(sorted(s["keywords"]))])

# ---- shared inputs (identical to baseline5_fusion.py) ----
d = np.load(ROOT / "data/embeddings.npz", allow_pickle=True)
by_id = {}
for name, f in zip(d["ids"], d["feats"].astype(np.float32)):
    by_id.setdefault(str(name).rsplit("_", 1)[0], []).append(f)
X_map = {k: np.mean(v, axis=0) for k, v in by_id.items()}
zs = json.load(open(ROOT / "data/vision_zeroshot_scores.json"))
reg = {r["pdok_vbo"]: r for r in csv.DictReader(open(ROOT / "data/registry_features.csv"))}
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

all_labels = list(csv.DictReader(open(ROOT / "data/labels.csv")))
rows = [r for r in all_labels if r["label"] in ("1", "2") and r["pdok_vbo"] in X_map]
y = np.array([1 if r["label"] == "1" else 0 for r in rows])
city = np.array([r["city"] for r in rows])
vbos = [r["pdok_vbo"] for r in rows]
Xemb = np.array([X_map[v] for v in vbos])
zsc = np.array([zs.get(v, 0) for v in vbos])
Xreg = np.array([regfeats(v) for v in vbos], dtype=float)
s_lvl = np.array([sign.get(v, {"level": 0})["level"] for v in vbos], dtype=float)
s_vw = np.array([sign.get(v, {"views": 0})["views"] for v in vbos], dtype=float)
n_pos = int(y.sum()); n_tot = len(y)

# ---- metrics (baseline 5 definition; ties broken at random and averaged) ----
# Sign scores are heavily tied (most buildings score 0). A stable sort would break
# ties by file order, and labels.csv appends the active-batch positives last, so
# file order is not neutral. Continuous scores have no ties and are unaffected.
TIE_RNG = np.random.default_rng(11)
def _pr_auc_once(y, s, perm):
    o = perm[np.argsort(-s[perm], kind="stable")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    P = tp / np.maximum(tp + fp, 1); R = tp / max(tp[-1], 1)
    return float(np.sum((R - np.concatenate([[0], R[:-1]])) * P))

def pr_auc(y, s, k=20):
    if len(np.unique(s)) == len(s):
        return _pr_auc_once(y, s, np.arange(len(s)))
    return float(np.mean([_pr_auc_once(y, s, TIE_RNG.permutation(len(s))) for _ in range(k)]))

def curve(y, s):
    perm = TIE_RNG.permutation(len(s)); o = perm[np.argsort(-s[perm], kind="stable")]; yy = y[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    return [0] + list(tp / max(tp[-1], 1)), [1] + list(tp / np.maximum(tp + fp, 1))

RNG = np.random.default_rng(7)
def boot_ci(y, s, n=2000):
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n):
        idx = np.concatenate([RNG.choice(pos, len(pos)), RNG.choice(neg, len(neg))])
        vals.append(pr_auc(y[idx], s[idx], k=3))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]

def report(y, s):
    return {"pr_auc": round(pr_auc(y, s), 3), "ci95": boot_ci(y, s), "n": int(len(y)), "pos": int(y.sum())}

# ---- 2. training-free scores ----
zs_all = np.array([zs.get(v, 0) for v in vbos])
zs_rank = np.argsort(np.argsort(zs_all, kind="stable"), kind="stable") / len(zs_all)
score_sign = 10 * s_lvl + s_vw
score_sign_zs = s_lvl + zs_rank
tr = city == "Amsterdam"; te = ~tr
free = {}
for nm, sc in [("sign_only", score_sign), ("sign_then_zeroshot", score_sign_zs), ("zeroshot", zs_all)]:
    free[nm] = {"geo_test_RU": report(y[te], sc[te]), "rotated_test_Adam": report(y[tr], sc[tr]),
                "all_labels": report(y, sc)}

# ---- 3. labeling-function quality per city ----
lf = {}
for c in ["Amsterdam", "Rotterdam", "Utrecht", "all"]:
    m = np.ones(n_tot, bool) if c == "all" else city == c
    out = {}
    for k in (2, 1):
        fire = s_lvl[m] >= k; yy = y[m]
        out[f"level>={k}"] = {"fires": int(fire.sum()), "coverage": round(float(fire.mean()), 3),
                              "precision": round(float(yy[fire].mean()), 3) if fire.any() else None,
                              "recall": round(float(fire[yy == 1].mean()), 3) if yy.any() else None,
                              "tp": int((fire & (yy == 1)).sum())}
    out["base_rate"] = round(float(y[m].mean()), 3)
    lf[c] = out

# ---- 4. baseline 5 (re-run) and baseline 6 — identical machinery ----
CV = StratifiedKFold(5, shuffle=True, random_state=7)
nrm = lambda A: A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
LR = lambda: LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)
GBM = lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8,
                                             min_samples_leaf=25, l2_regularization=1.0, random_state=7)
def probe_scores(Xtr, ytr, Xte):
    m = LR(); m.fit(nrm(Xtr), ytr); return m.decision_function(nrm(Xte))

p_oof = cross_val_predict(LR(), nrm(Xemb), y, cv=CV, method="decision_function")
SIGN = np.column_stack([s_lvl, s_vw])

def run(extra):
    """extra: callable(mask) -> extra columns (None for baseline 5)."""
    ex = (lambda m: extra[m]) if extra is not None else (lambda m: np.empty((int(m.sum()), 0)))
    allm = np.ones(n_tot, bool)
    oof = cross_val_predict(GBM(), np.column_stack([Xreg, zsc, p_oof, ex(allm)]), y, cv=CV,
                            method="predict_proba")[:, 1]
    g = GBM(); g.fit(np.column_stack([Xreg[tr], zsc[tr], probe_scores(Xemb[tr], y[tr], Xemb[tr]), ex(tr)]), y[tr])
    s_geo = g.predict_proba(np.column_stack([Xreg[te], zsc[te], probe_scores(Xemb[tr], y[tr], Xemb[te]), ex(te)]))[:, 1]
    g2 = GBM(); g2.fit(np.column_stack([Xreg[te], zsc[te], probe_scores(Xemb[te], y[te], Xemb[te]), ex(te)]), y[te])
    s_rot = g2.predict_proba(np.column_stack([Xreg[tr], zsc[tr], probe_scores(Xemb[te], y[te], Xemb[tr]), ex(tr)]))[:, 1]
    return {"cv": report(y, oof), "geo": report(y[te], s_geo), "rotated": report(y[tr], s_rot)}, (oof, s_geo, s_rot)

b5, b5_s = run(None)
b6, b6_s = run(SIGN)

results = {"n": n_tot, "pos": n_pos, "training_free": free, "labeling_function": lf,
           "baseline5": b5, "baseline6": b6,
           "buildings_with_ocr": len(sign),
           "keyword_counts": {k: int(sum(k in s["keywords"] for s in sign.values())) for k in STRONG + WEAK}}
(ROOT / "data/baseline6_results.json").write_text(json.dumps(results, indent=2))

# ---- side output: re-pass queue for unclear labels ----
unclear = [r for r in all_labels if r["label"] == "3"]
def imagery(vbo):
    k = kinds.get(vbo, {"placeholder": 0, "user_photo": 0, "official": 0})
    return "none" if k["official"] + k["user_photo"] == 0 else f"{k['official']} official / {k['user_photo']} user"
def key(r):
    s = sign.get(r["pdok_vbo"], {"level": 0, "views": 0})
    return (imagery(r["pdok_vbo"]) == "none", -s["level"], -s["views"], -zs.get(r["pdok_vbo"], 0))
with open(ROOT / "data/repass_batch.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "city", "pdok_vbo", "address", "imagery", "sign_level", "sign_views", "keywords", "zeroshot"])
    for i, r in enumerate(sorted(unclear, key=key), 1):
        s = sign.get(r["pdok_vbo"], {"level": 0, "views": 0, "keywords": set()})
        w.writerow([i, r["city"], r["pdok_vbo"], r["address"], imagery(r["pdok_vbo"]), s["level"], s["views"],
                    ";".join(sorted(s["keywords"])), round(zs.get(r["pdok_vbo"], 0), 4)])
results["repass"] = {"unclear": len(unclear),
                     "no_imagery": sum(imagery(r["pdok_vbo"]) == "none" for r in unclear),
                     "with_strong_sign": sum(sign.get(r["pdok_vbo"], {"level": 0})["level"] == 2 for r in unclear),
                     "with_any_sign": sum(sign.get(r["pdok_vbo"], {"level": 0})["level"] >= 1 for r in unclear)}
lab_of = {r["pdok_vbo"]: r["label"] for r in all_labels}
audit = {}
for vbo, k in kinds.items():
    cat = ("no_imagery" if k["official"] + k["user_photo"] == 0 else
           "some_placeholder" if k["placeholder"] else "has_user_photo" if k["user_photo"] else "clean")
    lab = {"1": "vacant", "2": "occupied", "3": "unclear"}.get(lab_of.get(vbo), "unlabeled")
    audit.setdefault(cat, {}).setdefault(lab, 0); audit[cat][lab] += 1
results["imagery_audit"] = {"buildings": audit,
                            "images": {c: sum(k[c] for k in kinds.values()) for c in ("official", "placeholder", "user_photo")}}
(ROOT / "data/baseline6_results.json").write_text(json.dumps(results, indent=2))

# ---- figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150)
panels = [("A. Geographic holdout — test Rotterdam + Utrecht", te, 1),
          ("A′. Rotated fold — test Amsterdam", tr, 2)]
for ax, (title, m, j) in zip(axes[:2], panels):
    for nm, sc, col, ls in [("sign → zero-shot (free)", score_sign_zs, "#e60012", "-"),
                            ("sign only (free)", score_sign, "#f59e0b", "-"),
                            ("zero-shot (free)", zs_all, "#8b94a7", "-"),
                            ("baseline 6 fused+sign", None, "#2563eb", "--"),
                            ("baseline 5 fused", None, "#8b5cf6", ":")]:
        s = sc[m] if sc is not None else (b6_s if nm.startswith("baseline 6") else b5_s)[j]
        R, P = curve(y[m], s); ax.plot(R, P, color=col, ls=ls, lw=2, label=f"{nm} ({pr_auc(y[m], s):.3f})")
    ax.axhline(y[m].mean(), ls=":", c="#c0c4cc", lw=1)
    ax.set(xlabel="recall", ylabel="precision", title=title); ax.legend(frameon=False, fontsize=7.5); ax.grid(alpha=.18)
ax = axes[2]
for nm, s, col in [("baseline 6 fused+sign", b6_s[0], "#2563eb"), ("baseline 5 fused", b5_s[0], "#8b5cf6"),
                   ("sign → zero-shot (free)", score_sign_zs, "#e60012")]:
    R, P = curve(y, s); ax.plot(R, P, color=col, lw=2, label=f"{nm} ({pr_auc(y, s):.3f})")
ax.axhline(y.mean(), ls=":", c="#c0c4cc", lw=1)
ax.set(xlabel="recall", ylabel="precision", title="B. All labels (5-fold CV for fused models)")
ax.legend(frameon=False, fontsize=7.5); ax.grid(alpha=.18)
fig.tight_layout(); fig.savefig(ROOT / "docs/figures/baseline6_signs.png")
print(json.dumps({k: results[k] for k in ("training_free", "baseline5", "baseline6", "repass")}, indent=1))
print(json.dumps(lf, indent=1)); print(results["keyword_counts"])

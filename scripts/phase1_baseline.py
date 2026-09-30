#!/usr/bin/env python3
"""
scripts/phase1_baseline.py — VacancyOracle Phase 1

Reads data/labels.csv (label: 1=vacant 2=occupied 3=unclear) + data/candidates.csv
and produces docs/baseline-report.md with:
  * label QC (counts per city, class balance, unknown rate)
  * majority-class baseline  (PR-AUC == positive prevalence)
  * "street prior" rules-lite baseline (registry-free proxy from the candidate pool)
  * hook: if data/scores_rules_full.json exists (ParkScan scoresite() scores
    aligned by pdok_vbo), it is reported as the full-rules baseline too.

Metrics are pure numpy: PR curve, PR-AUC (step interp), precision at
recall>=0.5/0.7, confusion @ threshold chosen for demo precision target.

Usage:
    python3 scripts/phase1_baseline.py            # real labels (if present)
    python3 scripts/phase1_baseline.py --fixture  # demo run on a synthetic sample
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, DOCS = ROOT / "data", ROOT / "docs"
LABEL_MAP = {"1": 1, "2": 0, "3": None}
PRECISION_TARGET = 0.90


# ── io ────────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_labels(labels_path):
    cands = {r["pdok_vbo"]: r for r in load_csv(DATA / "candidates.csv") if r["pdok_vbo"]}
    rows = []
    for lab in load_csv(labels_path):
        pv = lab.get("pdok_vbo") or ""
        c = cands.get(pv)
        if c is None:
            continue  # labeled row not in current candidate pool
        rows.append({**c, "label": lab["label"], "raw": lab})
    return rows


def fixture_labels(n=160, seed=7):
    """Synthetic labels to prove the pipeline: 30% vacant, spatially clustered
    along 'commercial-sounding' streets so the street-prior baseline has signal."""
    rng = random.Random(seed)
    cands = load_csv(DATA / "candidates.csv")
    commercial_hints = ("straat", "weg", "kade", "haven", "terrain", "park", "dreef")
    picked = []
    for r in cands:
        if rng.random() < n / len(cands):
            vac = rng.random() < (0.38 if any(h in r["address"].lower() for h in commercial_hints[:4]) else 0.22)
            if rng.random() < 0.07:
                lab = "3"
            else:
                lab = "1" if vac else "2"
            picked.append({**r, "label": lab})
    return picked


# ── features for the rules-lite baseline ───────────────────────────────────
def street_prior(row):
    """A deliberately weak, honest proxy: vacancy tendency by street token & city.
    Not the ParkScan engine — that comes via scores_rules_full.json."""
    a = row["address"].lower()
    score = 0.32
    if any(t in a for t in ("haven", "kade", "terrain", "terrein", "industrieweg")):
        score += 0.24
    elif any(t in a for t in ("straat", "weg", "laan")) and any(d in a for d in ("noord", "oost", "west")):
        score += 0.08
    if "centrum" in a or "gracht" in a or "plein" in a:
        score -= 0.06
    return float(np.clip(score, 0, 1))


# ── metrics (pure numpy) ───────────────────────────────────────────────────
def pr_curve(y, scores):
    order = np.argsort(-scores, kind="mergesort")
    y, scores = y[order], scores[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P = tp / np.maximum(tp + fp, 1)
    R = tp / max(tp[-1], 1)
    return P, R, scores


def pr_auc(y, scores):
    P, R, _ = pr_curve(y, scores)
    R_prev = np.concatenate([[0.0], R[:-1]])
    return float(np.sum((R - R_prev) * P))  # step integration of the PR curve


def precision_at_recall(y, scores, target):
    P, R, s = pr_curve(y, scores)
    ok = np.where(R >= target)[0]
    return (float(P[ok[-1]]), float(s[ok[-1]])) if len(ok) else (np.nan, np.nan)


def confusion_at(y, scores, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


# ── report ─────────────────────────────────────────────────────────────────
def evaluate(y, scores):
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    prev = float(y.mean())
    p50, t50 = precision_at_recall(y, s, 0.5)
    p70, _ = precision_at_recall(y, s, 0.7)
    pT, tT = precision_at_recall(y, s, 0.0)
    # threshold meeting PRECISION_TARGET
    P, R, thr = pr_curve(y, s)
    cand = [(r, t) for p, r, t in zip(P, R, thr) if p >= PRECISION_TARGET]
    demo = max(cand, key=lambda x: x[0]) if cand else (0.0, float("inf"))
    tp, fp, fn, tn = confusion_at(y, s, demo[1] if np.isfinite(demo[1]) else 2.0)
    return {
        "pr_auc": pr_auc(y, s), "prevalence": prev,
        "p@r50": p50, "p@r70": p70,
        "demo_precision": PRECISION_TARGET, "demo_recall": demo[0],
        "confusion": (tp, fp, fn, tn),
    }


def fmt(m):
    return (f"PR-AUC **{m['pr_auc']:.3f}** · p@r.5 {m['p@r50']:.2f} · p@r.7 {m['p@r70']:.2f} · "
            f"@p≥{m['demo_precision']:.0%}: recall {m['demo_recall']:.2f} "
            f"(TP {m['confusion'][0]} / FP {m['confusion'][1]} / FN {m['confusion'][2]} / TN {m['confusion'][3]})")


def main():
    fixture = "--fixture" in sys.argv
    rows = fixture_labels() if fixture else merge_labels(DATA / "labels.csv")
    if not rows:
        sys.exit("data/labels.csv not found or empty — label some buildings first (or run --fixture).")

    use = [r for r in rows if r["label"] in ("1", "2")]
    n_unknown = len(rows) - len(use)
    y = np.array([LABEL_MAP[r["label"]] for r in use])
    cities = np.array([r["city"] for r in use])
    per_city = {c: (int((cities == c).sum()), int(((cities == c) & (y == 1)).sum())) for c in set(cities)}
    prev = float(y.mean())

    banner = "> **FIXTURE RUN** — synthetic labels, pipeline proof only. Re-run with real data/labels.csv.\n" if fixture else ""
    md = ["# VacancyOracle — Phase 1 baseline report\n", banner,
          f"- labeled rows: **{len(rows)}** ({n_unknown} unclear excluded) → evaluable: **{len(use)}**",
          f"- vacancy prevalence: **{prev:.1%}**"]
    md.append("- per city: " + ", ".join(f"{c} {n} rows/{v} vacant" for c, (n, v) in sorted(per_city.items())))

    md.append(f"\n## Baselines\n")
    md.append(f"1. **Majority class** — PR-AUC = prevalence = **{prev:.3f}**")
    m_street = evaluate(y, [street_prior(r) for r in use])
    md.append(f"2. **Street-prior rules-lite** — {fmt(m_street)}")

    full_rules = DATA / "scores_rules_full.json"
    if full_rules.exists():
        smap = json.loads(full_rules.read_text())
        scores = [smap.get(r["pdok_vbo"], prev) for r in use]
        m_full = evaluate(y, scores)
        md.append(f"3. **ParkScan rules engine (full)** — {fmt(m_full)}")
    else:
        md.append("3. **ParkScan rules engine** — pending `data/scores_rules_full.json` "
                  "(needs image-evidence frontage context; arrives with Phase 2 active learning).")

    zs = DATA / "vision_zeroshot_scores.json"
    if zs.exists():
        zmap = json.loads(zs.read_text())
        m_zs = evaluate(y, [zmap.get(r["pdok_vbo"], prev) for r in use])
        md.append(f"4. **SigLIP zero-shot vision** (local, no training) — {fmt(m_zs)}")

    md.append("\n## Decision framing\nFP = unwanted legal outreach mail. Demo threshold chosen at "
              f"precision ≥ {PRECISION_TARGET:.0%}; recall shown per baseline.\n")
    md.append("_Protocol frozen in `docs/eval-protocol.md` before model fitting._\n")

    out = DOCS / ("baseline-report.FIXTURE.md" if fixture else "baseline-report.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"rows={len(use)} prevalence={prev:.3f}")
    print(f"street-prior → {fmt(m_street)}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

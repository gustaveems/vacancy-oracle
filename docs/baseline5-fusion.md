# Baseline 5 — fused model (registry + vision)

⚠ **n_pos = 26.** Sparse-positive regime — every number below is
directional, as with all baselines so far.

Architecture: late fusion — HistGradientBoosting over 21 features:
19 BAG registry features | zero-shot margin | linear-probe score
(probe config mirrors `probe_supervised.py`). Early fusion (raw 768-d
embeddings into the GBM) was rejected: 26 positives cannot support 768
extra dimensions. Level-0 vision scores are out-of-fold per row (standard
stacking, same folds); on geographic splits probe and GBM train on the same
city set — test scores are honest. Fixed a-priori hyperparameters, no
tuning.

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train A'dam n=320, 5 pos → test R+U n=523, 21 pos) | **fused** | **0.064** |
|  | zero-shot margin | 0.126 |
|  | probe | 0.024 |
|  | registry GBM | 0.046 |
| **A′. Rotated fold** (train R+U, 21 pos → test A'dam, 5 pos) | **fused** | **0.023** |
| **B. Stratified 5-fold CV** (all 843, 26 pos) | **fused** | **0.468** |
|  | image-only (ablation: 5 − registry) | 0.481 |
|  | registry-only (ablation: 5 − image, baseline 3) | 0.407 |
|  | probe (baseline 4) | 0.390 |
|  | zero-shot | 0.082 |
|  | majority floor | 0.031 |

Secondary (CV OOF): precision@recall 0.5 = 0.48, 0.7 = 0.34; Brier
0.024 (no-skill 0.030).

**Demo threshold (≥90% precision):** threshold 0.933 — TP 1 / FP 0 / FN 25 / TN 817 (recall 0.04, precision 1.00).

Calibration: reliability plot in the figure — at 26 positives the
per-bin rates are noise; Brier is the trustworthy summary.

**Read:** in-city the fused model posts the family-best CV PR-AUC
(0.468) — but the ablation reveals the plateau: image-only stacks to
0.481 and registry-only reaches 0.407, all within noise of each
other at 26 positives. The two views are not yet complementary — they
are three ways of ranking roughly the same handful of buildings.
Geographic transfer remains the wall: fused 0.064 is the best
supervised geo number yet still below the training-free zero-shot margin
(0.126), and the rotated fold — training on R+U's 21 positives, four times
Amsterdam's — collapses to 0.023. The wall is symmetric and is not
bought off with training-side positives. The ≥90%-precision demo threshold
technically exists but catches 1 of 26 vacancies — vacuous, reported
as such. Against the rules-only baseline (0.032, 701-label era) the fused
model is an order of magnitude ahead, so the README claim stands; the
binding constraints are positive count and cross-city signal — more
Amsterdam labeling, not more model.

_Frozen protocol unchanged: metrics only on held-out splits._

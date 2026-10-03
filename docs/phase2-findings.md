# Phase 2 findings — zero-shot vision vs the rules

_2026-09-29 · 1,050 buildings · 4,071 Street View images embedded locally (SigLIP base/16, CPU) · 539 human labels (13 vacant)_

## Result

| Baseline | PR-AUC | Top-25 vacant hits (of 13) |
|---|---|---|
| Majority / prevalence | 0.024 | ~0.6 expected |
| ParkScan rules engine (registry-free run) | 0.032 | — |
| **SigLIP zero-shot vision** | **0.048** | **2** (ranks 14, 16) |

Direction is right — vision doubles the rules signal — but with 13 positives the
statistical power is nil and 11 positives sit at ranks 102–482. Classic
distribution problem, not a model problem.

## Why it's weak here

1. **Class sparsity**: downtown cores gave 2.4% prevalence; zero-shot margin
   (vacant-prompts − active-prompts) is a blunt scalar.
2. **No training yet** — the 539 labels are only used for evaluation so far.
3. **View aggregation**: max-of-4-headings is crude; a small probe on stored
   embeddings is the immediate upgrade.

## Active learning turn (what happens next)

349 unlabeled industrial-belt candidates were ranked by the zero-shot score.
The **"model picks" batch (top 100, scores 1.00 → 0.51)** is now the labeling
queue in the studio. If positives concentrate there the way they should,
Phase 3 (supervised probe over cached embeddings + registry features,
geographic holdout per `docs/eval-protocol.md`) gets a balanced-ish training
set and the PR curves finally separate.

## Artifacts

- `data/vision_zeroshot.csv` — per-view sims — 4,071 rows
- `data/vision_zeroshot_scores.json` — per-building calibrated score
- `data/active_batch.csv` — rank-ordered model picks (label these next)
- `docs/figures/pr_curves.png` — the figure above

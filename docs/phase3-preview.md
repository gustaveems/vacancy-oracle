# Phase 3 — supervised probe over embeddings

⚠ **n_pos = 26.** Still sparse — but doubled by the labeled active batch
(all 13 new positives landed inside the model's top-100 picks: 3.5× enrichment).

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train Amsterdam → test Rotterdam+Utrecht, n=523, 21 pos) | linear probe on SigLIP feats | **0.024** |
|  | zero-shot SigLIP margin | 0.126 |
| **B. Stratified 5-fold CV** (all 843, 26 pos) | linear probe | **0.390** |

Prevalence floor: 0.031.

**Read:** in-city the probe wins (CV 0.390 vs zero-shot full-set
0.082); geographic transfer flips it — with only 5
Amsterdam positives to train on, the probe collapses while the training-free
zero-shot margin holds. Positives by city: Amsterdam 5 · Rotterdam 5 · Utrecht 16. The geographic
headline claim still belongs to zero-shot; the fix is more Amsterdam positives
(fresh candidate collection) plus the fused model with registry features per
`docs/eval-protocol.md` (baselines 3–5).

_Frozen protocol unchanged: metrics only on held-out splits._

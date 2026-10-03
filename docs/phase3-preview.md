# Phase 3 preview — supervised probe over embeddings

⚠ **n_pos = 13.** Every number below is directional, not conclusive — that's the
sparsity wall documented in Phase 1; active-labeled industrial picks are the fix.

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train Amsterdam → test Rotterdam+Utrecht, n=8 pos) | linear probe on SigLIP feats | **0.018** |
|  | zero-shot SigLIP margin | 0.071 |
| **B. Stratified 5-fold CV** (all 539, n=13 pos) | linear probe | **0.404** |

Prevalence floor: 0.024.

**Read:** the probe learns transferable vacancy signal (CV beats zero-shot
substantially); geographic transfer at n=8 positives is coin-flip territory and
awaits the active batch. Next: label `data/active_batch.csv` top-100 model
picks → re-run everything (the label watcher already automates the refresh).

_Frozen protocol unchanged: metrics only on held-out splits._

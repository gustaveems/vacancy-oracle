# Baseline 3 — registry-only GBM (BAG features)

⚠ **n_pos = 26** (24 with full VBO features; the rest are
ligplaats/standplaats or ended registrations). Sparse-positive regime — the
same caveat as every baseline so far.

Features: use indicators (gebruiksdoel, 12 flags), build year, area,
age, VBO/pand status flags (verbouwing / gevormd / bouwvergunning).
Ownership form is BRK-only and key-required → not included (README documents
this; frozen `docs/eval-protocol.md` untouched). No image features, no city
feature. Fixed a-priori hyperparameters, no tuning (HistGradientBoosting:
150 trees, lr 0.08, ≤8 leaves, min 25/leaf, L2 1.0).

| Setup | Model | PR-AUC |
|---|---|---|
| **A. Geographic holdout** (train Amsterdam n=320, 5 pos → test R+U n=523, 21 pos) | registry GBM | **0.046** |
| **B. Stratified 5-fold CV** (all 843, 26 pos) | registry GBM | **0.407** |

Reference on the same 843-label set: zero-shot full 0.082 / geo 0.126,
probe CV 0.390 / geo 0.024 (`docs/phase3-preview.md`). Majority floor
0.031.

Secondary (CV): precision@recall 0.5 = 0.35, 0.7 = 0.19; Brier
0.023 (majority no-skill 0.030). Permutation importances
omitted — at 26 positives they are noise, not signal.

**Read:** in-city, BAG metadata alone is a real signal — CV 0.407 is
on par with the image probe (0.390) and well above zero-shot (0.082); at
26 positives the gap to the probe is noise, call it parity. Geographic
transfer is the same wall as every supervised model here: 0.046
with 5 Amsterdam training positives, below the training-free
zero-shot (0.126). The registry's role is fusion input (baseline 5), not
standalone detection — statuses like *verbouwing*/*gevormd*/*bouwvergunning*
mark transition states that images see directly, so the two views should
complement rather than compete. Calibration and FP-cost framing deferred to
the fused model per protocol.

_Frozen protocol unchanged: metrics only on held-out splits._

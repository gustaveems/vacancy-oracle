# Phase 4 — write-up: from ParkScan rules to a trained Oracle

*Successor to [parkscan-nl](https://github.com/gustaveems/parkscan-nl). Everything
below is measured on self-collected labels under a protocol frozen before any
model was fit (`docs/eval-protocol.md`). Numbers are directional at this
positive count — every doc says so, including this one.*

## The question

ParkScan NL guessed at vacancy with hand-written rules over a scraped site
assessment. On real labels, are learned signals better — and is the transfer
claim honest? The frozen protocol's answer clause: *"a model that can't beat
[the rules baseline] gets said so in the README."*

## What was built

**Data.** 1,050 BAG candidates across Amsterdam / Rotterdam / Utrecht
(PDOK Locatieserver seeds: downtown cores + industrial belts), 4,071 Google
Street View frontages fetched locally and resumably, and 1,050 human labels
(26 vacant / 817 occupied / 207 unclear) self-collected through a
localStorage-first label studio. Registry features for 1,046 of them via
batched WFS 2.0 POST queries against the keyless PDOK BAG service
(`data/registry_features.csv`).

**Active learning, validated.** The model's top-100 picks (zero-shot margin
over embedded street views) were labeled as one batch: all 13 vacancies found
in the unlabeled pool landed inside those 100 picks — **3.5× enrichment**
over the 3.7% base rate of that pool. Positives doubled, 13 → 26.

**Models** (frozen splits: 5-fold in-city CV; geographic headline = train
Amsterdam → test Rotterdam+Utrecht, plus one rotated fold):

| Model | In-city CV PR-AUC | Geographic PR-AUC |
|---|---|---|
| Majority floor | 0.031 | 0.031 |
| Rules v1 (ParkScan engine, recovered) | 0.032 | — |
| SigLIP zero-shot margin (no training) | 0.082 | **0.126** |
| Linear probe on embeddings | 0.390 | 0.024 |
| Registry-only GBM (baseline 3) | 0.407 | 0.046 |
| **Fused model (baseline 5, proposed)** | **0.468** | 0.064 (rotated 0.023) |

**Infrastructure.** A label watcher (`scripts/watch_labels.sh`) ingests any
`labels*.csv` export dropped in Downloads, re-runs scores → baselines →
probe → GBMs → fusion → studio → demo, and commits+pushes. Every doc in
`docs/` is machine-regenerated, so this repository is self-updating as long
as labeling continues.

## Findings, stated honestly

1. **The rules engine is the majority floor.** Rules v1 scores 0.032 against
   a 0.031 prevalence baseline — it carries essentially no ranking signal on
   this data. Every learned model beats it by an order of magnitude or more.
   The protocol's shaming clause applies to the rules, not the Oracle.
2. **In-city, three views of the same truth.** Probe (0.390), registry GBM
   (0.407) and fusion (0.468) — and the image-only stack (0.481) — sit
   within noise of each other at 26 positives. Fusion is not yet adding
   complementarity; that is a data problem (positives), not a model problem.
3. **The geographic wall is real and symmetric.** No supervised model
   transfers cities at this label volume: best supervised geo 0.064, while
   the *training-free* zero-shot margin holds 0.126. The rotated fold —
   training on Rotterdam+Utrecht's 21 positives, 4× Amsterdam's — makes
   transfer *worse* (0.023). More training-side positives do not buy
   generalization; more **target-city** positives might.
4. **The demo threshold is not deployable yet.** The protocol asks for a
   ≥90%-precision outreach threshold (a false positive is an unwanted legal
   letter). It technically exists on CV scores — and catches **1 of 26**
   vacancies. Reported as vacuous, with the FP-cost curve in
   `docs/baseline5-fusion.md`.

## Limitations (declared, not buried)

- **26 positives.** PR-AUCs carry wide, un-quantified CIs; no bootstrap done.
- **Selection bias.** Seed-based sampling (downtown + industrial belts), not
  a census — prevalence numbers describe commercial cores like these.
- **Single annotator, photo-time truth.** Labels are one person's read of a
  Google photo whose capture date ≠ photo-viewing date; `unclear` (207)
  labels are excluded, never adjudicated.
- **Temporal drift in features.** BAG statuses were fetched Oct 7 2026;
  labels were made Sep 30–Oct 3 — a *verbouwing* recorded today may postdate
  the photo a building was labeled from.
- **Ownership form is missing** from baseline 3 — it lives in BRK
  (Kadaster, key-required), not BAG. The protocol's feature list is
  partially unreachable; documented, protocol left frozen.
- **Rules v1 runs on the recovered engine** (`parkscan-restored`) with its
  mock places/vision frontage — closer to ParkScan-as-published than any
  production system.
- **CPU-only SigLIP-base.** Embedded on a laptop in hours; bigger backbones
  are untried scale left on the table.
- **Demo ranks are in-sample** (the model refit on everything for ranking);
  the demo page carries this caveat in its own banner.

## The demo

`node scripts/oracle_score.py && node scripts/make_demo.mjs` →
**`~/Desktop/VacancyOracle Demo.html`** — 1,050 candidates with their local
street views, toggle between *Rank by Rules v1* and *Rank by Oracle v2*,
city filters, live stats (top-50 catches: 1 vs 26 in-sample; rank agreement
ρ = 0.05 — the two systems disagree almost entirely). Imagery is referenced
by local `file://` path; no API keys are ever embedded.

## What's next

1. **Amsterdam positives** — the binding constraint on every geographic
   number above. The pipeline (collect → fetch → embed → studio → label →
   watcher) costs nothing to run again; industrial-belt expansion worked
   once (3.5× enrichment).
2. **Unclear re-pass** — 207 labels excluded; 19 of them are active-batch
   picks, probably recoverable.
3. Full three-way rotated folds + bootstrap CIs once positives grow;
   calibration refinement (isotonic on 26 positives would be theater).

*Dataset: address IDs + labels published under CC-BY-NC in this repo.
No personal data beyond publicly registered business addresses.*

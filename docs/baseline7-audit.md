# Baseline 7 — audit: building-grouped CV, photo-age confound, facade sign reader

Code: `scripts/fetch_facade.mjs` → `scripts/ocr_signs.swift` → `scripts/baseline7_audit.py`.
Figure: `docs/figures/baseline7_audit.png`. Results: `data/baseline7_results.json`.

Everything below except §4 is **post-hoc**: these are corrections found while
looking at the facade imagery, not pre-registered tests. Earlier documents are
left as they were; this one supersedes their in-city numbers.

## 1. The in-city CV leaked across units of the same building

Labels are units (BAG *verblijfsobjecten*), not buildings. The facade imagery
made it obvious: 5 of the "vacant" examples are the same brick facade, 4 are the
same café terrace.

| | Units | Buildings (*panden*) |
|---|---|---|
| All labels | 843 | 468 |
| Vacant | 26 | 17 |

Unit-level `StratifiedKFold` puts units of one facade in train *and* test. Every
in-city number in baselines 3–5 was computed that way. Corrected with
`StratifiedGroupKFold` by `pand_id` at every stacking level, over 5 fold seeds:

| Model (5-fold CV PR-AUC) | Split by unit (as published) | Split by building (corrected) |
|---|---|---|
| Fused (baseline 5) | 0.554 (0.469 – 0.703) | **0.374** (0.331 – 0.415) |
| Image-only stack | 0.473 (0.418 – 0.524) | 0.351 (0.280 – 0.422) |
| Registry-only GBM (baseline 3) | 0.451 (0.389 – 0.508) | **0.208** (0.168 – 0.237) |
| SigLIP linear probe (baseline 4) | 0.433 (0.391 – 0.481) | **0.428** (0.406 – 0.449) |

Mean over seeds, range in brackets. Building-level bootstrap 95% CI for the fused
model (seed 7): unit split 0.24–0.69, building split 0.16–0.54. Prevalence 0.031.

**Read:**
- The published 0.468 was one fold seed; across seeds the unit-split fused model
  ranges 0.47–0.70. Single-seed CV at 26 positives is not a number to quote.
- **Registry features were mostly leakage.** They describe the building, so a
  vacant unit's neighbours carried its answer. Registry-only more than halves.
- **The image probe is the robust signal**: 0.433 → 0.428. Under honest splits it
  beats the fused model — stacking registry features on top now costs accuracy.
- Geographic numbers (holdout and rotated fold) are unaffected: a building never
  spans two cities.

## 2. Photo age is a confound

The Street View metadata gives each panorama's capture date — something the
pipeline never recorded, although labels mean "vacant *at photo time*".

| Capture year | Labelled units | Vacant | Rate |
|---|---|---|---|
| ≤ 2015 | 71 | 7 | **9.9%** |
| 2016 – 2021 | 207 | 2 | 1.0% |
| 2022 – 2026 | 565 | 17 | 3.0% |

All 7 old-photo positives are in Utrecht, some from 2010. Used as a
training-free score (older photo → higher score), **photo age alone reaches
0.167 PR-AUC on the geographic holdout** — above training-free zero-shot (0.126)
and the fused model (0.062). Building-level CI over all labels is wide
(0.02–0.32), but the direction is a warning: some of what the models call
"vacancy" may be "old camera generation, run-down street, 2010".

Two consequences:
- For ParkScan's purpose (vacant *now*), a 2010 photo is a 16-year-old label.
- Dropping the 71 units with ≤ 2015 photos (7 positives), the grouped fused CV
  falls further to **0.245** (0.211 – 0.338).

## 3. Facade-aimed imagery

`fetch_facade.mjs` over all 1,050 candidates: metadata first (free), then three
views per building aimed at the facade at 45° FOV.

| Metadata status | Buildings |
|---|---|
| Official Google imagery | 837 (2,511 images, 0 errors, 101 s) |
| No imagery | 68 — all labelled unclear (matches the OCR audit's 67) |
| Nearest panorama is a user photo | 145 — skipped |

Facade aiming works: units with any readable non-watermark text rise from
241 to 324 of 837.

## 4. Frozen sign reader on facade imagery (pre-registered): null again

Same keywords, same rules, 2,511 images, 133 s on-device. **Zero hits, strong or
weak.** This time the null is about the world, not the pixels: OCR now reads
storefront text, and inspecting the vacant facades shows no letting boards at
all. In this label set, vacancy is visible as empty interiors and closed
shutters, not as signage. The sign-reader hypothesis is retired.

## What this changes

- **Headline in-city number**: SigLIP linear probe, building-grouped CV,
  **0.43 PR-AUC** (0.41–0.45 across seeds) at 3.1% prevalence. Not 0.47.
- **Geographic headline unchanged**: training-free zero-shot 0.126 — now with
  the caveat that photo age alone does 0.167 on the same split.
- **Next**: capture year becomes a mandatory control (stratify or report
  per-era), protocol updated to group by building, and the label pool needs
  recent-photo Amsterdam positives — the old-photo Utrecht cluster cannot carry
  a model that has to work today.

_Original baseline documents are unchanged; this audit supersedes their
in-city numbers._

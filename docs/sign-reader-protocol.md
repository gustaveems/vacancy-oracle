# Sign reader — pre-registration (frozen before any OCR output is read)

Addendum to `eval-protocol.md`. Written and committed before
`scripts/ocr_signs.swift` is run over the imagery; nothing below may be
changed after the first look at results. Any later variant is reported as a
post-hoc exploration, never as the headline.

## Why
Every supervised number so far hits the same wall: 26 positives (5 in
Amsterdam) and no geographic transfer. A *"TE HUUR"* board reads the same in
Amsterdam and Utrecht, so a text signal needs no training data and should not
care which city it is in. This is a **labeling function**, not a model.

## Extraction
- Apple Vision `VNRecognizeTextRequest`, accurate mode, languages nl-NL + en-US,
  language correction off. On-device; no API calls, no keys, no cost.
- All four views (0/90/180/270°) of every candidate are read.
- Normalisation per image: join all recognised strings with spaces, uppercase,
  strip diacritics, replace every non-alphanumeric run with one space.

## Keyword levels (whole-word matches on the normalised text)
| Level | Phrases |
|---|---|
| **2 — strong** | `TE HUUR`, `TE KOOP`, `VERHUURD`, `VERKOCHT`, `BESCHIKBAAR`, `TO LET`, `FOR RENT`, `FOR SALE`, `LEEGSTAND` |
| **1 — weak** | `HUUR`, `KOOP`, `BEDRIJFSRUIMTE`, `KANTOORRUIMTE`, `WINKELRUIMTE`, `BEDRIJFSPAND`, `KANTOORPAND`, `WINKELPAND`, `MAKELAAR`, `MAKELAARS`, `MAKELAARDIJ`, `GESLOTEN`, `OPHEFFINGSUITVERKOOP` |

Image level = highest level matched (0 if none).

## Per-building features
- `sign_level` = max image level over the building's views (0/1/2)
- `sign_views` = number of views with level ≥ 1 (0–4)

## Pre-registered scores and evaluations
1. **Sign-only (training-free)**: `score = 10·sign_level + sign_views`.
2. **Sign → zero-shot (training-free)**: rank by `sign_level`, break ties with
   the existing zero-shot margin: `score = sign_level + rank(zs)/n`.
   Both have no fitted parameters, so they are evaluated directly on every split
   — geographic holdout (test R+U), rotated fold (test Amsterdam) and all labels.
3. **Labeling-function quality** per city: coverage, precision and recall of
   `sign_level = 2` and `sign_level ≥ 1` against human labels 1/2.
4. **Baseline 6 — fusion + sign**: baseline 5 unchanged (same GBM, same fixed
   hyperparameters, same folds and stacking) with `sign_level` and `sign_views`
   appended. Splits A, A′, B as in baseline 5.
5. **Uncertainty**: 95% bootstrap intervals (2,000 resamples, stratified by
   label) on every headline PR-AUC — for baseline 5 as well, so the comparison
   is like-for-like.

## Expected failure modes (declared up front)
- Views are wide: a board on the *neighbouring* unit triggers a false positive.
- `VERHUURD`/`VERKOCHT` boards mean "just let" — the unit may be empty at photo
  time or already re-occupied.
- Small or angled boards at 640×480 will be missed (recall ceiling).
- `GESLOTEN` also matches opening-hours signs ("zondag gesloten").

## Side output
`data/repass_batch.csv`: the `unclear` (label 3) buildings ordered by
`sign_level`, then `sign_views`, then zero-shot margin — a re-labeling queue
for the 207 excluded labels (write-up "What's next" #2). Matched keywords only,
no raw OCR text, are written to tracked files; raw text stays in the
git-ignored `data/ocr_text.jsonl` with the imagery it came from.

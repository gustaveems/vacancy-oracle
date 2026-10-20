# Baseline 6 — sign reader (pre-registered) + imagery audit

Protocol: `docs/sign-reader-protocol.md`, committed before the OCR ran.
Code: `scripts/ocr_signs.swift` (Apple Vision, on-device, 4,071 images in
110 s) → `scripts/baseline6_signs.py`. Figure: `docs/figures/baseline6_signs.png`.

## 1. Pre-registered result: a clean null

**Not one keyword fired on any of the 4,071 street views** — strong or weak,
in any city. Coverage 0 means the sign features are constant, so:

| Setup | Score | PR-AUC | 95% bootstrap CI |
|---|---|---|---|
| A. Geographic holdout (test R+U, 523 / 21 pos) | sign → zero-shot (free) | 0.127 | 0.084 – 0.201 |
|  | zero-shot margin (free) | 0.126 | 0.084 – 0.204 |
|  | baseline 6 fused + sign | 0.062 | 0.046 – 0.093 |
|  | baseline 5 fused | 0.062 | 0.046 – 0.092 |
| A′. Rotated fold (test A'dam, 320 / 5 pos) | sign → zero-shot (free) | 0.032 | 0.017 – 0.050 |
|  | baseline 6 / baseline 5 fused | 0.023 / 0.023 | 0.019 – 0.032 |
| B. All labels (843 / 26 pos) | baseline 6 fused + sign (5-fold CV) | 0.469 | 0.312 – 0.656 |
|  | baseline 5 fused (5-fold CV) | 0.469 | 0.304 – 0.662 |

Sign-only scores are all-zero ties, so their PR-AUC is the random-ordering
expectation (ties broken at random and averaged — the stable sort used in
earlier baselines would have ordered ties by file position, and `labels.csv`
appends the active-batch positives last). Baseline 6 = baseline 5 exactly,
as it must with constant features. Baseline 5 is re-run here with the same
tie handling — the GBM emits tied probabilities too — so it reads 0.469 / 0.062
against the published 0.468 / 0.064; the difference is tie order alone.

**Why it failed — validated, not assumed.** The OCR works: 4,063 / 4,071
images return text. Almost all of it is the Google watermark. At 640×480 with
a 90° field of view, aimed at fixed compass headings rather than at the
building, a "TE HUUR" board is a handful of pixels. The hypothesis was not
falsified — the imagery cannot test it.

**What the bootstrap adds** (first CIs in this project): geographically the
training-free zero-shot margin beats the best supervised model with
**non-overlapping 95% intervals** (0.084–0.204 vs 0.046–0.092). Phase 4 called
this "directional"; it is now a measured result. In-city, the fused model's
0.469 spans 0.30–0.66 — every in-city comparison in baselines 3–5 sits inside
that band.

## 2. Imagery audit (post-hoc, found while validating the null)

The OCR also tells you what each image *is*:

| Images | Count |
|---|---|
| Official Street View | 3,653 |
| Google "Sorry, we have no imagery here." placeholder | 268 |
| User-contributed photo (personal © credit) | 150 |

| Buildings | Vacant | Occupied | Unclear |
|---|---|---|---|
| No imagery at all (4 / 4 placeholders) | 0 | 0 | **67** |
| Some user-contributed views | 0 | 60 | 12 |
| Clean | 26 | 757 | 128 |

- **No metric was contaminated**: every no-imagery building was already
  labelled `unclear` and excluded. But `fetch_imagery.mjs` stores the
  placeholder as `ok`, and those 268 placeholders were embedded and zero-shot
  scored alongside real storefronts.
- **The unclear re-pass is 140 buildings, not 207.** 67 of the 207 cannot be
  resolved from Street View at any effort. `data/repass_batch.csv` now lists
  the 140 with imagery first (zero-shot order), the 67 last, marked `none`.

## 3. Fix: facade-aimed imagery (`scripts/fetch_facade.mjs`, not yet run)

Checks the free metadata endpoint first (no-imagery and user photos are skipped
before any image is billed), aims at the building from the panorama's own
position, zooms to 45° FOV (≈2× pixels on the shopfront, 640×640), optional
±30° flanks, and logs the panorama **capture date** — labels mean "vacant at
photo time", and that date was never recorded.

```bash
node scripts/fetch_facade.mjs --dry-run          # metadata only — no image quota used
node scripts/fetch_facade.mjs --unclear-only     # 140 re-pass buildings, ~140 images
node scripts/fetch_facade.mjs --flank            # all 1,050, ~3,150 images
swift scripts/ocr_signs.swift data/imagery_facade data/ocr_facade.jsonl
```

Re-running the frozen sign reader on facade imagery is the real test of the
hypothesis. It stays pre-registered: same keywords, same scores, same splits.

_Frozen protocol unchanged: metrics only on held-out splits; all post-hoc
material labelled as such._

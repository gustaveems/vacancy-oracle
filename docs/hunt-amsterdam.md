# Amsterdam positive hunt — label queue

Amsterdam has 5 vacant labels (in fewer buildings), which caps every
geographic number. This round builds a ranked queue of *new* Amsterdam
candidates for the label studio. Labeling itself is manual.

## Pipeline
```bash
node scripts/hunt_amsterdam.mjs --max 400          # PDOK reverse-geocode, 20 new districts, append-only
node scripts/fetch_facade.mjs --city Amsterdam --flank   # metadata + facade views (capture dates)
node scripts/fetch_imagery.mjs --city Amsterdam    # 4-heading views (what the probe was trained on)
python3 scripts/embed_zero_shot.py                 # then: git checkout data/vision_zeroshot_scores.json
python3 scripts/embed_facade.py
python3 scripts/hunt_rank.py                       # → data/active_batch.csv, data/hunt_scores.csv
node scripts/make_studio.mjs                       # → ~/Desktop/Vacancy Label Studio.html
```

`embed_zero_shot.py` re-normalises every building's zero-shot score when new
buildings are added, which would shift the published baselines; restore the
committed `vision_zeroshot_scores.json` after it runs.

## What came out

- **388 new candidates**, all gemeente Amsterdam (`0363` prefix — PDOK's reverse
  search also returns Ouder-Amstel, filtered out), spread over 24 one-km cells.
- Metadata: 276 with official Google imagery, of which **203 photographed 2022
  or later**. Older photos are excluded from the queue (photo-age confound,
  `docs/baseline7-audit.md`).
- **Ranker choice, building-grouped CV on the 418 labeled buildings with both
  view types** (`data/hunt_results.json`):

  | Probe | Grouped 5-fold CV PR-AUC | Geo holdout (test R+U) | Rotated (test A'dam) |
  |---|---|---|---|
  | 4-heading views (90° FOV) | **0.465** (0.40–0.55) | 0.026 | 0.013 |
  | Facade-aimed views (45° FOV) | 0.331 (0.31–0.36) | 0.031 | 0.017 |

  Prevalence 0.034. The tighter facade crop *loses* in-city signal: the wide
  views carry street context. That fits the confound story — part of what the
  probe reads is the street and the photo era, not the shopfront. Neither view
  transfers across cities. The queue uses the 4-heading probe.

- **Queue**: top 100 recent-photo candidates by probe score →
  `data/active_batch.csv`, highlighted in the label studio.

## Success criterion (set before labeling)
At the training base rate (~3%), 100 random Amsterdam candidates would hold about
3 vacancies. The first active batch found 13 in its top 100 (3.5× enrichment).
This queue is worth repeating if it yields **≥ 6 vacant labels in distinct
buildings**; fewer means the probe's in-city signal does not survive new
districts either.

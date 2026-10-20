# VacancyOracle

**Weakly-supervised detection of vacant & underused buildings in the Netherlands —
street-level vision + BAG registry signals, with geographically held-out evaluation.**

The ML successor to ParkScan NL (IE Hackathon 2026): the hand-tuned scoring rules
become a trained model you can measure, ablate and generalize.

## The question

ParkScan scored buildings 0–100 with weighted rules. Were the rules right?
VacancyOracle answers with data:

1. **Label** a few hundred storefronts from Street View (this repo ships the tooling).
2. **Baseline**: the original rules engine, measured against human labels (PR-AUC).
3. **Model**: CLIP/SigLIP street-image embeddings + BAG registry features +
   geospatial context → gradient boosting with **calibration**.
4. **Generalize**: train Amsterdam → test Rotterdam/Utrecht. If it transfers,
   the signal is real, not city-specific memorization.

## Layout

```
scripts/collect-candidates.mjs   PDOK reverse-geocode → data/candidates.csv (1,050 addresses)
scripts/fetch_registry.mjs       batched WFS POST → data/registry_features.csv (BAG features)
scripts/watch_labels.sh          self-refreshing pipeline: new label export → all scores/models/docs/demo
label/index.html                 keyboard-driven label studio (localStorage → export CSV)
docs/eval-protocol.md            frozen evaluation rules — written before any model is fit
docs/phase4-writeup.md           final write-up: findings, declared limitations, what's next
```

## Labeling (Phase 0 — you)

```bash
cd ~/Projects/vacancy-oracle && python3 -m http.server 8000
# open http://localhost:8000/label/
```

- Paste a Google *Street View Static API* key (stored only in your browser), or use
  the “open Street View ↗” link per row.
- `↑/↓` navigate · `1` vacant · `2` occupied · `3` unclear.
- Pace: **40/day ≈ 12 min** → 300 labels in ~7 school-days.
- `⬇ export labels.csv` → drop into `data/labels.csv` → commit.

## Roadmap

- [x] Phase 0 — candidates + label studio
- [x] Phase 1 — first real baseline (docs/baseline-report.md): 701 labels, **vacancy prevalence 2.4%** in downtown commercial cores, street-prior PR-AUC 0.020 — weaker than majority. Finding: signal density, not label volume, is the bottleneck → pool expanded with 349 industrial-belt candidates (haven/kade/terrain seeds).
- [x] Phase 2 — local SigLIP zero-shot over 4,071 street views: PR-AUC 0.048 vs rules 0.032 vs majority 0.024 (`docs/phase2-findings.md`)
- [x] Phase 2b — active-learning batch: top-100 model picks labeled — **1,050 labels** total; all 13 new positives landed inside the top-100 picks (3.5× enrichment over the 3.7% base rate)
- [x] Phase 3 — supervised probe + geographic holdout (`docs/phase3-preview.md`): in-city 5-fold CV PR-AUC **0.390** vs zero-shot 0.082; geographic transfer flips — probe **0.024** vs zero-shot **0.126**, because only 5 Amsterdam positives exist to train on. Zero-shot owns the geographic headline for now.
- [x] Baseline 3 — registry-only GBM (`docs/baseline3-registry.md`): in-city CV PR-AUC **0.407** — parity with the image probe and far above zero-shot: BAG metadata alone carries real in-city signal. Geographic holdout **0.046**, same 5-Amsterdam-positives wall.
- [x] Baseline 5 — fused model + calibration (`docs/baseline5-fusion.md`): in-city CV PR-AUC **0.468** — family best, but the ablation plateau is reported honestly: image-only 0.481 / registry-only 0.407, all within noise at 26 positives. Geographic holdout **0.064** (best supervised, still under training-free zero-shot 0.126); rotated fold **0.023** — the geographic wall is symmetric, more training-side positives don't buy transfer. ≥90%-precision demo threshold exists but catches 1/26 vacancies (vacuous until positives grow). Precision@recall 0.5/0.7 = 0.48/0.34; Brier 0.024 vs 0.030 no-skill.
- [x] Phase 4 — demo + write-up: Rules v1 vs Oracle v2 toggle over the 1,050 labeled candidates with street views (`scripts/make_demo.mjs` → `~/Desktop/VacancyOracle Demo.html`, in-sample caveat printed on the page; rank agreement ρ = 0.05 — the systems disagree almost entirely). Final write-up: `docs/phase4-writeup.md`. More Amsterdam positives remain the binding constraint on every geographic number above.
- [x] Baseline 6 — sign reader, pre-registered (`docs/sign-reader-protocol.md` → `docs/baseline6-signs.md`): on-device Apple Vision OCR over all 4,071 views (110 s) for Dutch vacancy keywords ("TE HUUR", "TE KOOP", …). **Clean null: zero keyword hits** — validated, not assumed: the OCR reads text fine, but at 640×480 / 90° FOV aimed at fixed compass headings a board is a few pixels. First **bootstrap CIs**: geographically, training-free zero-shot (0.126, 95% CI 0.084–0.204) beats fused (0.062, 0.046–0.092) with non-overlapping intervals. **Imagery audit** from the same OCR pass: 268 views are Google's "no imagery" placeholder (67 buildings, all labelled unclear → the re-pass is 140, not 207; `data/repass_batch.csv`) and 150 are user-contributed photos.
- [x] Baseline 7 — audit (`docs/baseline7-audit.md`): facade-aimed imagery for 837 buildings (2,511 images; 68 no-imagery, 145 user photos skipped) and the frozen sign reader re-run on it — **null again, now about the world, not the pixels**: no letting boards on the vacant facades; hypothesis retired. The facade views exposed two problems with the in-city numbers: **(1) unit-level CV leaked** — 843 units sit in 468 buildings and the 26 positives in 17; with building-grouped CV the fused model drops to **0.374** (0.33–0.42 over 5 seeds; the published 0.468 was one seed of a 0.47–0.70 range) and registry-only to 0.208, while the **SigLIP probe holds at 0.428** — the robust signal. **(2) Photo age is a confound**: labelled vacancy is 9.9% in ≤2015 photos vs 3.0% in 2022+ (all 7 old-photo positives in Utrecht), and capture year alone scores 0.167 on the geographic holdout. In-city headline is now **probe, building-grouped: 0.43 PR-AUC**.
- [ ] Next: group-by-building and capture year written into the eval protocol; recent-photo Amsterdam positives (label queue from the probe); live demo map.

## Demo (Phase 4)

```bash
python3 scripts/oracle_score.py && node scripts/make_demo.mjs
# → ~/Desktop/VacancyOracle Demo.html
```

Toggle ranking between ParkScan **Rules v1** and the fused **Oracle v2**,
filter by city, flip through street views. Demo ranks come from the model
refit on all usable labels (in-sample — the page says so in its own banner);
held-out metrics live in `docs/`. Imagery is referenced by local `file://`
path; no API keys are ever embedded.

Part of the Gustave Soulas portfolio · successor to [parkscan-nl](https://github.com/gustaveems/parkscan-nl).

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
scripts/collect-candidates.mjs   PDOK reverse-geocode → data/candidates.csv (701 addresses)
label/index.html                 keyboard-driven label studio (localStorage → export CSV)
data/candidates.csv              Phase-0 candidate list (Amsterdam / Rotterdam / Utrecht)
docs/eval-protocol.md            frozen evaluation rules — written before any model is fit
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
- [ ] Phase 3b — fused model + geographic holdout + calibration — BAG registry features fetched (`scripts/fetch_registry.mjs` → `data/registry_features.csv`, 1,046/1,050: use, area, VBO status, build year, pand status; ownership form is BRK-only and unavailable keyless, so baseline 3 runs on the BAG set); still needs more Amsterdam positives
- [ ] Phase 4 — ParkScan demo upgrade (Rules v1 vs Oracle v2 toggle) + write-up

Part of the Gustave Soulas portfolio · successor to [parkscan-nl](https://github.com/gustaveems/parkscan-nl).

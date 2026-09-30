# VacancyOracle — Phase 1 baseline report

> **FIXTURE RUN** — synthetic labels, pipeline proof only. Re-run with real data/labels.csv.

- labeled rows: **174** (14 unclear excluded) → evaluable: **160**
- vacancy prevalence: **37.5%**
- per city: Amsterdam 52 rows/21 vacant, Rotterdam 47 rows/20 vacant, Utrecht 61 rows/19 vacant

## Baselines

1. **Majority class** — PR-AUC = prevalence = **0.375**
2. **Street-prior rules-lite** — PR-AUC **0.438** · p@r.5 0.38 · p@r.7 0.38 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 60 / TN 100)
3. **ParkScan rules engine** — pending `data/scores_rules_full.json` (needs image-evidence frontage context; arrives with Phase 2 active learning).

## Decision framing
FP = unwanted legal outreach mail. Demo threshold chosen at precision ≥ 90%; recall shown per baseline.

_Protocol frozen in `docs/eval-protocol.md` before model fitting._

# VacancyOracle — Phase 1 baseline report


- labeled rows: **701** (162 unclear excluded) → evaluable: **539**
- vacancy prevalence: **2.4%**
- per city: Amsterdam 216 rows/5 vacant, Rotterdam 168 rows/5 vacant, Utrecht 155 rows/3 vacant

## Baselines

1. **Majority class** — PR-AUC = prevalence = **0.024**
2. **Street-prior rules-lite** — PR-AUC **0.020** · p@r.5 0.02 · p@r.7 0.02 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 13 / TN 526)
3. **ParkScan rules engine (full)** — PR-AUC **0.032** · p@r.5 0.02 · p@r.7 0.02 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 13 / TN 526)
4. **SigLIP zero-shot vision** (local, no training) — PR-AUC **0.048** · p@r.5 0.02 · p@r.7 0.02 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 13 / TN 526)

## Decision framing
FP = unwanted legal outreach mail. Demo threshold chosen at precision ≥ 90%; recall shown per baseline.

_Protocol frozen in `docs/eval-protocol.md` before model fitting._

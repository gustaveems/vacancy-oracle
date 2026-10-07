# VacancyOracle — Phase 1 baseline report


- labeled rows: **1050** (207 unclear excluded) → evaluable: **843**
- vacancy prevalence: **3.1%**
- per city: Amsterdam 320 rows/5 vacant, Rotterdam 273 rows/5 vacant, Utrecht 250 rows/16 vacant

## Baselines

1. **Majority class** — PR-AUC = prevalence = **0.031**
2. **Street-prior rules-lite** — PR-AUC **0.021** · p@r.5 0.03 · p@r.7 0.03 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 26 / TN 817)
3. **ParkScan rules engine (full)** — PR-AUC **0.034** · p@r.5 0.03 · p@r.7 0.03 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 26 / TN 817)
4. **SigLIP zero-shot vision** (local, no training) — PR-AUC **0.082** · p@r.5 0.03 · p@r.7 0.03 · @p≥90%: recall 0.00 (TP 0 / FP 0 / FN 26 / TN 817)

## Decision framing
FP = unwanted legal outreach mail. Demo threshold chosen at precision ≥ 90%; recall shown per baseline.

_Protocol frozen in `docs/eval-protocol.md` before model fitting._

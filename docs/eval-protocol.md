# Evaluation Protocol (frozen before any model is fit)

## Task
Binary: is a street-facing ground-floor unit **vacant/underused** at photo time?
Human Street-View labels are ground truth. `unclear` labels are excluded from
metrics but retained for analysis.

## Splits
- **In-city holdout**: random 80/20 within each city — baseline metrics.
- **Geographic holdout (headline)**: train on Amsterdam labels, test on
  Rotterdam + Utrecht labels (and one rotated fold). City transfer is the
  generalization claim; in-city numbers are reported but never headline.

## Metrics
- Primary: **PR-AUC** (vacancy is the minority class).
- Secondary: precision@recall thresholds (0.5 / 0.7), calibration
  (Brier + reliability plot), per-class confusion.
- Decision framing: false positive = unwanted legal outreach mail →
  report the FP cost curve and pick the demo threshold at ≥90% precision.

## Baselines (run in this order)
1. Majority class
2. Rules-only (ParkScan `scoresite` vacancy half, unchanged)
3. Registry features only (use, build year, area, ownership form) — GBM
4. Image embeddings only (SigL2/CLIP zero-shot + linear probe)
5. Fusion (3+4) — the proposed model

Ablation = 5 minus each input; a model that can't beat 2 gets said so in the README.

## Data governance
- Labels: self-collected via public Street View for research; dataset published
  with address IDs under CC-BY-NC. No personal data beyond publicly registered
  business addresses.
- API keys never committed; localStorage-only in the label tool.

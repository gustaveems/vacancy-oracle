#!/bin/bash
# Auto-refresh pipeline: whenever a fresh label export lands in ~/Downloads, rerun everything.
# Browsers name re-exports "labels (1).csv", "labels (2).csv", … so watch the newest labels*.csv.
cd /Users/alain/Projects/vacancy-oracle || exit 1

newest() { ls -t "$HOME"/Downloads/labels*.csv 2>/dev/null | head -1; }

refresh() {
  cp "$1" data/labels.csv
  node scripts/rules_score.mjs >/dev/null 2>&1
  python3 scripts/embed_zero_shot.py >/dev/null 2>&1
  python3 scripts/make_active_batch.py >/dev/null 2>&1
  python3 scripts/phase1_baseline.py >/dev/null 2>&1
  node scripts/make_studio.mjs >/dev/null 2>&1
  python3 scripts/probe_supervised.py >/dev/null 2>&1
  python3 scripts/baseline3_gbm.py >/dev/null 2>&1
  python3 scripts/baseline5_fusion.py >/dev/null 2>&1
  git add -A 2>/dev/null && git commit -qm "Refresh: new label export — scores, reports, active batch, studio, phase3, baselines 3+5" 2>/dev/null && git push -q origin master:main 2>/dev/null
  echo "$(date +%H:%M) refreshed from $(basename "$1")" >> /tmp/watch.log
}

CUR_FILE=$(newest)
# ingest on startup only if the newest export differs from what's already in data/
if [ -n "$CUR_FILE" ] && ! cmp -s "$CUR_FILE" data/labels.csv; then
  sleep 2
  refresh "$CUR_FILE"
fi
LAST=$(stat -f %m "$CUR_FILE" 2>/dev/null || echo 0)

while true; do
  CUR_FILE=$(newest)
  CUR=$(stat -f %m "$CUR_FILE" 2>/dev/null || echo 0)
  if [ -n "$CUR_FILE" ] && [ "$CUR" != "$LAST" ]; then
    LAST=$CUR
    sleep 2
    refresh "$CUR_FILE"
  fi
  sleep 20
done

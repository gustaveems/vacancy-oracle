#!/bin/bash
# Auto-refresh pipeline: whenever a fresh labels.csv export lands, rerun everything.
cd /Users/alain/Projects/vacancy-oracle
LAST=""
while true; do
  CUR=$(stat -f %m ~/Downloads/labels.csv 2>/dev/null || echo 0)
  if [ "$CUR" != "$LAST" ] && [ -f ~/Downloads/labels.csv ]; then
    LAST=$CUR
    sleep 2
    cp ~/Downloads/labels.csv data/labels.csv
    node scripts/rules_score.mjs >/dev/null 2>&1
    python3 scripts/embed_zero_shot.py >/dev/null 2>&1
    python3 scripts/make_active_batch.py >/dev/null 2>&1
    python3 scripts/phase1_baseline.py >/dev/null 2>&1
    node scripts/make_studio.mjs >/dev/null 2>&1
    git add -A 2>/dev/null && git commit -qm "Refresh: new label export — rules/vision scores, report, active batch, studio" 2>/dev/null && git push -q origin master:main 2>/dev/null
    echo "$(date +%H:%M) refreshed" >> /tmp/watch.log
  fi
  sleep 20
done

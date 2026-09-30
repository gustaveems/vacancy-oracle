#!/bin/bash
# wait for the imagery fetch to finish, then run local zero-shot vision scoring
cd "$(dirname "$0")/.."
while pgrep -f "fetch_imagery" > /dev/null; do sleep 30; done
python3 scripts/embed_zero_shot.py >> /tmp/embed.log 2>&1

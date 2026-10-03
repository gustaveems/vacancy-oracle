#!/usr/bin/env python3
"""Rank unlabeled candidates by zero-shot vision score → data/active_batch.csv"""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
zs = json.load(open(ROOT/"data/vision_zeroshot_scores.json")) if (ROOT/"data/vision_zeroshot_scores.json").exists() else {}
labeled = {r["pdok_vbo"] for r in csv.DictReader(open(ROOT/"data/labels.csv"))}
unl = [(c["pdok_vbo"], zs.get(c["pdok_vbo"], 0), c["city"], c["address"])
       for c in csv.DictReader(open(ROOT/"data/candidates.csv")) if c["pdok_vbo"] not in labeled]
unl.sort(key=lambda t: -t[1])
with open(ROOT/"data/active_batch.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rank","pdok_vbo","vision_vacancy","city","address"])
    for i, r in enumerate(unl[:100], 1): w.writerow([i, *r])
print(f"active_batch: {len(unl)} unlabeled, top score {unl[0][1]:.2f}" if unl else "all labeled 🎉")

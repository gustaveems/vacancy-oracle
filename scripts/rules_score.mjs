#!/usr/bin/env node
/**
 * ParkScan rules-engine scores for every labeled candidate.
 * Uses the REAL recovered engine (scoringEngine.scoresite + mock places/vision
 * frontage) from parkscan-restored → data/scores_rules_full.json maps
 * pdok_vbo → vacancy probability (vacancyScore/100).
 */
import { readFileSync, writeFileSync } from "fs";
import { resolve } from "path";
const ROOT = resolve(import.meta.dirname, "..");
const PS = resolve(ROOT, "../parkscan-restored/frontend/src/lib");

const { scoresite } = await import(resolve(PS, "demo/scoringEngine.js"));
const { generateNearbyPOIs } = await import(resolve(PS, "demo/mockData.js"));
const { analyzeFrontage } = await import(resolve(PS, "demo/vision.js"));

const lines = readFileSync(resolve(ROOT, "data/labels.csv"), "utf8").replace(/\r/g, "").trim().split("\n").slice(1);
const out = {};
let done = 0;
for (const line of lines) {
  const m = line.match(/^(\w+),([\d\w]+),"([^"]*)",([\d.]+),([\d.]+),(\d)$/);
  if (!m) continue;
  const [, city, pdok, address, lat, lng] = m;
  const site = {
    id: pdok, city, address, lat: +lat, lng: +lng,
    buildYear: null, areaSqm: null, bagStatus: null, usePurpose: null, status: "identified",
  };
  const context = generateNearbyPOIs(+lat, +lng);
  let frontage = null;
  try { frontage = await analyzeFrontage(+lat, +lng); } catch { /* mock-only env */ }
  const s = scoresite(site, context, { frontage });
  out[pdok] = (s.vacancyScore ?? 0) / 100;
  if (++done % 250 === 0) console.log(`scored ${done}`);
}
writeFileSync(resolve(ROOT, "data/scores_rules_full.json"), JSON.stringify(out));
console.log(`wrote data/scores_rules_full.json (${done} buildings, recovered ParkScan engine)`);

#!/usr/bin/env node
/**
 * vacancy-oracle · Amsterdam positive hunt — step 1: more candidates.
 * Same PDOK Locatieserver reverse-geocode as collect-candidates.mjs, but over
 * Amsterdam districts the first pass never seeded (business parks, Westpoort,
 * Zuidoost, Nieuw-West, Noord). APPENDS to data/candidates.csv — existing rows
 * and their order are never touched; addresses already present are skipped.
 *
 * Usage: node scripts/hunt_amsterdam.mjs [--max 400]
 */
import { readFileSync, appendFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const MAX = Number(args[args.indexOf("--max") + 1] || 400) || 400;

const SEEDS = [
  [52.3889, 4.8380], [52.3925, 4.8150], [52.3870, 4.8470], [52.3290, 4.9100],
  [52.3420, 4.9170], [52.3130, 4.9440], [52.3160, 4.9530], [52.3000, 4.9600],
  [52.4000, 4.9200], [52.3870, 4.9170], [52.3420, 4.8090], [52.3570, 4.8340],
  [52.3600, 4.8000], [52.3580, 4.8250], [52.3390, 4.8730], [52.3660, 4.8680],
  [52.3630, 4.9390], [52.3790, 4.8480], [52.3680, 4.9640], [52.3990, 4.8700],
];
const FL = "id,weergavenaam,centroide_ll,adresseerbaarobject_id,straatnaam,huisnummer,postcode";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const wkt = (w) => { const m = /POINT\s*\(([\d.+-]+)\s+([\d.+-]+)\)/.exec(w || ""); return m ? { lng: +m[1], lat: +m[2] } : null; };

const existing = new Set(readFileSync(resolve(ROOT, "data/candidates.csv"), "utf8").trim().split("\n").slice(1)
  .map((l) => (l.match(/("([^"]*)")|([^,]*)/g).filter(Boolean)[1] || "")));

const fresh = new Map();
const bySeed = [];
for (const [lat, lng] of SEEDS) {
  let docs = [];
  for (let attempt = 0; attempt < 3 && !docs.length; attempt++) {
    try {
      const url = `https://api.pdok.nl/bzk/locatieserver/search/v3_1/reverse?lat=${lat}&lon=${lng}` +
        `&distance=900&rows=50&fq=type:adres&fl=${FL}&wt=json`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(String(res.status));
      docs = (await res.json())?.response?.docs || [];
    } catch { await sleep(1500 * (attempt + 1)); }
  }
  const mine = [];
  for (const d of docs) {
    const vbo = d.adresseerbaarobject_id; const ll = wkt(d.centroide_ll);
    // 0363 = gemeente Amsterdam; reverse search also returns Ouder-Amstel (0437) etc.
    if (!vbo || !vbo.startsWith("0363") || !ll || existing.has(vbo) || fresh.has(vbo)) continue;
    const row = { city: "Amsterdam", pdok_vbo: vbo,
      address: d.weergavenaam || `${d.straatnaam} ${d.huisnummer}, ${d.postcode} Amsterdam`,
      lat: ll.lat.toFixed(6), lng: ll.lng.toFixed(6) };
    fresh.set(vbo, row); mine.push(row);
  }
  bySeed.push(mine);
  console.log(`seed ${lat},${lng}: +${mine.length} (total new ${fresh.size})`);
  await sleep(150);
}

// round-robin across seeds so the cap keeps every district represented
const rows = [];
for (let i = 0; rows.length < MAX && bySeed.some((s) => i < s.length); i++)
  for (const s of bySeed) if (i < s.length && rows.length < MAX) rows.push(s[i]);
const q = (v) => (String(v).includes(",") ? `"${v}"` : v);
if (rows.length) appendFileSync(resolve(ROOT, "data/candidates.csv"),
  "\n" + rows.map((r) => [r.city, r.pdok_vbo, r.address, r.lat, r.lng].map(q).join(",")).join("\n"));
console.log(`appended ${rows.length} new Amsterdam candidates to data/candidates.csv`);

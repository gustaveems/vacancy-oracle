#!/usr/bin/env node
/**
 * vacancy-oracle · Phase 2 prep — bulk street-level imagery
 * Downloads 4 heading views (N/E/S/W) per candidate to data/imagery/,
 * resumable (skips existing files), rate-limited (12 qps-ish), honest
 * about Google's static-view quota (~25k/day free tier).
 *
 * Usage:
 *   node scripts/fetch_imagery.mjs            # all candidates
 *   node scripts/fetch_imagery.mjs --city Amsterdam --limit 50
 */
import { mkdirSync, existsSync, writeFileSync, readFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const KEY = JSON.parse(readFileSync(resolve(ROOT, "data/config.json"), "utf8")).gsv_key;
if (!KEY) { console.error("missing data/config.json with {gsv_key}"); process.exit(1); }

const OUT = resolve(ROOT, "data/imagery");
mkdirSync(OUT, { recursive: true });

const args = process.argv.slice(2);
const arg = (n) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : null; };
const CITY = arg("--city");
const LIMIT = Number(arg("--limit") || Infinity);
const SIZE = "640x480";
const HEADINGS = [0, 90, 180, 270];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const lines = readFileSync(resolve(ROOT, "data/candidates.csv"), "utf8").trim().split("\n").slice(1);
const rows = lines
  .map((l) => {
    const c = l.match(/("([^"]*)")|([^,]*)/g).map((x) => (x.startsWith('"') ? x.slice(1, -1) : x));
    const [city, pdok, addr, lat, lng] = c.filter(Boolean);
    return { city, pdok, addr, lat, lng };
  })
  .filter((r) => !CITY || r.city === CITY)
  .slice(0, LIMIT);

const manifestPath = resolve(OUT, "manifest.csv");
const manifest = existsSync(manifestPath)
  ? new Set(readFileSync(manifestPath, "utf8").trim().split("\n").slice(1))
  : new Set(["city,pdok_vbo,heading,file,status"]);
if (!existsSync(manifestPath)) manifest.forEach((l) => {});

let done = 0, skipped = 0, denied = 0;
const t0 = Date.now();
for (const r of rows) {
  for (const h of HEADINGS) {
    const file = `${r.pdok}_${h}.jpg`;
    const key = `${r.city},${r.pdok},${h},data/imagery/${file}`;
    if (manifest.has(key) || existsSync(resolve(OUT, file))) { skipped++; continue; }
    const url =
      `https://maps.googleapis.com/maps/api/streetview?size=${SIZE}&location=${r.lat},${r.lng}` +
      `&heading=${h}&pitch=6&key=${KEY}`;
    try {
      const res = await fetch(url);
      const buf = Buffer.from(await res.arrayBuffer());
      const isJpeg = buf[0] === 0xff && buf[1] === 0xd8;
      const status = res.ok && isJpeg ? "ok" : res.status === 403 ? "denied" : res.ok ? "no_imagery" : `http_${res.status}`;
      if (isJpeg) writeFileSync(resolve(OUT, file), buf);
      if (status === "denied") { denied++; }
      manifest.add(`${r.city},${r.pdok},${h},data/imagery/${file},${status}`);
      done++;
      if (done % 50 === 0) {
        console.log(`${done} fetched (${denied} denied) — ${((Date.now() - t0) / 1000).toFixed(0)}s, ${rows.length * 4} total`);
      }
    } catch (e) {
      manifest.add(`${r.city},${r.pdose ?? r.pdok},${h},,error`);
      denied++;
    }
    await sleep(70);
  }
}
writeFileSync(manifestPath, [...manifest].join("\n"));
console.log(`Fetched ${done}, skipped ${skipped}, denied ${denied} → data/imagery/ + manifest.csv`);
if (denied) console.log("⚠ denied = Street View Static API not enabled/billed on this key — enable 'Maps Street View Static API' in console.");

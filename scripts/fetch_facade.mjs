#!/usr/bin/env node
/**
 * vacancy-oracle · facade-aimed imagery (fixes what the sign-reader audit found)
 *
 * fetch_imagery.mjs shoots 4 fixed compass headings at 90° FOV from wherever
 * Google snaps the address, and saves the "no imagery here" placeholder as a
 * normal JPEG. This fetcher instead:
 *   1. asks the Street View *metadata* endpoint first (no charge, no image quota):
 *      ZERO_RESULTS → logged as no_imagery, nothing downloaded; non-Google
 *      copyright → user photo, skipped unless --allow-user
 *   2. computes the bearing from the panorama to the building and aims at the facade
 *   3. zooms in (FOV 45°, ≈2× the pixels on the shopfront) — with --flank, two extra
 *      views at ±30° catch boards on the edge of the unit
 *   4. logs the panorama capture date: labels mean "vacant at photo time"
 *
 * Output: data/imagery_facade/<vbo>[_l|_r].jpg + manifest.csv (resumable).
 * Usage:
 *   node scripts/fetch_facade.mjs --dry-run            # metadata only, 0 image requests
 *   node scripts/fetch_facade.mjs --unclear-only       # the label-3 re-pass set
 *   node scripts/fetch_facade.mjs --city Amsterdam --limit 50 --flank
 */
import { mkdirSync, existsSync, writeFileSync, readFileSync, appendFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const KEY = JSON.parse(readFileSync(resolve(ROOT, "data/config.json"), "utf8")).gsv_key;
if (!KEY) { console.error("missing data/config.json with {gsv_key}"); process.exit(1); }

const OUT = resolve(ROOT, "data/imagery_facade");
mkdirSync(OUT, { recursive: true });

const args = process.argv.slice(2);
const arg = (n) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : null; };
const flag = (n) => args.includes(n);
const CITY = arg("--city");
const LIMIT = Number(arg("--limit") || Infinity);
const DRY = flag("--dry-run");
const FLANK = flag("--flank");
const ALLOW_USER = flag("--allow-user");
const UNCLEAR_ONLY = flag("--unclear-only");
const SIZE = "640x640";
const FOV = 45;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const parseCsv = (path) =>
  readFileSync(path, "utf8").trim().split("\n").slice(1).map((l) =>
    l.match(/("([^"]*)")|([^,]*)/g).filter(Boolean).map((x) => (x.startsWith('"') ? x.slice(1, -1) : x)));

const unclear = new Set(parseCsv(resolve(ROOT, "data/labels.csv")).filter((c) => c[5] === "3").map((c) => c[1]));
const rows = parseCsv(resolve(ROOT, "data/candidates.csv"))
  .map(([city, pdok, addr, lat, lng]) => ({ city, pdok, addr, lat: +lat, lng: +lng }))
  .filter((r) => (!CITY || r.city === CITY) && (!UNCLEAR_ONLY || unclear.has(r.pdok)))
  .slice(0, LIMIT);

const manifestPath = resolve(OUT, "manifest.csv");
const HEADER = "pdok_vbo,status,pano_id,pano_date,distance_m,heading,copyright,files";
if (!existsSync(manifestPath)) writeFileSync(manifestPath, HEADER + "\n");
const seen = new Set(readFileSync(manifestPath, "utf8").trim().split("\n").slice(1).map((l) => l.split(",")[0]));

const rad = (d) => (d * Math.PI) / 180;
function bearing(a, b) {
  const y = Math.sin(rad(b.lng - a.lng)) * Math.cos(rad(b.lat));
  const x = Math.cos(rad(a.lat)) * Math.sin(rad(b.lat)) - Math.sin(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.cos(rad(b.lng - a.lng));
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}
function metres(a, b) {
  const dLat = rad(b.lat - a.lat), dLng = rad(b.lng - a.lng);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 6371000 * 2 * Math.asin(Math.sqrt(h));
}

const stats = { ok: 0, no_imagery: 0, user_photo: 0, error: 0, images: 0 };
async function one(r) {
  const base = "https://maps.googleapis.com/maps/api/streetview";
  const meta = await (await fetch(`${base}/metadata?location=${r.lat},${r.lng}&radius=50&source=outdoor&key=${KEY}`)).json();
  if (meta.status !== "OK") {
    stats[meta.status === "ZERO_RESULTS" ? "no_imagery" : "error"]++;
    return [r.pdok, meta.status === "ZERO_RESULTS" ? "no_imagery" : `error_${meta.status}`, "", "", "", "", "", ""];
  }
  const pano = meta.location;
  const copy = (meta.copyright || "").replace(/,/g, " ");
  const head = Math.round(bearing(pano, r));
  const dist = Math.round(metres(pano, r));
  if (!/google/i.test(copy) && !ALLOW_USER) {
    stats.user_photo++;
    return [r.pdok, "user_photo", meta.pano_id, meta.date || "", dist, head, copy, ""];
  }
  const files = [];
  if (!DRY) {
    const shots = FLANK ? [["", head], ["_l", (head + 330) % 360], ["_r", (head + 30) % 360]] : [["", head]];
    for (const [suffix, h] of shots) {
      const res = await fetch(`${base}?size=${SIZE}&pano=${meta.pano_id}&heading=${h}&fov=${FOV}&pitch=8&key=${KEY}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (res.ok && buf[0] === 0xff && buf[1] === 0xd8) {
        writeFileSync(resolve(OUT, `${r.pdok}${suffix}.jpg`), buf);
        files.push(`${r.pdok}${suffix}.jpg`); stats.images++;
      }
      await sleep(60);
    }
  }
  stats.ok++;
  return [r.pdok, DRY ? "ok_dry" : "ok", meta.pano_id, meta.date || "", dist, head, copy, files.join(";")];
}

const todo = rows.filter((r) => DRY || !seen.has(r.pdok));
console.log(`${todo.length} candidates (${rows.length - todo.length} already in manifest)${DRY ? " — DRY RUN, metadata only" : ""}`);
const t0 = Date.now();
let cursor = 0, done = 0;
const dryRows = [];
async function worker() {
  while (cursor < todo.length) {
    const r = todo[cursor++];
    let row;
    try { row = await one(r); } catch (e) { stats.error++; row = [r.pdok, "error", "", "", "", "", "", ""]; }
    if (DRY) dryRows.push(row.join(",")); else appendFileSync(manifestPath, row.join(",") + "\n");
    if (++done % 50 === 0) console.log(`  ${done}/${todo.length} — ${JSON.stringify(stats)} — ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    await sleep(60);
  }
}
await Promise.all(Array.from({ length: 6 }, worker));
if (DRY) writeFileSync(resolve(OUT, "metadata_dry_run.csv"), HEADER + "\n" + dryRows.join("\n") + "\n");
console.log(`done — ${JSON.stringify(stats)}${DRY ? " → data/imagery_facade/metadata_dry_run.csv" : ""}`);

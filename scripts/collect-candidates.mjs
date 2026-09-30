#!/usr/bin/env node
/**
 * vacancy-oracle · Phase 0 — candidate collection
 * Seeds commercial districts in Amsterdam / Rotterdam / Utrecht and reverse-
 * geocodes real BAG addresses via PDOK Locatieserver (the same live endpoint
 * ParkScan uses). Output: data/candidates.csv
 *
 * Usage:  node scripts/collect-candidates.mjs
 */
import { writeFileSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// (lat, lng) seeds — downtown/commercial cores + inner districts per city.
const SEEDS = {
  Amsterdam: [
    [52.3702, 4.8952], [52.3770, 4.9010], [52.3876, 4.9033], [52.3638, 4.9186],
    [52.3568, 4.8922], [52.3727, 4.8865], [52.3755, 4.9300], [52.3920, 4.9100],
    [52.3500, 4.8600], [52.3460, 4.8970], [52.3810, 4.8730], [52.3676, 4.9360],
    // industrial belts — the vacancy-rich grounds
    [52.3947, 4.7851], [52.3793, 4.7949], [52.3920, 4.9050], [52.3330, 4.8610],
    [52.4040, 4.9100], [52.3576, 4.9560],
  ],
  Rotterdam: [
    [51.9244, 4.4777], [51.9200, 4.4800], [51.9300, 4.4700], [51.9180, 4.4900],
    [51.9220, 4.5200], [51.9090, 4.4870], [51.9060, 4.4660], [51.9350, 4.4900],
    [51.8950, 4.4750], [51.9450, 4.4600], [51.9240, 4.4580], [51.9120, 4.5030],
    // Waalhaven / Eemhaven / Merwe-Vierhavens / Spaanse Polder / Botlek rim
    [51.8930, 4.5180], [51.8820, 4.5050], [51.9020, 4.5600], [51.9180, 4.4320],
    [51.8880, 4.4520], [51.9350, 4.5350],
  ],
  Utrecht: [
    [52.0907, 5.1214], [52.0950, 5.1100], [52.0850, 5.1350], [52.1000, 5.1200],
    [52.0800, 5.1050], [52.1100, 5.1400], [52.0720, 5.1230], [52.0960, 5.1500],
    [52.0870, 5.0960], [52.1180, 5.1090], [52.1040, 5.0980], [52.0930, 5.1700],
    // Lage Weide / Maarssen / Leidsche Rijn commercial strips
    [52.0660, 5.0780], [52.1050, 5.1100], [52.1020, 5.0300], [52.0580, 5.0900],
    [52.1230, 5.0750], [52.0790, 5.1650],
  ],
};

const COMMERCIAL = ["kantoorfunctie", "winkelfunctie", "industriefunctie", "celfunctie", "overige gebruiksfunctie"];
const FL = "id,weergavenaam,centroide_ll,adresseerbaarobject_id,nummeraanduiding_id,straatnaam,huisnummer,postcode,woonplaatsnaam,gemeentenaam";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function wkt(w) {
  const m = /POINT\s*\(([\d.+-]+)\s+([\d.+-]+)\)/.exec(w || "");
  return m ? { lng: +m[1], lat: +m[2] } : null;
}

async function reverse(lat, lng, distance = 900, rows = 20) {
  const url =
    `https://api.pdok.nl/bzk/locatieserver/search/v3_1/reverse` +
    `?lat=${lat}&lon=${lng}&distance=${distance}&rows=${rows}&fq=type:adres&fl=${FL}&wt=json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}`);
  const j = await res.json();
  return j?.response?.docs || [];
}

const out = new Map();
for (const [city, seeds] of Object.entries(SEEDS)) {
  let added = 0;
  for (const [lat, lng] of seeds) {
    let docs = [];
    for (let attempt = 0; attempt < 3 && !docs.length; attempt++) {
      try {
        docs = await reverse(lat, lng);
      } catch {
        await sleep(1500 * (attempt + 1));
      }
    }
    for (const d of docs) {
      const ll = wkt(d.centroide_ll);
      if (!ll) continue;
      const key = d.adresseerbaarobject_id || `${d.straatnaam}|${d.huisnummer}`;
      if (out.has(key)) continue;
      out.set(key, {
        city,
        pdok_vbo: d.adresseerbaarobject_id || "",
        address: d.weergavenaam || `${d.straatnaam} ${d.huisnummer}, ${d.postcode} ${city}`,
        lat: ll.lat.toFixed(6),
        lng: ll.lng.toFixed(6),
      });
      added++;
    }
    await sleep(120);
  }
  console.log(`${city}: +${added} (running total ${out.size})`);
}

// Keep a manageable, city-balanced file: cap per city.
const perCity = {};
const rowsOut = [];
for (const r of out.values()) {
  perCity[r.city] = (perCity[r.city] || 0) + 1;
  if (perCity[r.city] <= 550) rowsOut.push(r);
}
rowsOut.sort((a, b) => (a.city === b.city ? a.address.localeCompare(b.address) : a.city.localeCompare(b.city)));

mkdirSync(resolve(ROOT, "data"), { recursive: true });
const header = "city,pdok_vbo,address,lat,lng";
const csv = [header, ...rowsOut.map((r) =>
  Object.values(r).map((v) => (String(v).includes(",") ? `"${v}"` : v)).join(",")
)].join("\n");
writeFileSync(resolve(ROOT, "data/candidates.csv"), csv);
console.log(`Wrote data/candidates.csv — ${rowsOut.length} addresses, target mix: kantoor/winkel/industrie (${COMMERCIAL.length} uses)`);

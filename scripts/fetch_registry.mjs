#!/usr/bin/env node
/**
 * vacancy-oracle · BAG registry features fetcher
 * Batched WFS 2.0 POST queries against the keyless PDOK BAG WFS (v2_0) — the
 * GET interface ignores filter params, POST with fes:Filter works. Fetches
 * per-VBO registry attributes for protocol baseline 3/5:
 *   use (gebruiksdoel), area (oppervlakte), VBO status, build year (bouwjaar,
 *   via the related pand), pand status.
 * Ownership form is NOT in BAG (BRK/Kadaster, API-key-only) — baseline 3 runs
 * on the BAG-available set; the limitation is noted in the README, the frozen
 * eval-protocol.md stays untouched.
 *
 * IDs missing from the verblijfsobject layer are retried against the ligplaats
 * and standplaats layers (houseboats / pitches — BAG id segment 02/03); their
 * object type lands in `use`, layer status in `vbo_status`.
 *
 * Resumable: ids with non-empty features in data/registry_features.csv are
 * skipped. Ids absent from all layers stay as empty rows (ended registrations).
 *
 * Usage:  node scripts/fetch_registry.mjs
 */
import { readFileSync, writeFileSync, appendFileSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = resolve(ROOT, "data/registry_features.csv");
const WFS = "https://service.pdok.nl/lv/bag/wfs/v2_0";
const BATCH = 25;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// candidates.csv: city,pdok_vbo,address (quoted, may contain commas),lat,lng
const lines = readFileSync(resolve(ROOT, "data/candidates.csv"), "utf8").trim().split("\n").slice(1);
const ids = lines.map((l) => l.match(/^[^,]*,([^,]*)/)[1]).filter(Boolean);

const done = new Set();
const existing = new Map();
if (existsSync(OUT)) {
  for (const l of readFileSync(OUT, "utf8").trim().split("\n").slice(1)) {
    const c = l.split(",");
    existing.set(c[0], c);
    if (c[1] || c[3]) done.add(c[0]);
  }
}
const todo = ids.filter((id) => !done.has(id));
console.log(`${ids.length} candidates, ${done.size} already fetched, ${todo.length} to go`);

const HEAD = "pdok_vbo,use,area,vbo_status,build_year,pand_status,pand_id";
if (!existsSync(OUT)) writeFileSync(OUT, HEAD + "\n");

const esc = (v) => (v == null ? "" : String(v).replace(/,/g, ";").replace(/\s+/g, " ").trim());

function batchXml(batch, layer) {
  const clauses = batch
    .map((id) => `<fes:PropertyIsEqualTo><fes:ValueReference>identificatie</fes:ValueReference><fes:Literal>${id}</fes:Literal></fes:PropertyIsEqualTo>`)
    .join("");
  return `<?xml version="1.0"?><wfs:GetFeature service="WFS" version="2.0.0" outputFormat="application/json" xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:fes="http://www.opengis.net/fes/2.0"><wfs:Query typeNames="${layer}"><fes:Filter><fes:Or>${clauses}</fes:Or></fes:Filter></wfs:Query></wfs:GetFeature>`;
}

async function postBatch(batch, layer) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(WFS, {
        method: "POST",
        headers: { "Content-Type": "application/xml" },
        body: batchXml(batch, layer),
        signal: AbortSignal.timeout(30000),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const j = await res.json();
      return new Map((j.features || []).map((f) => [f.properties.identificatie, f.properties]));
    } catch {
      await sleep(1500 * (attempt + 1));
    }
  }
  throw new Error(`batch failed after 3 attempts (${batch[0]}…)`);
}

// ---- pass 1: verblijfsobject layer (append checkpointed for resume) ----
let fetched = 0;
let absent = [];
const nBatches = Math.ceil(todo.length / BATCH);
for (let i = 0; i < todo.length; i += BATCH) {
  const batch = todo.slice(i, i + BATCH);
  const byId = await postBatch(batch, "bag:verblijfsobject");
  const rows = batch.map((id) => {
    const p = byId.get(id);
    if (!p) { absent.push(id); return [id, "", "", "", "", "", ""].join(","); }
    const use = Array.isArray(p.gebruiksdoel) ? p.gebruiksdoel.join("|") : p.gebruiksdoel;
    return [id, esc(use), p.oppervlakte ?? "", esc(p.status), p.bouwjaar ?? "", esc(p.pandstatus), p.pandidentificatie ?? ""].join(",");
  });
  appendFileSync(OUT, rows.join("\n") + "\n");
  for (const r of rows) {
    const c = r.split(",");
    existing.set(c[0], c);
    if (c[1] || c[3]) fetched++;
  }
  console.log(`batch ${Math.floor(i / BATCH) + 1}/${nBatches} — ${done.size + fetched}/${ids.length} rows`);
  await sleep(200);
}

// ---- pass 2: ligplaats / standplaats for ids absent from the VBO layer ----
if (todo.length) absent = ids.filter((id) => { const c = existing.get(id); return c && !c[1] && !c[3]; });
if (absent.length) {
  console.log(`${absent.length} ids not in verblijfsobject — trying ligplaats/standplaats layers`);
  let filled = 0;
  for (const layer of ["bag:ligplaats", "bag:standplaats"]) {
    for (let i = 0; i < absent.length; i += BATCH) {
      const batch = absent.slice(i, i + BATCH);
      const byId = await postBatch(batch, layer);
      const type = layer === "bag:ligplaats" ? "ligplaats" : "standplaats";
      for (const id of batch) {
        const p = byId.get(id);
        if (p && !existing.get(id)[1]) {
          existing.set(id, [id, type, "", esc(p.status), "", "", ""]);
          filled++;
        }
      }
      await sleep(200);
    }
  }
  writeFileSync(OUT, HEAD + "\n" + ids.map((id) => existing.get(id).join(",")).join("\n") + "\n");
  console.log(`filled ${filled} of ${absent.length} from ligplaats/standplaats layers`);
}

const finalAbsent = ids.filter((id) => { const c = existing.get(id); return c && !c[1] && !c[3]; });
console.log(`Wrote data/registry_features.csv — ${ids.length - finalAbsent.length}/${ids.length} candidates with registry features, ${finalAbsent.length} absent from BAG (ended registrations)`);

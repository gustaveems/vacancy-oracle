#!/usr/bin/env node
/** Rebuild the Desktop label app from current candidates + local key. Never commits secrets. */
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = process.env.STUDIO_OUT || `${process.env.HOME}/Desktop/Vacancy Label Studio.html`;
const key = JSON.parse(readFileSync(resolve(ROOT, "data/config.json"), "utf8")).gsv_key;
const cands = readFileSync(resolve(ROOT, "data/candidates.csv"), "utf8").trim().split("\n").slice(1)
  .map((l) => {
    const c = l.match(/("([^"]*)")|([^,]*)/g).map((x) => (x.startsWith('"') ? x.slice(1, -1) : x));
    const [city, pdok, addr, lat, lng] = c.filter(Boolean);
    return { city, pdok, addr, lat, lng };
  });
import { existsSync } from "fs";
let scores = "{}", active = "[]";
if (existsSync(resolve(ROOT, "data/vision_zeroshot_scores.json")))
  scores = readFileSync(resolve(ROOT, "data/vision_zeroshot_scores.json"), "utf8");
if (existsSync(resolve(ROOT, "data/active_batch.csv"))) {
  const rows = readFileSync(resolve(ROOT, "data/active_batch.csv"), "utf8").trim().split("\n").slice(1);
  active = JSON.stringify(rows.map(l => l.split(",")[1]));
}
let tpl = readFileSync(resolve(ROOT, "templates/studio.template.html"), "utf8");
tpl = tpl.replace("__ROWS__", JSON.stringify(cands)).replace("__KEY__", key)
         .replace("__VSCORE__", scores).replace("__ACTIVE__", active);
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, tpl);
console.log(`${OUT} rebuilt — ${cands.length} candidates`);

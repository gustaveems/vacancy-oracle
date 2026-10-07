#!/usr/bin/env node
/**
 * Phase 4 — rebuild the Oracle demo (Rules v1 vs Oracle v2 toggle).
 * Same artifact pattern as make_studio.mjs: local HTML on the Desktop, imagery
 * via absolute file:// paths to data/imagery (gitignored, never ships).
 * No API keys touch this file — labels + scores are all local data.
 *
 * Usage:  node scripts/make_demo.mjs          → ~/Desktop/VacancyOracle Demo.html
 *         DEMO_OUT=/path/to/file.html node scripts/make_demo.mjs
 */
import { readFileSync, writeFileSync, readdirSync, existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = process.env.DEMO_OUT || `${process.env.HOME}/Desktop/VacancyOracle Demo.html`;

const imgDir = resolve(ROOT, "data/imagery");
const byB = {};
if (existsSync(imgDir))
  for (const f of readdirSync(imgDir)) {
    const pdok = f.replace(/_\d+\.jpg$/, "");
    (byB[pdok] ||= []).push(f);
  }

const parseCsv = (p) => {
  const [head, ...rest] = readFileSync(p, "utf8").replace(/\r/g, "").trim().split("\n");
  const cols = head.split(",");
  return rest.map((l) => {
    const cells = l.match(/("([^"]*)")|([^,]*)/g).map((x) => (x.startsWith('"') ? x.slice(1, -1) : x)).filter((x) => x !== "");
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
};

const cands = parseCsv(resolve(ROOT, "data/candidates.csv"));
const scores = parseCsv(resolve(ROOT, "data/oracle_scores.csv"));
const sc = Object.fromEntries(scores.map((s) => [s.pdok_vbo, s]));

const data = cands
  .filter((c) => sc[c.pdok_vbo] && (byB[c.pdok_vbo] || []).length && sc[c.pdok_vbo].label !== "")
  .map((c) => {
    const s = sc[c.pdok_vbo];
    return {
      pdok: c.pdok_vbo,
      addr: c.address,
      city: c.city,
      label: s.label,
      oracle: +s.oracle,
      rules: +(s.rules || 0),
      imgs: byB[c.pdok_vbo].sort(),
    };
  })
  .sort((a, b) => b.oracle - a.oracle);

if (!data.length) { console.error("no candidates with scores + imagery — run oracle_score.py first"); process.exit(1); }

let tpl = readFileSync(resolve(ROOT, "templates/demo.template.html"), "utf8");
tpl = tpl
  .replace("__DATA__", JSON.stringify(data))
  .replace("__IMG__", `file://${imgDir}/`)
  .replace("__GEN__", new Date().toISOString().slice(0, 10));
writeFileSync(OUT, tpl);
console.log(`${OUT} rebuilt — ${data.length} candidates, ${data.filter(d => d.label === "1").length} vacant labeled`);

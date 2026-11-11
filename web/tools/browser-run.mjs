#!/usr/bin/env node
// Run WebGPU test pages in Chromium and print their JSON results.
//
//   node tools/browser-run.mjs <page> [--json out.json] [--timeout ms]
//   node tools/browser-run.mjs --all            # every page in DEFAULT_PAGES, sequentially
//
// A page sets window.__testReady = true and exposes window.__runTests() returning
// { passed, failed, results }. Env: CHROME_PATH, WEBGPU_SOFTWARE=1 (SwiftShader), DISPLAY.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium } from "./chromium.mjs";
import { out, parseArgs } from "./cli.mjs";
import { serve } from "./server.mjs";

const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DEFAULT_PAGES = [
  "packages/wasm-oversampler/test/browser/oversampler.html",
  "packages/inference/test/browser/placement.html",
  "packages/inference/test/browser/head.html",
  "packages/inference/test/browser/engine.html",
  "packages/viewer/test/browser/viewer.html",
  "packages/viewer/test/browser/splat2d.html",
  "packages/viewer/test/browser/element.html",
  "test/e2e.html?fixture=bench_128&states=7",
  "test/e2e.html?fixture=bench_input_512&states=7,5",
];

async function runPage(page, { timeout }) {
  const { browser, close } = await launchChromium({ protocolTimeout: timeout + 60000 });
  const { server, port } = await serve({ root: webRoot });
  let result = { failed: 1, error: "no result" };
  try {
    const tab = await browser.newPage();
    tab.setDefaultTimeout(timeout);
    tab.on("console", (m) => { const t = m.text(); if (!/favicon/.test(t)) out("  [page] " + t.slice(0, 500)); });
    tab.on("pageerror", (e) => out("  [pageerror] " + String(e).slice(0, 500)));
    await tab.goto(`http://127.0.0.1:${port}/${page}`);
    await tab.waitForFunction("window.__testReady === true", { timeout: 60000 });
    result = await tab.evaluate(async () => { try { return await window.__runTests(); } catch (e) { return { failed: 1, error: String(e), stack: String(e.stack).slice(0, 2000) }; } });
  } catch (e) {
    result = { failed: 1, error: String(e).slice(0, 1000) };
  } finally {
    await close();
    server.close();
  }
  return result;
}

async function main() {
  const args = parseArgs(process.argv.slice(2), { json: null, timeout: "600000", all: false });
  const jsonOut = args.json;
  const timeout = Number(args.timeout);
  const pages = args.all ? DEFAULT_PAGES : args._;
  if (pages.length === 0) { out("usage: browser-run.mjs <page> [--json file] [--timeout ms] | --all"); process.exit(2); }
  let failed = 0;
  const summary = [];
  for (const page of pages) {
    out(`== ${page}`);
    const result = await runPage(page, { timeout });
    failed += result.failed ? 1 : 0;
    summary.push({ page, passed: result.passed ?? 0, failed: result.failed ?? 0, error: result.error });
    if (pages.length === 1) { const text = JSON.stringify(result, null, 1); if (jsonOut) fs.writeFileSync(jsonOut, text); out(text); }
  }
  if (pages.length > 1) { for (const s of summary) out(`${s.failed ? "FAIL" : "PASS"} ${s.page}: ${s.passed} passed, ${s.failed} failed${s.error ? " " + s.error : ""}`); if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(summary, null, 1)); }
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { out(String(e)); process.exit(1); });

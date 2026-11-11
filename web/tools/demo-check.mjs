#!/usr/bin/env node
// Drive the built demo end to end: load the page, wait for the automatic run of the sample
// image (there is no Run button), optionally change a control for a second run, read the stats.
//
//   node tools/demo-check.mjs                       # serves apps/demo/dist locally
//   node tools/demo-check.mjs --url https://host:8443/ [--shot prefix] [--theme dark|light] [--scale S] [--states K] [--sample kodim07]
// --sample pins the Kodak image (the page picks one at random otherwise).
// --scale S sets the count slider to S times the model's own count (0.25..4); the default run is x1.
// --states sets the refinements slider (K - 1); the check then scrubs to the first kept refinement and back.
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium } from "./chromium.mjs";
import { out, parseArgs } from "./cli.mjs";
import { serve } from "./server.mjs";

const dist = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../apps/demo/dist");
const args = parseArgs(process.argv.slice(2), { url: null, shot: null, theme: null, scale: null, states: null, sample: null });

/** Runs finished so far: the page increments data-runs on the status line after every run. */
const runsDone = (page) => page.evaluate(() => Number(document.getElementById("runStatus").dataset.runs ?? 0));
const waitForRun = (page, n) => page.waitForFunction(
  (n) => Number(document.getElementById("runStatus").dataset.runs ?? 0) >= n && /^(Done|Run failed)/.test(document.getElementById("runStatus").textContent),
  { timeout: 300000 }, n);
const text = (page, id) => page.$eval(`#${id}`, (e) => e.textContent);
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

/** Change the refinements and count controls as a user would and wait for the run they trigger. */
async function rerunWith(page, { states, scale }) {
  const before = await runsDone(page);
  if (states) await page.evaluate((v) => { const el = document.getElementById("refine"); el.value = String(v); el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); }, Number(states));
  // The count slider is logarithmic: value v gives 2^(v/50).
  if (scale) await page.evaluate((s) => { const el = document.getElementById("count"); el.value = String(Math.round(50 * Math.log2(s))); el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); }, Number(scale));
  const t0 = Date.now();
  // A change during a run queues one more; wait until the queue has drained.
  await waitForRun(page, before + 1);
  await page.waitForFunction("/^(Done|Run failed)/.test(document.getElementById('runStatus').textContent)", { timeout: 300000 });
  return Date.now() - t0;
}

/** Scrub the shown-refinement slider to the first kept refinement and back, reading the label and PSNR at each end. */
async function scrubRefinements(page) {
  return page.evaluate(() => {
    const row = document.getElementById("shownRow"), el = document.getElementById("shown");
    if (row.hidden) return { kept: 1 };
    const read = () => ({ label: document.getElementById("shownValue").textContent, psnr: document.getElementById("psnr").textContent });
    const kept = Number(el.max) + 1;
    el.value = "0"; el.dispatchEvent(new Event("input"));
    const first = read();
    el.value = el.max; el.dispatchEvent(new Event("input"));
    return { kept, first, last: read() };
  });
}

async function readStats(page) {
  return page.evaluate(() => ({
    sample: document.getElementById("imageName").textContent,
    status: document.getElementById("runStatus").textContent,
    gaussians: Number(document.getElementById("statN").textContent.replace(/,/g, "")),
    psnr: document.getElementById("psnr").textContent,
    adapter: document.getElementById("gpuPill").title,
  }));
}

async function main() {
  const { browser, close } = await launchChromium({ ignoreCertErrors: !!args.url, protocolTimeout: 600000 });
  const local = args.url ? null : await serve({ root: dist });
  const url = (args.url ?? `http://127.0.0.1:${local.port}/index.html`) + (args.sample ? `?sample=${args.sample}` : "");
  let result = { ok: false, url };
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 900 });
    if (args.theme) await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: args.theme }]);
    page.setDefaultTimeout(600000);
    page.on("console", (m) => { const t = m.text(); if (!/favicon|onnxruntime:/.test(t)) out("  [page] " + t.slice(0, 300)); });
    page.on("pageerror", (e) => out("  [pageerror] " + String(e).slice(0, 300)));
    await page.goto(url);
    await waitForRun(page, 1);
    if (args.shot) { await pause(800); await page.screenshot({ path: args.shot + "-rest.png" }); }
    const autoCount = Number((await text(page, "statN")).replace(/,/g, ""));
    const changed = args.states || args.scale;
    const wallMs = changed ? await rerunWith(page, args) : null;
    if (args.shot) { await pause(500); await page.screenshot({ path: args.shot + "-done.png" }); }
    const scrub = await scrubRefinements(page);
    const info = await readStats(page);
    // A scaled run places trunc(sum(rate) * scale); the slider rounds the scale to 2^(k/50), so allow 2%.
    const expected = args.scale ? Math.trunc(autoCount * Number(args.scale)) : 1000;
    const countOk = args.scale ? Math.abs(info.gaussians - expected) <= Math.max(2, 0.02 * expected) : info.gaussians > expected;
    const wantStates = args.states ? Number(args.states) : 6;
    const statesOk = scrub.kept === Math.max(1, wantStates);
    result = { ok: info.status.startsWith("Done") && countOk && statesOk, url, wall_ms: wallMs, requested: args.scale ? `x${args.scale} of ${autoCount} = ${expected}` : "auto", states: wantStates, scrub, ...info };
  } catch (e) {
    result = { ok: false, url, error: String(e).slice(0, 800) };
  } finally {
    await close();
    local?.server.close();
  }
  out(JSON.stringify(result, null, 1));
  process.exit(result.ok ? 0 : 1);
}

main().catch((e) => { out(String(e)); process.exit(1); });

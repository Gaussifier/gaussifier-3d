#!/usr/bin/env node
// Drive the built 3D demo: load the page, wait for the automatic run of a Kodak sample, read the
// stats, screenshot the frontal and two orbited views, and encode the SOG once.
//
//   node tools/planar-demo-check.mjs [--sample kodim07] [--model fast|quality] [--shot prefix] [--root dir | --url https://...]
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium } from "./chromium.mjs";
import { out, parseArgs } from "./cli.mjs";
import { serve } from "./server.mjs";

const args = parseArgs(process.argv.slice(2), { sample: "kodim07", model: "fast", shot: null, root: null, url: null });
// --root serves another copy of the built site (e.g. the published repository's checkout).
const dist = args.root ? path.resolve(args.root) : path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../apps/planar-demo/dist");
const pause = (ms) => new Promise((r) => setTimeout(r, ms));

const { browser, close } = await launchChromium({ windowSize: "1440,900" });
// --url checks a deployed copy (e.g. the published page) instead of serving one.
const { server, port } = args.url ? { server: { close() {} }, port: 0 } : await serve({ root: dist, isolate: true });
let failed = 0;
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  page.on("console", (m) => { const t = m.text(); if (!/favicon|DevTools/.test(t)) out("  [page] " + t.slice(0, 300)); });
  page.on("pageerror", (e) => { failed++; out("  [pageerror] " + String(e).slice(0, 500)); });
  page.on("response", (r) => { if (r.status() >= 400) out(`  [http ${r.status()}] ${r.url()}`); });
  await page.goto(args.url ?? `http://127.0.0.1:${port}/`);
  const waitDone = () => page.waitForFunction(() => /^Done|failed/.test(document.getElementById("runStatus").textContent) && !window.__planarDemo?.running, { timeout: 300000 });
  await waitDone();
  // The page picks a random sample; pin the requested one and the model.
  if (args.model === "quality") await page.click("#modelQuality");
  // Three runs of the pinned sample (alternating with another image so each click starts a run).
  const totals = [];
  for (const name of [args.sample, "kodim01", args.sample, "kodim01", args.sample]) {
    await page.click(`#samples button[title="${name}"]`);
    await pause(300);
    await waitDone();
    totals.push(await page.evaluate(() => window.__planarDemo.result?.run.timings.total));
  }
  out(JSON.stringify({ run_totals_ms: totals }));
  const status = await page.$eval("#runStatus", (e) => e.textContent);
  const n = await page.$eval("#statN", (e) => e.textContent);
  const psnr = await page.$eval("#psnr", (e) => e.textContent);
  const timings = await page.evaluate(() => window.__planarDemo.result?.run.timings);
  out(JSON.stringify({ sample: args.sample, model: args.model, status, n, psnr, timings }));
  if (!/^Done/.test(status)) failed++;
  if (args.shot) {
    await pause(500);
    await page.screenshot({ path: `${args.shot}_frontal.png` });
    const canvas = await page.$("#view");
    const box = await canvas.boundingBox();
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    for (const [name, dx, dy] of [["orbit_left", 140, 30], ["orbit_steep", 0, -150]]) {
      await page.mouse.move(cx, cy);
      await page.mouse.down();
      for (let i = 1; i <= 10; i++) await page.mouse.move(cx + (dx * i) / 10, cy + (dy * i) / 10);
      await page.mouse.up();
      await pause(400);
      await page.screenshot({ path: `${args.shot}_${name}.png` });
    }
  }
  const t0 = Date.now();
  const sog = await page.evaluate(() => window.__planarDemo.sogBytes());
  out(JSON.stringify({ sog_bytes: sog, sog_ms: Date.now() - t0 }));
  if (!(sog > 1000)) failed++;
} finally {
  await close();
  server.close();
}
process.exit(failed ? 1 : 0);

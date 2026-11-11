#!/usr/bin/env node
// Exercise the 3D demo's controls on the built page: Refine, Count, the refinement scrubber, the
// overlays, Centers, the SOG export, and reopening the exported SOG as a saved scene.
//
//   node tools/planar-demo-ui-check.mjs --shot <prefix> [--sample kodim07] [--root dir | --url https://...]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium } from "./chromium.mjs";
import { out, parseArgs } from "./cli.mjs";
import { serve } from "./server.mjs";

const args = parseArgs(process.argv.slice(2), { sample: "kodim07", shot: null, root: null, url: null });
const dist = args.root ? path.resolve(args.root) : path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../apps/planar-demo/dist");
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const shot = args.shot ?? path.join(process.cwd(), "planar-ui");
const downloads = path.join(path.dirname(shot), "downloads");
fs.mkdirSync(downloads, { recursive: true });

const { browser, close } = await launchChromium({ windowSize: "1440,900" });
const { server, port } = args.url ? { server: { close() {} }, port: 0 } : await serve({ root: dist, isolate: true });
let failed = 0;
const check = (name, ok, detail) => { if (!ok) failed++; out(`${ok ? "PASS" : "FAIL"} ${name} ${JSON.stringify(detail)}`); };
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  page.on("pageerror", (e) => { failed++; out("  [pageerror] " + String(e).slice(0, 300)); });
  const cdp = await page.createCDPSession();
  await cdp.send("Browser.setDownloadBehavior", { behavior: "allow", downloadPath: downloads });
  const runs = () => page.evaluate(() => Number(document.getElementById("runStatus").dataset.runs ?? 0));
  const waitRuns = (n) => page.waitForFunction((n) => Number(document.getElementById("runStatus").dataset.runs ?? 0) >= n && !window.__planarDemo.running, { timeout: 300000 }, n);
  const text = (id) => page.$eval(`#${id}`, (e) => e.textContent);
  const setRange = async (id, value) => page.evaluate((id, v) => { const el = document.getElementById(id); el.value = String(v); el.dispatchEvent(new Event("input")); el.dispatchEvent(new Event("change")); }, id, value);
  await page.goto(args.url ?? `http://127.0.0.1:${port}/?sample=${args.sample}`);
  await waitRuns(1);
  check("default run", /^Done on /.test(await text("runStatus")), { status: await text("runStatus"), n: await text("statN"), psnr: await text("psnr"), shown: await text("shownValue"), states: await page.$eval("#shown", (e) => Number(e.max) + 1) });
  await page.screenshot({ path: `${shot}_default.png` });

  // Refine: 3 refinements -> 3 kept states (the initializer is dropped).
  let n0 = await runs();
  await setRange("refine", 3);
  await waitRuns(n0 + 1);
  const r3 = await page.evaluate(() => { const r = window.__planarDemo.result?.run; return r ? { states: r.states.length, psnrs: r.psnrPerIteration.map((v) => +v.toFixed(3)), size: [r.width, r.height] } : null; });
  check("refine 3", (await page.$eval("#shown", (e) => Number(e.max) + 1)) === 3 && (await text("refineValue")) === "3" && r3?.states === 4, { shownMax: await page.$eval("#shown", (e) => e.max), psnr: await text("psnr"), run: r3 });

  // Count: x2 (slider 50).
  n0 = await runs();
  await setRange("count", 50);
  await waitRuns(n0 + 1);
  const n2 = Number((await text("statN")).replace(/,/g, ""));
  check("count x2", (await text("countValue")) === "×2.00", { countValue: await text("countValue"), n: n2, title: await page.$eval("#count", (e) => e.title) });
  n0 = await runs();
  await setRange("count", 0);
  await setRange("refine", 6);
  await waitRuns(n0 + 1);
  await pause(300);
  if (await page.evaluate(() => window.__planarDemo.running)) await waitRuns(n0 + 2);

  // Scrubber: '[' steps back one refinement; PSNR follows the shown state.
  const before = await text("psnr");
  await page.keyboard.press("[");
  await pause(800);
  check("scrub back", (await text("shownValue")) === "5", { shown: await text("shownValue"), psnrBefore: before, psnrAfter: await text("psnr") });
  await page.keyboard.press("]");
  await pause(800);

  // Overlays and Centers.
  for (const id of ["ov-image", "ov-diff", "ov-density"]) {
    await page.click(`label:has(#${id})`);
    await pause(400);
    await page.screenshot({ path: `${shot}_${id}.png` });
    check(id, !(await page.$eval("#overlay", (e) => e.hidden)), {});
    await page.click(`label:has(#${id})`);
    await pause(200);
  }
  await page.click("label:has(#ov-centers)");
  await pause(1500);
  await page.screenshot({ path: `${shot}_centers.png` });
  await page.click("label:has(#ov-centers)");
  await pause(800);

  // Export the SOG, then open it as a saved scene: the viewer applies the 3DGS viewers' load turn.
  for (const f of fs.readdirSync(downloads)) fs.unlinkSync(path.join(downloads, f));
  await page.click("#exportSog");
  let sog = null;
  for (let i = 0; i < 60 && !sog; i++) { await pause(500); sog = fs.readdirSync(downloads).find((f) => f.endsWith(".sog")); }
  check("export sog", !!sog, { file: sog, status: await text("status") });
  if (sog) {
    await page.click("summary");
    const input = await page.$("#file");
    await input.uploadFile(path.join(downloads, sog));
    await page.waitForFunction(() => /^Showing /.test(document.getElementById("runStatus").textContent), { timeout: 60000 });
    await pause(1500);
    await page.screenshot({ path: `${shot}_reopened_sog.png` });
    const canvas = await page.$("#view");
    const box = await canvas.boundingBox();
    const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await page.mouse.move(cx, cy); await page.mouse.down();
    for (let i = 1; i <= 10; i++) await page.mouse.move(cx + 14 * i, cy + 3 * i);
    await page.mouse.up();
    await pause(500);
    await page.screenshot({ path: `${shot}_reopened_sog_orbit.png` });
    check("reopen sog", true, { status: await text("runStatus") });
  }
} finally {
  await close();
  server.close();
}
process.exit(failed ? 1 : 0);

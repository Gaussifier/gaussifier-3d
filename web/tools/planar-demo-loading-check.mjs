#!/usr/bin/env node
// Watch the 3D demo's model-loading bar on a throttled connection: sample the bar and the status
// text while the model loads, and screenshot it halfway.
//
//   node tools/planar-demo-loading-check.mjs [--shot prefix] [--mbps 80] [--root dir | --url https://...]
import path from "node:path";
import { fileURLToPath } from "node:url";
import { launchChromium } from "./chromium.mjs";
import { out, parseArgs } from "./cli.mjs";
import { serve } from "./server.mjs";

const args = parseArgs(process.argv.slice(2), { shot: null, mbps: "80", root: null, url: null });
const dist = args.root ? path.resolve(args.root) : path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../apps/planar-demo/dist");
const { browser, close } = await launchChromium({ windowSize: "1440,900" });
const { server, port } = args.url ? { server: { close() {} }, port: 0 } : await serve({ root: dist, isolate: true });
let failed = 0;
try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });
  const cdp = await page.createCDPSession();
  await cdp.send("Network.enable");
  await cdp.send("Network.setCacheDisabled", { cacheDisabled: true });
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 20, downloadThroughput: (Number(args.mbps) * 1e6) / 8, uploadThroughput: 1e6 });
  const t0 = Date.now();
  await page.goto(args.url ?? `http://127.0.0.1:${port}/?sample=kodim07`, { waitUntil: "domcontentloaded" });
  const samples = [];
  let shot = false;
  for (let i = 0; i < 600; i++) {
    const s = await page.evaluate(() => {
      const bar = document.getElementById("progress");
      return { hidden: bar.hidden, indeterminate: bar.classList.contains("indeterminate"), width: bar.firstElementChild.style.width, status: document.getElementById("runStatus").textContent };
    });
    const last = samples[samples.length - 1];
    if (!last || last.status !== s.status || last.hidden !== s.hidden) samples.push({ t: ((Date.now() - t0) / 1000).toFixed(1), ...s });
    if (!shot && args.shot && /Loading the model… [4-6]\d%/.test(s.status)) { await page.screenshot({ path: `${args.shot}_loading.png` }); shot = true; }
    if (/^Done|failed/.test(s.status)) break;
    await new Promise((r) => setTimeout(r, 100));
  }
  const shown = samples.filter((s) => /Loading the model… \d+%/.test(s.status));
  for (const s of samples.filter((_, i) => i % Math.max(1, Math.floor(samples.length / 12)) === 0 || i === samples.length - 1)) out(JSON.stringify(s));
  const ok = shown.length >= 3 && samples.some((s) => s.indeterminate) && /^Done/.test(samples[samples.length - 1].status) && samples[samples.length - 1].hidden;
  if (!ok) failed++;
  out(`${ok ? "PASS" : "FAIL"} loading bar: ${shown.length} percentage updates, prepare phase ${samples.some((s) => s.indeterminate)}, ends hidden after ${samples[samples.length - 1].t} s`);
} finally {
  await close();
  server.close();
}
process.exit(failed ? 1 : 0);

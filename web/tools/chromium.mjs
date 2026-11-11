// Shared Chromium launcher for the browser tools: finds a binary, starts Xvfb when
// there is no display, and passes the flags that expose the real GPU through WebGPU.
import { execSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function findChrome() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const cache = path.join(os.homedir(), ".cache", "ms-playwright");
  if (fs.existsSync(cache)) {
    for (const d of fs.readdirSync(cache).filter((d) => d.startsWith("chromium-")).sort().reverse()) {
      for (const sub of ["chrome-linux64/chrome", "chrome-linux/chrome"]) {
        const p = path.join(cache, d, sub);
        if (fs.existsSync(p)) return p;
      }
    }
  }
  for (const c of ["chromium", "google-chrome", "chromium-browser"]) {
    try { return execSync(`which ${c}`, { stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); } catch { /* next */ }
  }
  throw new Error("no Chromium found; set CHROME_PATH");
}

/**
 * Launch Chromium with WebGPU enabled. Options: software (SwiftShader adapter),
 * ignoreCertErrors, windowSize, protocolTimeout. Returns { browser, close }.
 */
export async function launchChromium({ software = process.env.WEBGPU_SOFTWARE === "1", ignoreCertErrors = false, windowSize = "1440,900", protocolTimeout = 900000 } = {}) {
  const env = { ...process.env };
  let xvfb = null;
  if (!software && !env.DISPLAY && process.platform === "linux") {
    const display = ":" + (90 + Math.floor(Math.random() * 100));
    try {
      xvfb = spawn("Xvfb", [display, "-screen", "0", "1600x1200x24"], { stdio: "ignore" });
      env.DISPLAY = display;
      await new Promise((r) => setTimeout(r, 1500));
    } catch { xvfb = null; }
  }
  const args = ["--no-sandbox", "--disable-gpu-sandbox", "--enable-unsafe-webgpu", "--ignore-gpu-blocklist", `--window-size=${windowSize}`];
  if (software) args.push("--use-webgpu-adapter=swiftshader");
  else if (process.platform === "linux") args.push("--enable-features=Vulkan", "--use-vulkan=native");
  if (ignoreCertErrors) args.push("--ignore-certificate-errors");
  const puppeteer = (await import("puppeteer-core")).default;
  const browser = await puppeteer.launch({ executablePath: findChrome(), headless: software ? true : !env.DISPLAY, env, protocolTimeout, args });
  return { browser, close: async () => { await browser.close().catch(() => {}); if (xvfb) xvfb.kill(); } };
}

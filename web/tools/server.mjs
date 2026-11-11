#!/usr/bin/env node
// Static file server for tests, the demo, and LAN serving over HTTPS.
//
//   node tools/server.mjs --dir apps/demo/dist --http 8080 --https 8443 --host 0.0.0.0
//
// WebGPU needs a secure context, so remote browsers must use the HTTPS port and accept
// the self-signed certificate once; the certificate names this host's addresses.
import http from "node:http";
import https from "node:https";
import fs from "node:fs";
import { parseArgs } from "./cli.mjs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json",
  ".onnx": "application/octet-stream", ".bin": "application/octet-stream", ".data": "application/octet-stream", ".splat2d": "application/octet-stream",
  ".npz": "application/octet-stream", ".npy": "application/octet-stream", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml", ".map": "application/json", ".ico": "image/x-icon" };

export function lanAddresses() {
  const out = [];
  for (const [name, list] of Object.entries(os.networkInterfaces())) for (const a of list ?? []) if (a.family === "IPv4" && !a.internal) out.push({ name, address: a.address });
  return out;
}

export function createHandler(root, { isolate = false, log = false } = {}) {
  root = path.resolve(root);
  return (req, res) => {
    const started = Date.now();
    let url = decodeURIComponent(req.url.split("?")[0]);
    if (url.endsWith("/")) url += "index.html";
    const p = path.normalize(path.join(root, url));
    if (!p.startsWith(root)) { res.writeHead(403); res.end(); return; }
    fs.stat(p, (err, st) => {
      if (err || !st.isFile()) { res.writeHead(404); res.end("not found"); return; }
      const ext = path.extname(p);
      const headers = { "Content-Type": MIME[ext] ?? "application/octet-stream", "Content-Length": st.size, "Cache-Control": ext === ".html" ? "no-store" : "public, max-age=3600" };
      if (isolate) { headers["Cross-Origin-Opener-Policy"] = "same-origin"; headers["Cross-Origin-Embedder-Policy"] = "require-corp"; }
      res.writeHead(200, headers);
      fs.createReadStream(p).pipe(res);
      if (log) res.on("finish", () => console.log(`${new Date().toISOString()} ${req.socket.remoteAddress} ${req.method} ${url} ${st.size}B ${Date.now() - started}ms`));
    });
  };
}

/** Self-signed certificate for LAN testing, cached under tools/.certs. */
export function ensureCert(dir = path.join(path.dirname(fileURLToPath(import.meta.url)), ".certs")) {
  const key = path.join(dir, "key.pem"), cert = path.join(dir, "cert.pem");
  if (!fs.existsSync(key) || !fs.existsSync(cert)) {
    fs.mkdirSync(dir, { recursive: true });
    const sans = ["DNS:localhost", "IP:127.0.0.1", ...lanAddresses().map((a) => `IP:${a.address}`), `DNS:${os.hostname()}`].join(",");
    execFileSync("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert, "-days", "365", "-subj", "/CN=gaussifier-demo", "-addext", `subjectAltName=${sans}`], { stdio: "ignore" });
  }
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) };
}

/** Start one server. `tls` = true uses the cached self-signed certificate. Resolves { server, port }. */
export function serve({ root, port = 0, host = "127.0.0.1", isolate = false, log = false, tls = false } = {}) {
  const handler = createHandler(root, { isolate, log });
  const server = tls ? https.createServer(ensureCert(), handler) : http.createServer(handler);
  return new Promise((resolve) => server.listen(port, host, () => resolve({ server, port: server.address().port })));
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) {
  const args = parseArgs(process.argv.slice(2), { dir: "apps/demo/dist", host: "0.0.0.0", http: "8080", https: "0" });
  const root = path.resolve(args.dir);
  const host = args.host;
  const httpPort = Number(args.http), httpsPort = Number(args.https);
  if (!fs.existsSync(path.join(root, "index.html"))) { console.error(`no index.html in ${root}; run: npm run build:demo`); process.exit(2); }
  const started = [];
  if (httpPort) started.push(serve({ root, port: httpPort, host, log: true }).then(({ port }) => console.log(`HTTP  on ${host}:${port}`)));
  if (httpsPort) started.push(serve({ root, port: httpsPort, host, log: true, tls: true }).then(({ port }) => console.log(`HTTPS on ${host}:${port}`)));
  await Promise.all(started);
  console.log(`serving ${root}`);
  if (host === "0.0.0.0") for (const a of lanAddresses()) console.log(`  ${httpsPort ? `https://${a.address}:${httpsPort}/` : `http://${a.address}:${httpPort}/`}   (${a.name})`);
  console.log(`  ${httpsPort ? `https://localhost:${httpsPort}/` : `http://localhost:${httpPort}/`}`);
}

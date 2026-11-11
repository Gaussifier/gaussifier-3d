// Shared bits of the command-line tools: a small argument parser and synchronous stdout.
import fs from "node:fs";

/**
 * Parse `--name value` options and `--flag` switches from argv. `spec` maps option names to
 * defaults: a boolean default makes a switch, anything else a valued option. Returns the options
 * plus the remaining positional arguments under `_`.
 */
export function parseArgs(argv, spec) {
  const out = { _: [] };
  for (const [name, def] of Object.entries(spec)) out[name] = def;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) { out._.push(a); continue; }
    const name = a.slice(2);
    if (!(name in spec)) throw new Error(`unknown option ${a}`);
    if (typeof spec[name] === "boolean") out[name] = true;
    else out[name] = argv[++i];
  }
  return out;
}

/** Synchronous stdout so a timeout cannot swallow buffered output. */
export const out = (...parts) => fs.writeSync(1, parts.join(" ") + "\n");

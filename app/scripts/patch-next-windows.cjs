/*
 * Next 15.0.x bundles @vercel/nft with a symlink resolver that does not
 * recognise Node 24's Windows EISDIR result from readlinkSync on a regular
 * file.  Treat EISDIR like the existing EINVAL/ENOENT cases.  The patch is
 * idempotent and only touches the installed dependency; it is needed for
 * Windows builds and has no effect on application runtime code.
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..", "node_modules", "next");
const targets = [
  path.join(root, "dist", "compiled", "@vercel", "nft", "index.js"),
  path.join(root, "dist", "build", "webpack", "plugins", "next-trace-entrypoints-plugin.js"),
  path.join(root, "dist", "build", "collect-build-traces.js"),
  path.join(root, "dist", "esm", "build", "collect-build-traces.js"),
];

const replacements = [
  [
    'e.code!=="EINVAL"&&e.code!=="ENOENT"&&e.code!=="UNKNOWN"',
    'e.code!=="EINVAL"&&e.code!=="ENOENT"&&e.code!=="EISDIR"&&e.code!=="UNKNOWN"',
  ],
  [
    "e.code === 'EINVAL' || e.code === 'ENOENT' || e.code === 'UNKNOWN'",
    "e.code === 'EINVAL' || e.code === 'ENOENT' || e.code === 'EISDIR' || e.code === 'UNKNOWN'",
  ],
];

let changed = false;
for (const target of targets) {
  if (!fs.existsSync(target)) continue;
  const source = fs.readFileSync(target, "utf8");
  for (const [needle, replacement] of replacements) {
    if (source.includes(needle)) {
      fs.writeFileSync(target, source.replace(needle, replacement), "utf8");
      changed = true;
      break;
    }
  }
}
if (changed) console.log("Patched Next tracing for Node 24 Windows readlink behavior.");

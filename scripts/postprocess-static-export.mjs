import { readdir, readFile, stat, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";

const root = process.argv[2] ?? "out";
const base = process.env.GITHUB_PAGES_BASE_PATH ?? "/embodied-research-notes";
const textExtensions = new Set([".html", ".js", ".css", ".json", ".txt", ".xml"]);

async function visit(directory) {
  for (const name of await readdir(directory)) {
    const path = join(directory, name);
    const info = await stat(path);
    if (info.isDirectory()) {
      await visit(path);
      continue;
    }
    if (!textExtensions.has(extname(path))) continue;

    const original = await readFile(path, "utf8");
    const updated = original
      .replaceAll(new RegExp(`(?<!${base})/notes/`, "g"), `${base}/notes/`)
      .replaceAll(new RegExp(`(?<!${base})/downloads/`, "g"), `${base}/downloads/`)
      .replaceAll(new RegExp(`(?<!${base})/favicon\\.svg`, "g"), `${base}/favicon.svg`);
    if (updated !== original) await writeFile(path, updated);
  }
}

await visit(root);

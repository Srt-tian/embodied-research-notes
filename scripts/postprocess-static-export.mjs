import { readdir, readFile, stat } from "node:fs/promises";
import { extname, join } from "node:path";

const root = process.argv[2] ?? "out";
const base = process.env.GITHUB_PAGES_BASE_PATH ?? "/embodied-research-notes";
const textExtensions = new Set([".html", ".js", ".css", ".json", ".txt", ".xml"]);
const clientNavigationPrefixes = ["/notes", "/coding", "/coding-skill"];

async function visit(directory) {
  for (const name of await readdir(directory)) {
    const path = join(directory, name);
    const info = await stat(path);
    if (info.isDirectory()) {
      await visit(path);
      continue;
    }
    if (!textExtensions.has(extname(path))) continue;

    const content = await readFile(path, "utf8");
    const duplicatedBasePath = `${base}${base}/`;
    if (content.includes(duplicatedBasePath)) {
      throw new Error(`Duplicated GitHub Pages base path in ${path}`);
    }
    for (const prefix of clientNavigationPrefixes) {
      const prefixedClientHref = `\\"href\\":\\"${base}${prefix}`;
      if (content.includes(prefixedClientHref)) {
        throw new Error(`Client navigation already contains the base path in ${path}`);
      }
    }
  }
}

await visit(root);

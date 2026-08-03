# GitHub Pages publication

The repository's `main` branch is the source of truth for the notes site.

Every push to `main` automatically builds and deploys the static export to:

https://srt-tian.github.io/embodied-research-notes/

Pull requests run a read-only static-export validation before merge. The export is enabled only when `GITHUB_PAGES=true`.

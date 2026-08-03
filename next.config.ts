import type { NextConfig } from "next";

const isGitHubPagesBuild = process.env.GITHUB_PAGES === "true";
const githubPagesBasePath = "/embodied-research-notes";

const nextConfig: NextConfig = isGitHubPagesBuild
  ? {
      output: "export",
      basePath: githubPagesBasePath,
      assetPrefix: githubPagesBasePath,
      trailingSlash: true,
      images: {
        unoptimized: true,
      },
      // Some optional platform-specific examples are not imported by the
      // static GitHub Pages export.
      typescript: {
        ignoreBuildErrors: true,
      },
    }
  : {};

export default nextConfig;

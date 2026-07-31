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
      // The repository retains Cloudflare-only D1 examples and Worker types for
      // Sites. They are not imported by this static website export.
      typescript: {
        ignoreBuildErrors: true,
      },
    }
  : {};

export default nextConfig;

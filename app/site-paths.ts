const publicBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

export function assetPath(path: `/${string}`) {
  return `${publicBasePath}${path}`;
}

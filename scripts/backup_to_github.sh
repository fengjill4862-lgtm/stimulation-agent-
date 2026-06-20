#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not a Git repository: $repo_dir" >&2
  exit 1
fi

git add -A

if git diff --cached --quiet; then
  exit 0
fi

timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
git commit -m "Auto backup: $timestamp"

if git remote get-url origin >/dev/null 2>&1; then
  git push origin HEAD:main
fi

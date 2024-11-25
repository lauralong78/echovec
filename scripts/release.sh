#!/usr/bin/env bash
# Cut a release: bump must already be committed, then tag and push.
# Usage: scripts/release.sh v0.1.0
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 vX.Y.Z" >&2
  exit 1
fi

VERSION="$1"
PEP440="${VERSION#v}"

if ! grep -q "version = \"${PEP440}\"" pyproject.toml; then
  echo "error: pyproject.toml version does not match ${PEP440}" >&2
  exit 1
fi

echo "Tagging ${VERSION}..."
git tag -a "${VERSION}" -m "Release ${VERSION}"
git push origin "${VERSION}"
echo "Pushed tag ${VERSION}; the release workflow will build and attach artifacts."

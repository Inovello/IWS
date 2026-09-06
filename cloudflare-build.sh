#!/usr/bin/env bash
# Cloudflare build command. Downloads the pinned Hugo release so the build never
# depends on whatever Hugo version the build image happens to ship.
set -euo pipefail
HUGO_VERSION="${HUGO_VERSION:-0.165.0}"
curl -sSL "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz" | tar -xz hugo
./hugo version
./hugo --minify --gc

#!/usr/bin/env bash
# This is an installation script that will
# install qarton in the current environment

set -euo pipefail

REPO_URL="https://gitlab.inria.fr/capsule/qarton.git"
TMP_DIR="$(mktemp -d)"

echo "Cloning qarton into $TMP_DIR..."
git clone --depth 1 "$REPO_URL" "$TMP_DIR"

echo "Installing with pip..."
pip install "$TMP_DIR"

echo "Done."

# Optional cleanup
rm -rf "$TMP_DIR"

# --------- Additional imports ----------
pip install pytest

echo


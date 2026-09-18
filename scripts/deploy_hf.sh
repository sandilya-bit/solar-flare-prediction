#!/usr/bin/env bash
# Publish this project to a Hugging Face Space (Docker SDK).
#
#   HF_TOKEN=hf_xxxxxxxx scripts/deploy_hf.sh <user>/<space-name>
#
# The script never touches your working tree: it exports the committed files to a
# temporary directory, swaps in the Space front-matter from docs/hf/README.md
# (a Space cannot run without it) and pushes to the Space repository.
#
# Note: the push replaces the Space's initial scaffold commit. At that point the
# Space contains nothing but auto-generated files, so nothing of value is lost —
# which is why --force is used. Re-run this script after any change to redeploy.

set -euo pipefail

SPACE="${1:-}"
if [ -z "$SPACE" ]; then
    echo "usage: HF_TOKEN=hf_xxx $0 <user>/<space-name>" >&2
    exit 2
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: set HF_TOKEN to a Hugging Face write token (https://huggingface.co/settings/tokens)" >&2
    exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f docs/hf/README.md ]; then
    echo "error: docs/hf/README.md (Space front-matter) is missing" >&2
    exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "HEAD" ]; then
    echo "error: detached HEAD — check out a branch before deploying" >&2
    exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> Exporting ${BRANCH} to $STAGE"
git archive "$BRANCH" | tar -x -C "$STAGE"

echo "==> Writing Space front-matter"
cp docs/hf/README.md "$STAGE/README.md"

# Images of the repo are not useful inside the Space.
rm -rf "$STAGE/docs/screenshots" "$STAGE/tests" "$STAGE/.devcontainer"

cd "$STAGE"
git init -q -b main
git add -A
git -c user.name="deploy_hf.sh" -c user.email="deploy@localhost" \
    commit -q -m "Deploy solar flare dashboard from ${BRANCH}"

echo "==> Pushing to huggingface.co/spaces/${SPACE}"
git remote add space "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}.git"
git push --force space main:main

echo
echo "Done. First build takes a few minutes."
echo "Live URL: https://$(echo "$SPACE" | tr '/_' '--' | tr '[:upper:]' '[:lower:]').hf.space"
echo "Space page: https://huggingface.co/spaces/${SPACE}"

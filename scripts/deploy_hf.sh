#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN to a Hugging Face write token before deploying."
  exit 1
fi

SPACE_ID="${HF_SPACE:-lyyn002/plant-disease-live-demo}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

pip install -q huggingface_hub
huggingface-cli login --token "$HF_TOKEN"

if ! huggingface-cli repo info "$SPACE_ID" --repo-type space >/dev/null 2>&1; then
  huggingface-cli repo create plant-disease-live-demo \
    --type space \
    --space-sdk docker \
    --organization lyyn002
fi

cd "$REPO_DIR"
git remote remove space 2>/dev/null || true
git remote add space "https://huggingface.co/spaces/${SPACE_ID}"
git push space main --force

echo "Deployed to https://huggingface.co/spaces/${SPACE_ID}"

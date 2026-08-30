#!/usr/bin/env bash
# Deploy this project to Hugging Face Spaces (Docker SDK) and wait for a healthy runtime.
set -euo pipefail

SPACE_ID="${HF_SPACE:-lyyn002/plant-disease-live-demo}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_URL="https://huggingface.co/spaces/${SPACE_ID}"
APP_URL="https://${SPACE_ID//\//-}.hf.space"

if [[ -z "${HF_TOKEN:-}" ]]; then
  if huggingface-cli whoami >/dev/null 2>&1; then
    echo "Using cached Hugging Face credentials."
  else
    echo "Not logged in. Run: huggingface-cli login"
    echo "Or set HF_TOKEN from https://huggingface.co/settings/tokens (write access)."
    exit 1
  fi
else
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

USERNAME="$(huggingface-cli whoami | head -1 | awk '{print $1}')"
echo "Authenticated as: ${USERNAME}"

if ! huggingface-cli repo info "$SPACE_ID" --repo-type space >/dev/null 2>&1; then
  echo "Creating Space ${SPACE_ID}..."
  huggingface-cli repo create plant-disease-live-demo \
    --type space \
    --space_sdk docker \
    -y
fi

cd "$REPO_DIR"
git remote remove space 2>/dev/null || true
git remote add space "https://huggingface.co/spaces/${SPACE_ID}"

echo "Pushing to ${SPACE_URL}..."
git push space main --force

echo "Waiting for Space build (this can take several minutes)..."
python3 - "${SPACE_ID}" <<'PY'
import sys
import time

from huggingface_hub import HfApi, get_space_runtime

space_id = sys.argv[1]
api = HfApi()
deadline = time.time() + 900  # 15 minutes

while time.time() < deadline:
    try:
        runtime = get_space_runtime(space_id)
        stage = runtime.stage
        hardware = runtime.hardware or "cpu-basic"
        print(f"  stage={stage} hardware={hardware}")
        if stage == "RUNNING":
            print("Space is RUNNING.")
            sys.exit(0)
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "PAUSED"}:
            print(f"Space failed with stage={stage}")
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"  waiting... ({exc})")
    time.sleep(20)

print("Timed out waiting for Space to reach RUNNING.")
sys.exit(1)
PY

echo ""
echo "Space page: ${SPACE_URL}"
echo "App URL:    ${APP_URL}"
#!/usr/bin/env bash
# One-shot driver for reextract_response_position.py on a remote GPU.
# Assumes: an rclone remote 'gdrive' is already configured (rclone config done),
# and HF_TOKEN is exported. Copies caches Drive->local, runs, copies results back.
set -euo pipefail

# ── config (override via env) ────────────────────────────────────────────────
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive:MyDrive/mechanism_comparison_output}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/output}"
PUSH_ACTS="${PUSH_ACTS:-0}"        # set 1 to also upload the new 3-position .npy caches
export REPO_DIR OUTPUT_DIR

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: export HF_TOKEN=hf_... (Llama-3.1 is gated)"; exit 1
fi
command -v rclone >/dev/null || { echo "ERROR: rclone not found (curl https://rclone.org/install.sh | sudo bash)"; exit 1; }

echo "REPO_DIR=$REPO_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "GDRIVE_REMOTE=$GDRIVE_REMOTE"

# ── 1. python deps (torch assumed preinstalled with CUDA) ────────────────────
python -m pip install -q -U -r "$REPO_DIR/requirements.txt"

# ── 2. pull caches Drive -> local disk (fast random reads) ───────────────────
mkdir -p "$OUTPUT_DIR"
echo ">> copying caches from $GDRIVE_REMOTE -> $OUTPUT_DIR"
rclone copy "$GDRIVE_REMOTE" "$OUTPUT_DIR" --transfers 16 --checkers 16 --progress
[ -f "$OUTPUT_DIR/probe_state.pkl" ] || { echo "ERROR: probe_state.pkl missing in $OUTPUT_DIR — check GDRIVE_REMOTE"; exit 1; }

# ── 3. run ───────────────────────────────────────────────────────────────────
echo ">> running reextract_response_position.py"
python "$REPO_DIR/reextract_response_position.py"

# ── 4. push results back to Drive ────────────────────────────────────────────
echo ">> copying results back to $GDRIVE_REMOTE"
rclone copy "$OUTPUT_DIR" "$GDRIVE_REMOTE" \
  --include "probe_state_resp.pkl" --include "results_resp.json" \
  --include "experiment_A_resp.png" --transfers 8 --progress
if [ "$PUSH_ACTS" = "1" ]; then
  echo ">> PUSH_ACTS=1: also uploading the 3-position activation caches (large)"
  rclone copy "$OUTPUT_DIR" "$GDRIVE_REMOTE" \
    --include "acts_*_pre_*.npy" --include "acts_*_resp_*.npy" \
    --include "acts_*_respmean_*.npy" --transfers 16 --progress
fi
echo ">> done. New on Drive: probe_state_resp.pkl, results_resp.json, experiment_A_resp.png"

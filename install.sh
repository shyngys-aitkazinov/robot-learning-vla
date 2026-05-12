#!/usr/bin/env bash
# install.sh — set up the ETH Robot Learning sandbox.
#
# Uses `uv pip install` to provision lerobot into a uv-managed venv. This is
# the install path the team has verified works on macOS 14 / Apple Silicon.
# It does NOT write a pyproject.toml or uv.lock; if you want a reproducible
# declarative install, edit this script to call `uv sync` instead.
#
# Idempotent: safe to re-run.
#
# Override defaults via env vars:
#   PYTHON_VERSION=3.12   ./install.sh
#   LEROBOT_SPEC="lerobot[hardware,viz,feetech,dynamixel]"   ./install.sh
#   HF_TOKEN=hf_...       ./install.sh

set -euo pipefail
cd "$(dirname "$0")"

# --- Defaults ------------------------------------------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
# Default to the bare `lerobot` package — that's the install confirmed working
# on this macOS 14 / Apple Silicon setup. Add extras by overriding LEROBOT_SPEC.
LEROBOT_SPEC="${LEROBOT_SPEC:-lerobot}"

# --- 1. uv ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> uv $(uv --version)"

# --- 2. Virtualenv -------------------------------------------------------
if [[ ! -d .venv ]]; then
    echo ">> Creating .venv on Python ${PYTHON_VERSION}"
    uv venv --python "${PYTHON_VERSION}"
else
    echo ">> Reusing existing .venv at $(pwd)/.venv"
fi

# --- 3. Install lerobot --------------------------------------------------
echo ">> uv pip install ${LEROBOT_SPEC}"
uv pip install "${LEROBOT_SPEC}"

# --- 4. Hugging Face auth (optional) ------------------------------------
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo ">> Logging into Hugging Face..."
    uv run hf auth login --token "$HF_TOKEN" --add-to-git-credential
else
    echo ">> Skipping HF auth: HF_TOKEN env var not set."
    echo "   Log in later with:"
    echo "     uv run hf auth login --token <YOUR_TOKEN> --add-to-git-credential"
fi

# --- 5. Calibration scaffolding -----------------------------------------
CALIB_ROOT="$HOME/.cache/huggingface/lerobot/calibration"
mkdir -p "$CALIB_ROOT/teleoperators/so_leader" "$CALIB_ROOT/robots/so_follower"
echo ">> Ensured calibration dirs under $CALIB_ROOT"

# --- 6. Summary ---------------------------------------------------------
cat <<EOM

==========================================================================
 Setup complete.

 Activate the venv:
   source .venv/bin/activate

 Or run commands without activating:
   uv run lerobot-find-port
   uv run lerobot-find-cameras opencv

 Calibration files for this machine live at:
   $CALIB_ROOT/teleoperators/so_leader/my_awesome_leader_arm.json
   $CALIB_ROOT/robots/so_follower/my_awesome_follower_arm.json

 Re-run this script any time. It will reuse the existing .venv.
==========================================================================
EOM

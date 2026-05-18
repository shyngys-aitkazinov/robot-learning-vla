#!/usr/bin/env bash
# Create `.venv_openvla` at repo root and install integrations/openvla Python deps.
# You still must install PyTorch appropriate for CUDA/MPS/CPU BEFORE or AFTER this script.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv_openvla ]]; then
  uv venv .venv_openvla --python 3.12
fi

# shellcheck disable=SC1091
source .venv_openvla/bin/activate
pip install -U pip
pip install -r integrations/openvla/requirements-min.txt
echo "Bootstrap done. Install PyTorch for your platform if not already:"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"

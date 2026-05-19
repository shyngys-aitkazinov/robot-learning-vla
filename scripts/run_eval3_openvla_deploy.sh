#!/usr/bin/env bash
# Launch eval3_openvla_deploy.py with the OpenVLA training venv.
#
# Examples:
#   OPENVLA_SRC=external/openvla ./scripts/run_eval3_openvla_deploy.sh --dry_run=true ...
#   CHECKPOINT=outputs/train/eval3-openvla-lora-new66-50k/checkpoints/050000 \
#     ./scripts/run_eval3_openvla_deploy.sh --allow_live_motors=true --episode_time_s=20 ...

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv_openvla_train ]]; then
  # shellcheck disable=SC1091
  source .venv_openvla_train/bin/activate
elif [[ -d .venv_openvla ]]; then
  # shellcheck disable=SC1091
  source .venv_openvla/bin/activate
else
  echo "Missing .venv_openvla_train — see integrations/openvla/README.md" >&2
  exit 2
fi

export OPENVLA_SRC="${OPENVLA_SRC:-}"
export PYTHONPATH="${ROOT}/integrations/openvla:${ROOT}/scripts:${PYTHONPATH:-}"

exec python scripts/eval3_openvla_deploy.py "$@"

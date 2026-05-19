#!/usr/bin/env bash
# FlowerVLA deploy wrapper — calls scripts/eval3_flower_deploy.py with the
# .venv_flower interpreter (FlowerVLA needs torch==2.2.x baseline + hydra +
# pytorch-lightning — separate from the main .venv used by SmolVLA).
#
# Port + camera index baked in for Shyngys's rig (/dev/tty.usbmodem5B140317761, cam 0).
# Override per-run via env vars: FOLLOWER_TTY=... CAM_IDX=... ./scripts/run_eval3_flower_deploy.sh ...
#
# Setup (one-time, ~5 min):
#   uv venv .venv_flower --python=3.10
#   uv pip install --python .venv_flower/bin/python \
#     -r <(grep -v '^--find-links' external/flower_vla_calvin/requirements.txt) \
#     "numpy<2" "transformers>=4.40,<4.50" "tokenizers<0.21" \
#     lerobot feetech-servo-sdk huggingface_hub torchcodec pyarrow datasets einops_exts timm wandb
#
# Usage:
#   ./scripts/run_eval3_flower_deploy.sh --task='Place the coke on Taylor Swift'
#   ./scripts/run_eval3_flower_deploy.sh --task='Place the coke on Barack Obama' --fps=5 --episode_time_s=60
#   FOLLOWER_TTY=/dev/tty.usbserial-XXX ./scripts/run_eval3_flower_deploy.sh --task='...'
#   ./scripts/run_eval3_flower_deploy.sh --help

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Underlying script flags (--task etc.):"
  exec .venv_flower/bin/python scripts/eval3_flower_deploy.py --help 2>&1 | tail -40
fi

# Hard-require .venv_flower; SmolVLA's main .venv does NOT have hydra / pytorch-lightning
if [[ ! -x .venv_flower/bin/python ]]; then
  cat >&2 <<'EOF'
ERROR: .venv_flower/bin/python not found.

FlowerVLA needs its own venv (torch==2.2-2.10, hydra, pytorch-lightning).
Set it up once:

  uv venv .venv_flower --python=3.10
  uv pip install --python .venv_flower/bin/python \
    -r <(grep -v '^--find-links' external/flower_vla_calvin/requirements.txt) \
    "numpy<2" "transformers>=4.40,<4.50" "tokenizers<0.21" \
    lerobot feetech-servo-sdk huggingface_hub torchcodec pyarrow datasets \
    einops_exts timm wandb

Then re-run this script.
EOF
  exit 2
fi

if [[ ! -d external/flower_vla_calvin ]]; then
  cat >&2 <<'EOF'
ERROR: external/flower_vla_calvin/ missing. Clone it once:

  mkdir -p external && cd external && \
    git clone https://github.com/intuitive-robots/flower_vla_calvin.git

Then re-run this script.
EOF
  exit 2
fi

FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/tty.usbmodem5B140317761}"
# Defensive: if user supplied a port without a leading slash, fix it. This typo
# has bitten us multiple times and gets buried in 30 lines of lerobot stack trace.
case "$FOLLOWER_TTY" in
  /*) : ;;  # already absolute
  *)
    echo "WARN: --robot.port='$FOLLOWER_TTY' missing leading '/'; prepending it." >&2
    FOLLOWER_TTY="/$FOLLOWER_TTY"
    ;;
esac
CAM_IDX="${CAM_IDX:-0}"
CHECKPOINT="${EVAL3_FLOWER_CHECKPOINT:-RobotLearningVLA/eval3-flower-new66-50k}"

# Defaults mirror the user's tested command. Override any of them by passing
# the flag as a trailing argument — Python argparse last-wins.
COMMON_ARGS=(
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_TTY"
  --robot.id=my_awesome_follower_arm
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAM_IDX}, width: 640, height: 480, fps: 30}}"
  --checkpoint_path="$CHECKPOINT"
  --flower_src=external/flower_vla_calvin
  --device=auto
  --episode_time_s=60
  --fps=5
  --motion_gain=0.25
  --action_smoothing_alpha=0.35
  --max_action_delta_deg=4
  --allow_live_motors=true
)

echo ">> Eval 3 FlowerVLA deploy"
echo "   checkpoint     : $CHECKPOINT"
echo "   follower port  : $FOLLOWER_TTY"
echo "   camera index   : $CAM_IDX"
echo "   interpreter    : .venv_flower/bin/python"
echo ""

exec .venv_flower/bin/python scripts/eval3_flower_deploy.py \
  "${COMMON_ARGS[@]}" \
  "$@"

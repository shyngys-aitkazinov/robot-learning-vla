#!/usr/bin/env bash
# Eval 3 deploy battery — runs one of four checkpoints with the friend-recipe flags.
#
# Friend's recipe (2026-05-18) tests four checkpoints:
#   v8           : eval3-vla-v8-gripper-repair-smooth-50k    (3-way, latest smoothed)
#   v6_combined : eval3-vla-v6-smolvla-fresh-combined88-50k (3-way, combined corpus)
#   v6_new      : eval3-vla-v6-smolvla-fresh-new66-50k      (3-way, new-data only)
#   v7_d        : eval3-vla-v7-D-obama-only-10k             (Obama-only — use only with the Obama prompt)
#
# Port + camera index baked in for Shyngys's rig (/dev/tty.usbmodem5B140317761, cam 0).
# Override per-run via env vars: FOLLOWER_TTY=... CAM_IDX=... ./scripts/run_eval3_deploy_battery.sh v8
#
# Usage:
#   ./scripts/run_eval3_deploy_battery.sh v8
#   ./scripts/run_eval3_deploy_battery.sh v6_combined
#   ./scripts/run_eval3_deploy_battery.sh v6_new
#   ./scripts/run_eval3_deploy_battery.sh v7_d
#   ./scripts/run_eval3_deploy_battery.sh --help    (print this comment block)
#
# The script will prompt you for the task on stdin (e.g. "Place the coke on Taylor Swift"),
# or you can pass --task='Place the coke on X' as an extra argument.
#
# All other deploy_vla_deploy.py flags can also be appended after the checkpoint name.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/tty.usbmodem5B140317761}"
CAM_IDX="${CAM_IDX:-0}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

CHECKPOINT_NAME="$1"; shift

case "$CHECKPOINT_NAME" in
  v8)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v8-gripper-repair-smooth-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    ;;
  v6_combined)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v6-smolvla-fresh-combined88-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    ;;
  v6_new)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v6-smolvla-fresh-new66-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    ;;
  v7_d)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v7-D-obama-only-10k"
    DATASET_REPO_ID="RobotLearningVLA/barack_obama_1"
    echo ">> v7_d is Obama-only — use the prompt 'Place the coke on Barack Obama' only." >&2
    ;;
  *)
    echo "unknown checkpoint name: $CHECKPOINT_NAME (use v8 / v6_combined / v6_new / v7_d)" >&2
    exit 1
    ;;
esac

COMMON_ARGS=(
  --robot.type=so101_follower
  --robot.port="$FOLLOWER_TTY"
  --robot.id=my_awesome_follower_arm
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAM_IDX}, width: 640, height: 480, fps: 30}}"
  --rename_map='{"observation.images.front":"observation.images.camera1"}'
  --policy.device=mps
  --policy.empty_cameras=2
  --policy.num_steps=20
  --policy.n_action_steps=25
  --interpolation_multiplier=2
  --action_smoothing_alpha=0.25
  --max_action_delta_deg=6
  --gripper_open_bias_deg=5
  --gripper_open_bias_threshold_deg=20
  --episode_time_s=20
  --fps=30
  --display_data=true
)

echo ">> Eval 3 deploy battery"
echo "   checkpoint        : $CHECKPOINT_NAME"
echo "   policy.path       : $POLICY_PATH"
echo "   dataset_repo_id   : $DATASET_REPO_ID  (schema source only — no HF push)"
echo "   follower port     : $FOLLOWER_TTY"
echo "   camera index      : $CAM_IDX"
echo ""

exec python scripts/eval3_vla_deploy.py \
  "${COMMON_ARGS[@]}" \
  --dataset_repo_id="$DATASET_REPO_ID" \
  --policy.path="$POLICY_PATH" \
  "$@"

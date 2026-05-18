#!/usr/bin/env bash
# Eval3 SmolVLA deploy baseline — copy to your machine and fill placeholders.
# Uses train-matched dataset stats + smoothing flags from docs/eval3/friend_deploy_handoff.md §4.1.
#
# First print CLI flags from your checkpoint:
#   uv run python tools/eval3_deploy_flags_from_checkpoint.py outputs/train/.../pretrained_model
#
# Then merge printed flags below.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1090
test -f .venv/bin/activate && source .venv/bin/activate

: "${ROBOT_PORT:?Set ROBOT_PORT e.g. /dev/tty.usbmodem...}"
: "${CAM_INDEX:?Set CAM_INDEX e.g. 1}"

POLICY_PATH="${POLICY_PATH:?Local pretrained_model dir or Hub model id}"
DATASET_REPO_ID="${DATASET_REPO_ID:?Must match checkpoint train_config dataset.repo_id}"
RENAME_MAP="${RENAME_MAP:?JSON map e.g. {\"observation.images.front\":\"observation.images.camera1\"}}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-2}"
DEVICE="${DEVICE:-mps}"

exec uv run python scripts/eval3_vla_deploy.py \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID:-my_awesome_follower_arm}" \
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAM_INDEX}, width: 640, height: 480, fps: 30}}" \
  --dataset_repo_id="${DATASET_REPO_ID}" \
  --rename_map="${RENAME_MAP}" \
  --policy.path="${POLICY_PATH}" \
  --policy.device="${DEVICE}" \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --policy.num_steps="${POLICY_NUM_STEPS:-20}" \
  --policy.n_action_steps=25 \
  --interpolation_multiplier=2 \
  --action_smoothing_alpha="${ACTION_SMOOTHING_ALPHA:-0.25}" \
  --max_action_delta_deg="${MAX_ACTION_DELTA_DEG:-6}" \
  --gripper_open_bias_deg="${GRIPPER_OPEN_BIAS_DEG:-5}" \
  --gripper_open_bias_threshold_deg="${GRIPPER_OPEN_BIAS_THRESHOLD_DEG:-20}" \
  --episode_time_s=20 \
  --fps=30 \
  --rollout_log_dir=outputs/eval3_rollouts \
  "$@"

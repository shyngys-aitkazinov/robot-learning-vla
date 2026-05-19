#!/usr/bin/env bash
# Eval 3 deploy battery — runs one of nine checkpoints with the friend-recipe flags.
#
# Friend's recipe (2026-05-18) — four checkpoints, deploy biases ON:
#   v8                : eval3-vla-v8-gripper-repair-smooth-50k            (3-way, latest smoothed)
#   v6_combined       : eval3-vla-v6-smolvla-fresh-combined88-50k         (3-way, combined corpus)
#   v6_new            : eval3-vla-v6-smolvla-fresh-new66-50k              (3-way, new-data only)
#   v7_d              : eval3-vla-v7-D-obama-only-10k                     (Obama-only — use only with the Obama prompt)
#
# 2026-05-19 additions — biases DEACTIVATED (raw-policy deploy):
#   v6_synth_25k      : eval3-smolvla-3way-25k-b128-v6-synth-step25k      (3-way, ChArUco-synth final, 2.86 epochs, loss=0.017)
#   v6_synth_15k      : eval3-smolvla-3way-25k-b128-v6-synth-step15k      (3-way, ChArUco-synth mid, 1.67 epochs, loss=0.021)
#   v9_charuco        : eval3-vla-v9-smolvla-fresh-charuco-50k            (3-way, charuco-only, fresh-from-base)
#   v9_new66_charuco  : eval3-vla-v9-smolvla-fresh-new66-charuco-50k      (3-way, new66 + charuco, fresh-from-base)
#
# FlowerVLA (NOT supported by eval3_vla_deploy.py — entry exits with help):
#   flower_new66      : eval3-flower-new66-50k                            (Florence-2 + FlowerVLA, separate deploy needed)
#
# Port + camera index baked in for Shyngys's rig (/dev/tty.usbmodem5B140317761, cam 0).
# Override per-run via env vars: FOLLOWER_TTY=... CAM_IDX=... ./scripts/run_eval3_deploy_battery.sh v8
#
# Usage:
#   ./scripts/run_eval3_deploy_battery.sh v8
#   ./scripts/run_eval3_deploy_battery.sh v6_combined
#   ./scripts/run_eval3_deploy_battery.sh v6_new
#   ./scripts/run_eval3_deploy_battery.sh v7_d
#   ./scripts/run_eval3_deploy_battery.sh v6_synth_25k
#   ./scripts/run_eval3_deploy_battery.sh v6_synth_15k
#   ./scripts/run_eval3_deploy_battery.sh v9_charuco
#   ./scripts/run_eval3_deploy_battery.sh v9_new66_charuco
#   ./scripts/run_eval3_deploy_battery.sh flower_new66    (prints support note + exits)
#   ./scripts/run_eval3_deploy_battery.sh --help          (print this comment block)
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
  sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

CHECKPOINT_NAME="$1"; shift

# Per-checkpoint flag overrides. Appended AFTER COMMON_ARGS so they win the
# argparse last-wins rule. Use this to disable the friend-recipe biases when
# testing a checkpoint we want to evaluate "raw".
EXTRA_ARGS=()

# The four flags the friend-recipe enables — overriding them back to the
# eval3_vla_deploy.py defaults (see lines 204-207) gives a pure-policy deploy.
NO_BIASES=(
  --gripper_open_bias_deg=0
  --action_smoothing_alpha=0
  --max_action_delta_deg=0
  --interpolation_multiplier=1
)

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
  v6_synth_25k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step25k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    EXTRA_ARGS+=("${NO_BIASES[@]}")
    ;;
  v6_synth_15k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    EXTRA_ARGS+=("${NO_BIASES[@]}")
    ;;
  v9_charuco)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v9-smolvla-fresh-charuco-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    EXTRA_ARGS+=("${NO_BIASES[@]}")
    ;;
  v9_new66_charuco)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v9-smolvla-fresh-new66-charuco-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    EXTRA_ARGS+=("${NO_BIASES[@]}")
    ;;
  flower_new66)
    cat >&2 <<'EOF'
ERROR: flower_new66 is a FlowerVLA model and cannot be deployed via
       scripts/eval3_vla_deploy.py (which is SmolVLA-only).

       FlowerVLA repo layout: checkpoint.pt + dataset_statistics.json
       (raw torch state dict, not the lerobot processor-bundle layout this
       deploy script consumes). It also depends on external/flower_vla_calvin
       which is not checked out in this repo.

       To deploy this model, a flower-specific deploy script is needed.
       See scripts/train_eval3_flower.py for the FlowerVLA import pattern.
EOF
    exit 2
    ;;
  *)
    echo "unknown checkpoint name: $CHECKPOINT_NAME (use v8 / v6_combined / v6_new / v7_d / v6_synth_25k / v6_synth_15k / v9_charuco / v9_new66_charuco / flower_new66)" >&2
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
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "   extra overrides   : ${EXTRA_ARGS[*]}"
fi
echo ""

exec python scripts/eval3_vla_deploy.py \
  "${COMMON_ARGS[@]}" \
  --dataset_repo_id="$DATASET_REPO_ID" \
  --policy.path="$POLICY_PATH" \
  "${EXTRA_ARGS[@]}" \
  "$@"

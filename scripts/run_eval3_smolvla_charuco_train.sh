#!/usr/bin/env bash
# Eval3 SmolVLA v6-style fresh training on the synthetic ChArUco datasets.
#
# Recipes:
#   charuco        : nine dataset_v3_synth_*_2 repos only
#   new66_charuco  : exact new66 v6-truncated data plus the nine synth repos
#
# This intentionally mirrors the label behavior of
# RobotLearningVLA/eval3-vla-v6-smolvla-fresh-new66-50k:
#   - train from lerobot/smolvla_base unless EVAL3_RESUME_FROM is set
#   - keep exact labels: no v8 gripper repair, no action smoothing
#   - keep v6 image/task augmentation and single-camera compatibility

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RECIPE="${EVAL3_CHARUCO_RECIPE:-charuco}"

SYNTH_PRIMARY="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
SYNTH_EXTRAS="RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2,RobotLearningVLA/dataset_v3_synth_barack_obama_left_2,RobotLearningVLA/dataset_v3_synth_taylor_swift_middle_2,RobotLearningVLA/dataset_v3_synth_barack_obama_middle_2,RobotLearningVLA/dataset_v3_synth_yann_lecun_middle_2,RobotLearningVLA/dataset_v3_synth_yann_lecun_right_2,RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2,RobotLearningVLA/dataset_v3_synth_barack_obama_right_2"

NEW66_PRIMARY="RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated"
NEW66_EXTRAS="RobotLearningVLA/dataset_v2_taylor_swift_middle_1_v6_truncated,RobotLearningVLA/dataset_v2_taylor_swift_right_1_v6_truncated,RobotLearningVLA/dataset_v2_yann_lecun_left_1_v6_truncated,RobotLearningVLA/dataset_v2_yann_lecun_middle_1_v6_truncated,RobotLearningVLA/dataset_v2_yann_lecun_right_1_v6_truncated,RobotLearningVLA/dataset_v2_barack_obama_left_1_v6_truncated,RobotLearningVLA/dataset_v2_barack_obama_middle_1_v6_truncated,RobotLearningVLA/dataset_v2_barack_obama_right_1_v6_truncated"

case "$RECIPE" in
  charuco)
    export EVAL3_DATASET_REPO="${EVAL3_DATASET_REPO:-$SYNTH_PRIMARY}"
    export EVAL3_EXTRA_REPOS="${EVAL3_EXTRA_REPOS:-$SYNTH_EXTRAS}"
    export EVAL3_JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v9-smolvla-fresh-charuco-50k}"
    export EVAL3_TRAIN_OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3-vla-v9-smolvla-fresh-charuco-50k}"
    ;;
  new66_charuco)
    export EVAL3_DATASET_REPO="${EVAL3_DATASET_REPO:-$NEW66_PRIMARY}"
    export EVAL3_EXTRA_REPOS="${EVAL3_EXTRA_REPOS:-$NEW66_EXTRAS,$SYNTH_PRIMARY,$SYNTH_EXTRAS}"
    export EVAL3_JOB_NAME="${EVAL3_JOB_NAME:-eval3-vla-v9-smolvla-fresh-new66-charuco-50k}"
    export EVAL3_TRAIN_OUT="${EVAL3_TRAIN_OUT:-outputs/train/eval3-vla-v9-smolvla-fresh-new66-charuco-50k}"
    ;;
  *)
    echo "Unsupported EVAL3_CHARUCO_RECIPE=$RECIPE; expected charuco or new66_charuco" >&2
    exit 2
    ;;
esac

export EVAL3_POLICY_DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
export EVAL3_TRAIN_STEPS="${EVAL3_TRAIN_STEPS:-50000}"
export EVAL3_BATCH="${EVAL3_BATCH:-8}"
export EVAL3_SAVE_FREQ="${EVAL3_SAVE_FREQ:-10000}"
export EVAL3_MAX_FRAMES_PER_EP="${EVAL3_MAX_FRAMES_PER_EP:-600}"
export EVAL3_TASK_AUG="${EVAL3_TASK_AUG:-1}"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.8}"

# Keep the exact v6 labels. Do not use the later v8 label fixes here.
export EVAL3_GRIPPER_REPAIR="${EVAL3_GRIPPER_REPAIR:-0}"
export EVAL3_ACTION_SMOOTH_WINDOW="${EVAL3_ACTION_SMOOTH_WINDOW:-0}"
export EVAL3_ACTION_SMOOTH_GRIPPER="${EVAL3_ACTION_SMOOTH_GRIPPER:-0}"

# The called v6/v8 wrapper sets legacy old-data episode filters by default for
# taylor_swift/yann_lecun/barack_obama repos. Synthetic repos are not dataset_v2,
# so they would otherwise be treated like old data and keep only 14/4/4 episodes.
# Make those filters explicit all-episode filters for the 250-episode synth repos.
ALL_SYNTH_EPISODES="$(seq -s, 0 249)"
export EVAL3_SWIFT_EPISODE_FILTER="${EVAL3_SWIFT_EPISODE_FILTER:-$ALL_SYNTH_EPISODES}"
export EVAL3_LECUN_EPISODE_FILTER="${EVAL3_LECUN_EPISODE_FILTER:-$ALL_SYNTH_EPISODES}"
export EVAL3_OBAMA_EPISODE_FILTER="${EVAL3_OBAMA_EPISODE_FILTER:-$ALL_SYNTH_EPISODES}"

# The original image-transform stack remains enabled in the called wrapper.
# Disable optional mask-based local augmenters unless explicitly requested,
# because Brev training nodes do not necessarily have the generated masks.
export EVAL3_BG_REPLACE="${EVAL3_BG_REPLACE:-0}"
export EVAL3_PRINT_SHUFFLE="${EVAL3_PRINT_SHUFFLE:-0}"

echo ">> Eval3 SmolVLA ChArUco recipe: $RECIPE"
exec ./scripts/run_eval3_smolvla_aug_train.sh "$@"

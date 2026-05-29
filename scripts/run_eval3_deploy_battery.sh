#!/usr/bin/env bash
# Eval 3 deploy battery — runs a named checkpoint through eval3_vla_deploy.py.
#
# Raw vs smoothed: every checkpoint has a default deploy mode. Append a
# `_raw` or `_smooth` suffix to ANY checkpoint name to override it:
#   v8                     -> smoothed (the v8 default)
#   v8_raw                 -> v8 with the deploy biases OFF (pure-policy)
#   v6_synth_25k           -> raw (the v6_synth default)
#   v6_synth_25k_smooth    -> v6_synth_25k with the friend-recipe biases ON
# "raw" resets the four deploy guards to the eval3_vla_deploy.py defaults
# (gripper_open_bias=0, action_smoothing_alpha=0, max_action_delta=0,
# interpolation_multiplier=1); "smoothed" keeps the friend-recipe values
# baked into COMMON_ARGS (interp x2, EMA 0.25, 6 deg slew, +5 deg gripper bias).
#
# Friend's recipe (2026-05-18) — default mode: smoothed:
#   v8                : eval3-vla-v8-gripper-repair-smooth-50k     (3-way, latest smoothed)
#   v6_combined       : eval3-vla-v6-smolvla-fresh-combined88-50k  (3-way, combined corpus)
#   v6_new            : eval3-vla-v6-smolvla-fresh-new66-50k       (3-way, new-data only)
#   v7_d              : eval3-vla-v7-D-obama-only-10k              (Obama-only — Obama prompt only)
#
# 2026-05-19 additions — default mode: raw:
#   v6_synth_25k      : eval3-smolvla-3way-25k-b128-v6-synth-step25k   (ChArUco-synth final)
#   v6_synth_15k      : eval3-smolvla-3way-25k-b128-v6-synth-step15k   (ChArUco-synth mid)
#   v6_synth_5k       : eval3-smolvla-3way-25k-b128-v6-synth-step5k    (ChArUco-synth early)
#   v6_synth5k_500    : eval3-smolvla-3way-5k-b128-v6-synth-step500    (5k-step run, very early)
#   v6_synth5k_1k     : eval3-smolvla-3way-5k-b128-v6-synth-step1k     (5k-step run, 1000 steps)
#   v6_synth5k_1500   : eval3-smolvla-3way-5k-b128-v6-synth-step1500   (5k-step run, 1500 steps)
#   v9_charuco        : eval3-vla-v9-smolvla-fresh-charuco-50k         (charuco-only, fresh-from-base)
#   v9_new66_charuco  : eval3-vla-v9-smolvla-fresh-new66-charuco-50k   (new66 + charuco, fresh-from-base)
#   v6_pinned_warm15k_1k : eval3-smolvla-3way-2k-b128-v6-pinned-warm15k-step1k (pins-pool synth, warm 15k)
#   v6_pinned_warm15k_2k : eval3-smolvla-3way-2k-b128-v6-pinned-warm15k-step2k (pins-pool synth, warm 15k)
#   v10_balanced_new66_10k : eval3-smolvla-v10-balanced-new66-10k      (v4-balanced synth + new66 real)
#   v10_fresh_v4balanced_new66_50k : eval3-vla-v10-smolvla-fresh-v4balanced-new66-50k (v4-balanced + new66, fresh, 50k)
#   v10_h100_expert_20k : eval3-smolvla-v10-h100-expert-20k            (v10 H100 expert-only run, 20k snapshot)
#   v4slots_full        : eval3-vla-v6-smolvla-fresh-v4slots-50k       (9× dataset_v4_* real slots, full FT)
#   v4slots_expert      : eval3-vla-v6-smolvla-fresh-v4slots-expert-50k (9× dataset_v4_* real slots, expert FT)
#   v4slots_expert_10k  : local step 010000 (Brev train dir; not on Hub)
#   v4slots_expert_20k  : local step 020000
#   v4slots_expert_30k  : local step 030000
#   v4slots_expert_40k  : local step 040000
#   Override train dir: EVAL3_V4SLOTS_EXPERT_TRAIN_DIR=outputs/train/...
#
# 2026-05-21 — v16 slot-bottleneck, TWO real cameras, default mode: raw:
#   v16               : eval3-smolvla-v16-pinsv5-step5k (FINAL deployed model;
#                       camera1=current frame, camera2=frozen episode frame-0).
#                       POLICY_PATH defaults to that Hub repo; override with
#                       EVAL3_V16_CKPT=<path-or-hf-repo> for other checkpoints.
#                       The v16 case arm sets rename_map {front->camera1,
#                       front_frame0->camera2} + policy.empty_cameras=1.
#
# 2026-05-29 — v17 v4-anchored bigzoo (v16 slot bottleneck + camera-1 dropout),
# TWO real cameras, default mode: raw:
#   v17               : eval3-smolvla-v17-bigzoo-step3k (v17 mid-training snapshot;
#                       same artifact layout as v16, same rename_map + empty_cameras=1).
#                       POLICY_PATH defaults to that Hub repo; override with
#                       EVAL3_V17_CKPT=<path-or-hf-repo> for other v17 checkpoints
#                       (e.g. ./outputs/v17/checkpoints/004000/pretrained_model
#                       for a local mid-run snapshot). Cam-drop is training-only;
#                       at inference camera1 is always the live capture.
#
# 2026-05-20 additions — pinned ID+OOD aux-head runs, default mode: raw:
#   Fine-tuned (5k run, warm-started from v6-synth step15k, aux loss weight 0.5):
#     v6_idood_aux_warm_500  : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step500
#     v6_idood_aux_warm_1k   : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step1k
#     v6_idood_aux_warm_1500 : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step1500
#     v6_idood_aux_warm_2k   : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step2k
#     v6_idood_aux_warm_2500 : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step2500
#     v6_idood_aux_warm_3k   : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step3k
#     v6_idood_aux_warm_3500 : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step3500
#     v6_idood_aux_warm_4k   : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step4k
#     v6_idood_aux_warm_4500 : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step4500
#     v6_idood_aux_warm_5k   : eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step5k
#   Fresh (20k run, from base lerobot/smolvla_base, aux loss weight 0.5):
#     v6_idood_aux_fresh_5k  : eval3-smolvla-3way-20k-b128-v6-pinned-idood-aux-fresh-step5k
#     v6_idood_aux_fresh_10k : eval3-smolvla-3way-20k-b128-v6-pinned-idood-aux-fresh-step10k
#     v6_idood_aux_fresh_15k : eval3-smolvla-3way-20k-b128-v6-pinned-idood-aux-fresh-step15k
#
# 2026-05-20 — slot-bottleneck 5k run (prefix-side classifier), default mode: raw:
#   slot_3k    : eval3-smolvla-3way-5k-b128-slot-bottleneck-step3k
#   slot_2100  : eval3-smolvla-3way-5k-b128-slot-bottleneck-step2100
#
# FlowerVLA (NOT supported by eval3_vla_deploy.py — entry exits with help):
#   flower_new66      : eval3-flower-new66-50k                     (Florence-2 + FlowerVLA, separate deploy)
#
# Port + camera index baked in for Shyngys's rig (/dev/tty.usbmodem5B140317761, cam 0).
# Override per-run via env vars: FOLLOWER_TTY=... CAM_IDX=... ./scripts/run_eval3_deploy_battery.sh v8
#
# Usage:
#   ./scripts/run_eval3_deploy_battery.sh v8                       # checkpoint default mode
#   ./scripts/run_eval3_deploy_battery.sh v8_raw                   # force raw (biases OFF)
#   ./scripts/run_eval3_deploy_battery.sh v6_synth_25k_smooth      # force smoothed (biases ON)
#   ./scripts/run_eval3_deploy_battery.sh v6_idood_aux_warm_5k     # raw (its default)
#   ./scripts/run_eval3_deploy_battery.sh v6_idood_aux_warm_5k_smooth
#   ./scripts/run_eval3_deploy_battery.sh v6_idood_aux_fresh_15k
#   ./scripts/run_eval3_deploy_battery.sh v6_idood_aux_fresh_15k_smooth
#   ./scripts/run_eval3_deploy_battery.sh flower_new66             (prints support note + exits)
#   ./scripts/run_eval3_deploy_battery.sh --help                   (print this comment block)
#
# The script will prompt you for the task on stdin (e.g. "Place the coke on Taylor Swift"),
# or you can pass --task='Place the coke on X' as an extra argument.
#
# All other eval3_vla_deploy.py flags can also be appended after the checkpoint name.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

FOLLOWER_TTY="${FOLLOWER_TTY:-/dev/tty.usbmodem5B140317761}"
CAM_IDX="${CAM_IDX:-0}"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  awk '/^set -euo/{exit} NR>=2' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

CHECKPOINT_NAME="$1"; shift

# A `_raw` / `_smooth` suffix overrides the checkpoint's default deploy mode.
# Strip it here: the bare name drives the case below, DEPLOY_MODE_OVERRIDE
# drives the bias decision after it.
DEPLOY_MODE_OVERRIDE=""
case "$CHECKPOINT_NAME" in
  *_raw)    DEPLOY_MODE_OVERRIDE="raw";    CHECKPOINT_NAME="${CHECKPOINT_NAME%_raw}" ;;
  *_smooth) DEPLOY_MODE_OVERRIDE="smooth"; CHECKPOINT_NAME="${CHECKPOINT_NAME%_smooth}" ;;
esac

# Per-checkpoint flag overrides. Appended AFTER COMMON_ARGS so they win the
# argparse last-wins rule.
EXTRA_ARGS=()

# The four flags the friend-recipe enables — overriding them back to the
# eval3_vla_deploy.py defaults (see COMMON_ARGS below) gives a pure-policy
# "raw" deploy.
NO_BIASES=(
  --gripper_open_bias_deg=0
  --action_smoothing_alpha=0
  --max_action_delta_deg=0
  --interpolation_multiplier=1
)

# DEFAULT_MODE is set per arm below: "smooth" keeps the friend-recipe biases
# in COMMON_ARGS, "raw" applies NO_BIASES. A `_raw`/`_smooth` name suffix
# overrides it.
DEFAULT_MODE="smooth"

case "$CHECKPOINT_NAME" in
  v8)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v8-gripper-repair-smooth-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    DEFAULT_MODE="smooth"
    ;;
  v6_combined)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v6-smolvla-fresh-combined88-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    DEFAULT_MODE="smooth"
    ;;
  v6_new)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v6-smolvla-fresh-new66-50k"
    DATASET_REPO_ID="RobotLearningVLA/taylor_swift_1"
    DEFAULT_MODE="smooth"
    ;;
  v7_d)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v7-D-obama-only-10k"
    DATASET_REPO_ID="RobotLearningVLA/barack_obama_1"
    DEFAULT_MODE="smooth"
    echo ">> v7_d is Obama-only — use the prompt 'Place the coke on Barack Obama' only." >&2
    ;;
  v6_synth_25k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step25k"
    # Match the training primary (per train_config.json) so ds_meta.features
    # and the (unused but principled) dataset_stats path align with the checkpoint.
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_synth_15k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_synth_5k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step5k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_synth5k_500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-synth-step500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_synth5k_1k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-synth-step1k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_synth5k_1500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-synth-step1500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v9_charuco)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v9-smolvla-fresh-charuco-50k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v9_new66_charuco)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v9-smolvla-fresh-new66-charuco-50k"
    # v9_new66_charuco was trained on the v2 truncated corpus, not the v3 synth one.
    DATASET_REPO_ID="RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated"
    DEFAULT_MODE="raw"
    ;;
  v6_pinned_warm15k_1k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-2k-b128-v6-pinned-warm15k-step1k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_taylor_swift_left_1"
    DEFAULT_MODE="raw"
    ;;
  v6_pinned_warm15k_2k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-2k-b128-v6-pinned-warm15k-step2k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_taylor_swift_left_1"
    DEFAULT_MODE="raw"
    ;;
  v10_balanced_new66_10k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-v10-balanced-new66-10k"
    # v10 v4_balanced_new66 recipe's training primary is the v2 truncated new66 left repo
    # (extras = rest of new66 + 3 v4 balanced synth datasets — see scripts/run_eval3_smolvla_v10_train.sh).
    DATASET_REPO_ID="RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated"
    DEFAULT_MODE="raw"
    ;;
  v10_fresh_v4balanced_new66_50k)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v10-smolvla-fresh-v4balanced-new66-50k"
    # Same v10 v4_balanced_new66 recipe as v10_balanced_new66_10k; training
    # primary per the checkpoint's train_config.json is the v2 truncated new66 left repo.
    DATASET_REPO_ID="RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated"
    DEFAULT_MODE="raw"
    ;;
  v10_h100_expert_20k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-v10-h100-expert-20k"
    # train_config.json: same v2 truncated new66 left primary, expert-only H100 run.
    DATASET_REPO_ID="RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated"
    DEFAULT_MODE="raw"
    ;;

  # --- 2026-05-20: pinned ID+OOD aux-head fine-tuned run (5k, warm 15k) ---
  # All share the training primary dataset_v3_synth_pinned_idood_taylor_swift_left_2
  # so ds_meta.features align with the checkpoint. Default raw (aux-head run,
  # evaluated like the v6_synth entries); append _smooth for the friend-recipe biases.
  v6_idood_aux_warm_500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_1k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step1k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_1500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step1500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_2k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step2k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_2500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step2500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_3k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step3k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_3500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step3500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_4k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step4k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_4500)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step4500"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_warm_5k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-v6-pinned-idood-aux-warm15k-step5k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;

  # --- 2026-05-20: pinned ID+OOD aux-head fresh-from-base run (20k) ---
  v6_idood_aux_fresh_5k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-20k-b128-v6-pinned-idood-aux-fresh-step5k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_fresh_10k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-20k-b128-v6-pinned-idood-aux-fresh-step10k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  v6_idood_aux_fresh_15k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-20k-b128-v6-pinned-idood-aux-fresh-step15k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;

  # --- 2026-05-20: slot-bottleneck 5k run — mid-training test checkpoints ---
  # eval3_vla_deploy.py auto-detects the slot head from the checkpoint weights
  # and applies the slot-bottleneck patch (needed — these carry an h_slot
  # prefix token). Training primary = pinned ID+OOD taylor_left_2.
  slot_3k)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-slot-bottleneck-step3k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;
  slot_2100)
    POLICY_PATH="RobotLearningVLA/eval3-smolvla-3way-5k-b128-slot-bottleneck-step2100"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
    DEFAULT_MODE="raw"
    ;;

  # --- 2026-05-21: v16 slot-bottleneck, TWO real cameras ---
  # camera1 = current frame, camera2 = frozen episode frame-0. v16 needs the
  # 2-camera rename_map AND policy.empty_cameras=1 — both OVERRIDE the
  # single-camera COMMON_ARGS (EXTRA_ARGS is applied last, so it wins).
  # eval3_vla_deploy.py also auto-detects v16 and adopts the rename_map from
  # train_config.json, so this is belt-and-suspenders.
  v16)
    POLICY_PATH="${EVAL3_V16_CKPT:-RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k}"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v4_taylor_left"
    DEFAULT_MODE="raw"
    EXTRA_ARGS+=(
      --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}'
      --policy.empty_cameras=1
    )
    ;;

  # --- 2026-05-29: v17 v4-anchored "bigzoo" mid-training snapshot ---
  # v17 is architecturally identical to v16 at inference (slot bottleneck +
  # frame-0 cam2 + camera-1 noise during training only). Same 2-camera
  # rename_map + policy.empty_cameras=1. eval3_vla_deploy.py also auto-detects
  # the rename_map from train_config.json.
  v17)
    POLICY_PATH="${EVAL3_V17_CKPT:-RobotLearningVLA/eval3-smolvla-v17-bigzoo-step3k}"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v4_taylor_left"
    DEFAULT_MODE="raw"
    EXTRA_ARGS+=(
      --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}'
      --policy.empty_cameras=1
    )
    ;;

  # --- v4 real slot corpora (50k, May 2026) ---
  v4slots_full)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-50k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v4_taylor_left"
    DEFAULT_MODE="raw"
    ;;
  v4slots_expert)
    POLICY_PATH="RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k"
    DATASET_REPO_ID="RobotLearningVLA/dataset_v4_taylor_left"
    DEFAULT_MODE="raw"
    ;;
  v4slots_expert_10k|v4slots_expert_20k|v4slots_expert_30k|v4slots_expert_40k)
    _V4_EXPERT_TRAIN="${EVAL3_V4SLOTS_EXPERT_TRAIN_DIR:-outputs/train/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k}"
    case "$CHECKPOINT_NAME" in
      v4slots_expert_10k) _V4_STEP=010000 ;;
      v4slots_expert_20k) _V4_STEP=020000 ;;
      v4slots_expert_30k) _V4_STEP=030000 ;;
      v4slots_expert_40k) _V4_STEP=040000 ;;
    esac
    POLICY_PATH="${_V4_EXPERT_TRAIN}/checkpoints/${_V4_STEP}/pretrained_model"
    if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
      echo "ERROR: missing local checkpoint: ${POLICY_PATH}/model.safetensors" >&2
      echo "       Run: ./scripts/fetch_eval3_v4slots_expert_checkpoints.sh" >&2
      echo "       Or set EVAL3_V4SLOTS_EXPERT_TRAIN_DIR to your train output dir." >&2
      exit 2
    fi
    DATASET_REPO_ID="RobotLearningVLA/dataset_v4_taylor_left"
    DEFAULT_MODE="raw"
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
    echo "unknown checkpoint name: '$CHECKPOINT_NAME' — run with --help for the full list." >&2
    echo "(append _raw or _smooth to any name to override its default deploy mode.)" >&2
    exit 1
    ;;
esac

# Resolve the deploy mode: a `_raw`/`_smooth` name suffix wins over the
# per-checkpoint DEFAULT_MODE. "raw" applies NO_BIASES on top of COMMON_ARGS.
DEPLOY_MODE="${DEPLOY_MODE_OVERRIDE:-$DEFAULT_MODE}"
if [[ "$DEPLOY_MODE" == "raw" ]]; then
  EXTRA_ARGS+=("${NO_BIASES[@]}")
fi

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
  --display_data=false
)

if [[ "$DEPLOY_MODE" == "raw" ]]; then
  MODE_DESC="biases OFF — pure policy distribution"
else
  MODE_DESC="friend-recipe biases ON (interp x2, EMA 0.25, 6deg slew, +5deg gripper)"
fi

echo ">> Eval 3 deploy battery"
echo "   checkpoint        : $CHECKPOINT_NAME"
echo "   deploy mode       : $DEPLOY_MODE  ($MODE_DESC)"
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
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  "$@"

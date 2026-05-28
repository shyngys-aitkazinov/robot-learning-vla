#!/usr/bin/env bash
# Eval3 SmolVLA v17 — v16 slot-bottleneck + CAMERA-1 DROPOUT for robust
# "lang + frame-0 -> trajectory" learning. Trains on the same v16 corpus
# (9 real dataset_v4_* + 9 synthetic dataset_v3_synth_pinned_idood_*_3) and
# adds an optional 9 legacy dataset_v2_* repos for extra real signal.
#
# v17 vs v16 (run_eval3_smolvla_v16_real_data_slot_train.sh):
#   - CameraDropAugmenter is ON by default. Per training sample the cam1
#     stream (observation.images.front) has a chance of being replaced by
#     Gaussian noise. cam2 (observation.images.front_frame0 = the cached
#     frame-0 view that the slot head reads) is NEVER touched and is
#     bit-exactly preserved across the drop -- enforced by a .clone() in
#     the prep dataset's pre-grasp branch + a hard runtime assertion. See
#     tools/eval3_v16_dataset_test.py (Phase E) for the 4 invariant tests
#     and docs/eval3/v17_playbook.md for the rationale.
#   - Three drop modes combine, dominant first (env-tunable):
#       * per-episode drop (default 0.35): whole episode's cam1 = noise,
#         deterministic per (ep_idx, epoch_bucket) hash so workers agree.
#       * per-frame iid drop (default 0.10): on non-dropped episodes.
#       * pre/post-grasp asymmetry (default mult=3.0): post-grasp frames
#         get 3x the per-frame rate, targeting the can-motion shortcut.
#   - Optional corpus toggle EVAL3_V17_INCLUDE_V2=1 ALSO concats the 9
#     legacy dataset_v2_* repos (extra ~5k frames of real signal).
#   - Everything else (slot head, frame-0, pre-grasp CE mask, state aug,
#     image transforms, optimizer, scheduler) is IDENTICAL to v16.
#
# Smoke (~5 min on MPS, datasets cached from v16 if available):
#   EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=12 EVAL3_WANDB=0 \
#     EVAL3_POLICY_DEVICE=mps \
#     EVAL3_TRAIN_OUT=/tmp/eval3_v17_smoke \
#     EVAL3_JOB_NAME=eval3_v17_smoke \
#     ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh --log_freq=2
#
# Full run:
#   ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh 2>&1 \
#     | tee outputs/train/logs/eval3_v17_camdrop.log
#
# Knobs (defaults shown):
#   EVAL3_TRAIN_STEPS=10000  EVAL3_BATCH=128      EVAL3_SAVE_FREQ=1000
#   EVAL3_POLICY_DEVICE=cuda EVAL3_PEAK_LR=1e-4   EVAL3_DECAY_LR=1e-6
#   EVAL3_WARMUP_STEPS=250   EVAL3_DECAY_STEPS=(STEPS-WARMUP)
#   EVAL3_RESUME_FROM=lerobot/smolvla_base   (fresh-from-base)
#
# Slot bottleneck (inherited from v16; same defaults):
#   EVAL3_SLOT_LOSS_WEIGHT=0.5  EVAL3_SLOT_HIDDEN=256  EVAL3_SLOT_BOTTLENECK=64
#   EVAL3_SLOT_DROPOUT=0.2  EVAL3_SLOT_STOPGRAD=1  EVAL3_SLOT_GUMBEL=0
#   EVAL3_SLOT_FRAME0=1  EVAL3_SLOT_CE_PREGRASP_ONLY=1  EVAL3_GRASP_GRIP_DELTA=20
#
# Camera-1 dropout (NEW in v17; opt-OUT via EVAL3_CAM1_DROP=0):
#   EVAL3_CAM1_DROP=1                  master switch (default ON)
#   EVAL3_CAM1_DROP_EPISODE_P=0.35     dominant: full-episode drop probability
#   EVAL3_CAM1_DROP_FRAME_P=0.10       per-frame iid (on non-dropped episodes)
#   EVAL3_CAM1_DROP_POSTGRASP_MULT=3.0 per-frame multiplier post-grasp
#   EVAL3_CAM1_DROP_NOISE_MEAN=0.5  EVAL3_CAM1_DROP_NOISE_STD=0.25
#   EVAL3_CAM1_DROP_EPOCH_WINDOW=5000  steps per per-episode reroll
#
# Corpus toggles (one mode at a time; INCLUDE_V2 / INCLUDE_ALGVR / INCLUDE_PINS30Q5 are additive):
#   default                    -> 9 real v4 + 9 synth v3_3   (full v17 corpus)
#   EVAL3_V17_NO_SYNTH=1       -> 9 real v4 only
#   EVAL3_V17_SYNTH_ONLY=1     -> 9 synth v3_3 only
#   EVAL3_V17_INCLUDE_V2=1     -> ALSO append the 9 legacy dataset_v2_* repos
#   EVAL3_V17_INCLUDE_ALGVR=1  -> ALSO append all local dataset_v5_synth_algvr_*_full
#                                  (built by scripts/run_eval3_synth_algvr_dataset_gen.sh
#                                  from algvr-conference.json + v5 charuko sources;
#                                  up to 102 datasets / ~594 eps / ~296k frames at
#                                  the default M=3 generator knob)
#   EVAL3_V17_INCLUDE_PINS30Q5=1 -> ALSO append all local dataset_v5_synth_pins30q5_*_full
#                                  (built by scripts/run_eval3_synth_pins30q5_dataset_gen.sh
#                                  from pins-face-recognition-top30-quality.json + v5
#                                  charuko sources; up to 90 datasets / ~1350 eps /
#                                  ~640k frames at the default N=5 / M=3 knobs)
#   EVAL3_WANDB=1  EVAL3_WANDB_PROJECT=eval3-v17-camdrop
#   EVAL3_DRY_RUN=0
#
# Held-out validation watcher (OPT-IN; default OFF). When EVAL3_VAL_WATCH=1
# AND EVAL3_VAL_REPOS is set, this launcher backgrounds tools/eval3_val_watcher.py
# alongside training. The watcher polls $OUT/checkpoints/ and scores each
# new checkpoint on the held-out LeRobot-format datasets you list, writing
# JSONL metrics to $OUT/val_metrics.jsonl. See docs/eval3/v17_playbook.md
# §4.1 for details.
#   EVAL3_VAL_WATCH=0                  # set to 1 to enable
#   EVAL3_VAL_REPOS=org/name,org/other # required; comma-separated repo list
#   EVAL3_VAL_LOCAL_REPOS=             # subset to load from ./datasets/<name>
#   EVAL3_VAL_EPISODES_PER_REPO=3      # episodes sampled per val repo
#   EVAL3_VAL_FRAMES_PER_EPISODE=30    # frames per episode (uniform stride)
#   EVAL3_VAL_DEVICE=mps               # cpu/mps/cuda; defaults to EVAL3_POLICY_DEVICE
#   EVAL3_VAL_POLL_SEC=60              # checkpoint poll interval
#   EVAL3_VAL_IDLE_SEC=600             # stop after N seconds of no new ckpt
#   EVAL3_VAL_WANDB=0                  # 1 -> sidecar wandb run "<job>_val"

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- Dataset slates --------------------------------------------------------

# Slate 1 — real teleop data (9 datasets, ~30k frames).
REAL_PRIMARY="RobotLearningVLA/dataset_v4_taylor_left"
REAL_EXTRAS="RobotLearningVLA/dataset_v4_taylor_middle,\
RobotLearningVLA/dataset_v4_taylor_right,\
RobotLearningVLA/dataset_v4_barack_left,\
RobotLearningVLA/dataset_v4_barack_middle,\
RobotLearningVLA/dataset_v4_barack_right,\
RobotLearningVLA/dataset_v4_yann_left,\
RobotLearningVLA/dataset_v4_yann_middle,\
RobotLearningVLA/dataset_v4_yann_right"

# Slate 2 — Pins synthetic (9 datasets, ~180k frames). NOT on the Hub;
# load from ./datasets/<name> via EVAL3_LOCAL_REPOS.
SYNTH_PRIMARY="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_3"
SYNTH_EXTRAS="RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_middle_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_right_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_left_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_middle_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_barack_obama_right_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_left_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_middle_3,\
RobotLearningVLA/dataset_v3_synth_pinned_idood_yann_lecun_right_3"
SYNTH_ALL="${SYNTH_PRIMARY},${SYNTH_EXTRAS}"

# Slate 3 — legacy v2 real (9 datasets, ~5k frames). OPT-IN, additive.
LEGACY_V2_EXTRAS="RobotLearningVLA/dataset_v2_taylor_swift_left_1,\
RobotLearningVLA/dataset_v2_taylor_swift_middle_1,\
RobotLearningVLA/dataset_v2_taylor_swift_right_1,\
RobotLearningVLA/dataset_v2_barack_obama_left_1,\
RobotLearningVLA/dataset_v2_barack_obama_middle_1,\
RobotLearningVLA/dataset_v2_barack_obama_right_1,\
RobotLearningVLA/dataset_v2_yann_lecun_left_1,\
RobotLearningVLA/dataset_v2_yann_lecun_middle_1,\
RobotLearningVLA/dataset_v2_yann_lecun_right_1"

# Slate 4 — algvr.com conference synthetic (up to 102 datasets, ~594 episodes /
# ~296k frames at the default M=3 generation knob). OPT-IN, additive. LOCAL
# ONLY — discovered by glob to avoid hardcoding 102 names.
#
# Provenance: built by ./scripts/run_eval3_synth_algvr_dataset_gen.sh from
#   - sources: datasets/dataset_v5_charuko_{left,middle,right}_full
#   - pool   : datasets/algvr-conference.json (34 organizers + speakers from
#              https://algvr.com/conference/)
# Output names follow dataset_v5_synth_algvr_<celeb_slug>_<position>_full.
ALGVR_NAMES=""
ALGVR_EXTRAS=""
if [[ "${EVAL3_V17_INCLUDE_ALGVR:-0}" == "1" ]]; then
  # Resolve at script-startup time so re-generation under datasets/ is picked
  # up automatically. Stable lexical sort -> deterministic concat order.
  ALGVR_NAMES=$(/bin/ls -1d "$ROOT"/datasets/dataset_v5_synth_algvr_*_full 2>/dev/null \
                  | xargs -n1 basename 2>/dev/null | sort || true)
  if [[ -z "$ALGVR_NAMES" ]]; then
    echo "WARN EVAL3_V17_INCLUDE_ALGVR=1 but no datasets/dataset_v5_synth_algvr_*_full found." >&2
    echo "     Build them first: ./scripts/run_eval3_synth_algvr_dataset_gen.sh" >&2
  else
    ALGVR_EXTRAS=$(echo "$ALGVR_NAMES" | sed 's|^|RobotLearningVLA/|' | paste -sd, -)
  fi
fi

# Slate 5 — PINS top-30 quality-filtered synthetic (up to 90 datasets, ~1350
# episodes / ~640k frames at the default N=5 / M=3 generation knobs).
# OPT-IN, additive. LOCAL ONLY — discovered by glob, parallels slate 4.
#
# Provenance: built by ./scripts/run_eval3_synth_pins30q5_dataset_gen.sh from
#   - sources: datasets/dataset_v5_charuko_{left,middle,right}_full
#   - pool   : datasets/pins-face-recognition-top30-quality.json (30 celebs,
#              quality_photos field sorted best-first)
# Output names follow dataset_v5_synth_pins30q5_<celeb_slug>_<position>_full.
PINS30Q5_NAMES=""
PINS30Q5_EXTRAS=""
if [[ "${EVAL3_V17_INCLUDE_PINS30Q5:-0}" == "1" ]]; then
  PINS30Q5_NAMES=$(/bin/ls -1d "$ROOT"/datasets/dataset_v5_synth_pins30q5_*_full 2>/dev/null \
                     | xargs -n1 basename 2>/dev/null | sort || true)
  if [[ -z "$PINS30Q5_NAMES" ]]; then
    echo "WARN EVAL3_V17_INCLUDE_PINS30Q5=1 but no datasets/dataset_v5_synth_pins30q5_*_full found." >&2
    echo "     Build them first: ./scripts/run_eval3_synth_pins30q5_dataset_gen.sh" >&2
  else
    PINS30Q5_EXTRAS=$(echo "$PINS30Q5_NAMES" | sed 's|^|RobotLearningVLA/|' | paste -sd, -)
  fi
fi

# ---- Corpus selection ------------------------------------------------------

if [[ "${EVAL3_V17_SYNTH_ONLY:-0}" == "1" ]]; then
  REPO="$SYNTH_PRIMARY"
  EXTRA_REPOS="$SYNTH_EXTRAS"
  LOCAL_REPOS="$SYNTH_ALL"
elif [[ "${EVAL3_V17_NO_SYNTH:-0}" == "1" ]]; then
  REPO="$REAL_PRIMARY"
  EXTRA_REPOS="$REAL_EXTRAS"
  LOCAL_REPOS=""
else
  REPO="$REAL_PRIMARY"
  EXTRA_REPOS="${REAL_EXTRAS},${SYNTH_ALL}"
  LOCAL_REPOS="$SYNTH_ALL"
fi

if [[ "${EVAL3_V17_INCLUDE_V2:-0}" == "1" ]]; then
  EXTRA_REPOS="${EXTRA_REPOS},${LEGACY_V2_EXTRAS}"
fi

# Algvr slate is additive on top of whichever mode above (real-only, synth-only,
# or default real+synth) and is fully LOCAL so it also extends LOCAL_REPOS.
if [[ "${EVAL3_V17_INCLUDE_ALGVR:-0}" == "1" && -n "$ALGVR_EXTRAS" ]]; then
  EXTRA_REPOS="${EXTRA_REPOS},${ALGVR_EXTRAS}"
  LOCAL_REPOS="${LOCAL_REPOS:+$LOCAL_REPOS,}${ALGVR_EXTRAS}"
fi

# Pins30q5 slate — same additive + local-only wiring as the algvr slate.
if [[ "${EVAL3_V17_INCLUDE_PINS30Q5:-0}" == "1" && -n "$PINS30Q5_EXTRAS" ]]; then
  EXTRA_REPOS="${EXTRA_REPOS},${PINS30Q5_EXTRAS}"
  LOCAL_REPOS="${LOCAL_REPOS:+$LOCAL_REPOS,}${PINS30Q5_EXTRAS}"
fi

# ---- Run config ------------------------------------------------------------

OUT="${EVAL3_TRAIN_OUT:-/ephemeral/outputs/train/eval3_v17_camdrop}"
JOB="${EVAL3_JOB_NAME:-eval3_v17_camdrop}"
STEPS="${EVAL3_TRAIN_STEPS:-10000}"
BATCH="${EVAL3_BATCH:-128}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
# camera1 = current frame; camera2 = the episode's frame-0 scene (slot head).
RENAMES='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}'
POLICY_PATH="${EVAL3_RESUME_FROM:-lerobot/smolvla_base}"

PEAK_LR="${EVAL3_PEAK_LR:-1e-4}"
WARMUP_STEPS="${EVAL3_WARMUP_STEPS:-250}"
DECAY_STEPS="${EVAL3_DECAY_STEPS:-$(( STEPS - WARMUP_STEPS ))}"
[[ "$DECAY_STEPS" -lt 1 ]] && DECAY_STEPS=1   # guard tiny-step smoke runs
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-1000}"

# draccus wants literal true/false for bool flags — map the 1/0 env knob.
if [[ "${EVAL3_WANDB:-1}" == "1" || "${EVAL3_WANDB:-1}" == "true" ]]; then
  WANDB_ENABLE="true"
else
  WANDB_ENABLE="false"
fi
WANDB_PROJECT="${EVAL3_WANDB_PROJECT:-eval3-v17-camdrop}"

# ---- Common exports (same as v16) ------------------------------------------

export EVAL3_EXTRA_REPOS="$EXTRA_REPOS"
export EVAL3_LOCAL_REPOS="$LOCAL_REPOS"
export EVAL3_MAX_FRAMES_PER_EP="0"
export EVAL3_TASK_AUG="1"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-0.7}"
export EVAL3_BG_REPLACE="0"
export EVAL3_BG_REPLACE_P="0.0"
export EVAL3_PRINT_SHUFFLE="0"
export EVAL3_PRINT_SHUFFLE_P="0.0"
# v16/v17 frame-0 mode is incompatible with the prep cache (cache-hit path
# skips the frame-0 build) — keep the cache off.
export EVAL3_PREP_CACHE="0"

# --- Slot-bottleneck head (inherited from v16) ------------------------------
export EVAL3_SLOT_LOSS_WEIGHT="${EVAL3_SLOT_LOSS_WEIGHT:-0.5}"
export EVAL3_SLOT_HIDDEN="${EVAL3_SLOT_HIDDEN:-256}"
export EVAL3_SLOT_BOTTLENECK="${EVAL3_SLOT_BOTTLENECK:-64}"
export EVAL3_SLOT_DROPOUT="${EVAL3_SLOT_DROPOUT:-0.2}"
export EVAL3_SLOT_STOPGRAD="${EVAL3_SLOT_STOPGRAD:-1}"
export EVAL3_SLOT_GUMBEL="${EVAL3_SLOT_GUMBEL:-0}"
export EVAL3_SLOT_FRAME0="${EVAL3_SLOT_FRAME0:-1}"
export EVAL3_SLOT_CE_PREGRASP_ONLY="${EVAL3_SLOT_CE_PREGRASP_ONLY:-1}"
export EVAL3_GRASP_GRIP_DELTA="${EVAL3_GRASP_GRIP_DELTA:-20}"
# Lever A (NEW in v17): append the h_slot prefix token K=4 times instead of
# once. Boosts the action expert's attention budget for the slot signal from
# ~1/n_prefix to ~K/n_prefix without changing slot_proj (one adapter shared
# by all K copies). v16 baseline = K=1; set K=1 to reproduce v16 bit-exactly.
export EVAL3_SLOT_TOKEN_REPEAT="${EVAL3_SLOT_TOKEN_REPEAT:-4}"
unset EVAL3_AUX_POS_LOSS_WEIGHT   # the entry script picks slot XOR aux

# --- Camera-1 dropout (NEW in v17; ON by default) ---------------------------
export EVAL3_CAM1_DROP="${EVAL3_CAM1_DROP:-1}"
export EVAL3_CAM1_DROP_EPISODE_P="${EVAL3_CAM1_DROP_EPISODE_P:-0.35}"
export EVAL3_CAM1_DROP_FRAME_P="${EVAL3_CAM1_DROP_FRAME_P:-0.10}"
export EVAL3_CAM1_DROP_POSTGRASP_MULT="${EVAL3_CAM1_DROP_POSTGRASP_MULT:-3.0}"
export EVAL3_CAM1_DROP_NOISE_MEAN="${EVAL3_CAM1_DROP_NOISE_MEAN:-0.5}"
export EVAL3_CAM1_DROP_NOISE_STD="${EVAL3_CAM1_DROP_NOISE_STD:-0.25}"
export EVAL3_CAM1_DROP_EPOCH_WINDOW="${EVAL3_CAM1_DROP_EPOCH_WINDOW:-5000}"

# --- Stochastic state augmentation — unchanged from v16 ---------------------
export EVAL3_STATE_NOISE_SIGMA_MAX="${EVAL3_STATE_NOISE_SIGMA_MAX:-0.7}"
export EVAL3_STATE_NOISE_SIGMA_MIN="${EVAL3_STATE_NOISE_SIGMA_MIN:-0.15}"
export EVAL3_STATE_NOISE_CURRICULUM_STEPS="${EVAL3_STATE_NOISE_CURRICULUM_STEPS:-${STEPS}}"
export EVAL3_STATE_REPLACE_PROB="${EVAL3_STATE_REPLACE_PROB:-0.65}"
export EVAL3_STATE_REPLACE_MODES="${EVAL3_STATE_REPLACE_MODES:-home:0.5,zero:0.5}"
export EVAL3_STATE_HOME_JITTER_SIGMA="${EVAL3_STATE_HOME_JITTER_SIGMA:-1.5}"
export EVAL3_STATE_GRIPPER_NOISE_SCALE="${EVAL3_STATE_GRIPPER_NOISE_SCALE:-0.15}"
export EVAL3_STATE_POSTGRASP_GRIP_THRESH="${EVAL3_STATE_POSTGRASP_GRIP_THRESH:-35}"
export EVAL3_STATE_POSTGRASP_SIGMA_MULT="${EVAL3_STATE_POSTGRASP_SIGMA_MULT:-4.0}"
export EVAL3_STATE_POSTGRASP_REPLACE_PROB="${EVAL3_STATE_POSTGRASP_REPLACE_PROB:-0.95}"

unset EVAL3_SWIFT_EPISODE_FILTER
unset EVAL3_LECUN_EPISODE_FILTER
unset EVAL3_OBAMA_EPISODE_FILTER

# ---- Pre-launch summary ----------------------------------------------------

echo ">> Eval3 v17 SmolVLA slot-bottleneck + cam1-drop — corpus:"
if [[ "${EVAL3_V17_SYNTH_ONLY:-0}" == "1" ]]; then
  echo "     mode             : SYNTH-ONLY (9 dataset_v3_synth_pinned_idood_*_3)"
elif [[ "${EVAL3_V17_NO_SYNTH:-0}" == "1" ]]; then
  echo "     mode             : REAL-ONLY (9 dataset_v4_*)"
else
  echo "     mode             : REAL + SYNTH (9 dataset_v4_* + 9 dataset_v3_synth_pinned_idood_*_3)"
fi
[[ "${EVAL3_V17_INCLUDE_V2:-0}" == "1" ]] && echo "     v2 add-on        : +9 legacy dataset_v2_*"
if [[ "${EVAL3_V17_INCLUDE_ALGVR:-0}" == "1" ]]; then
  algvr_n=$(echo "$ALGVR_NAMES" | grep -c . || true)
  echo "     algvr add-on     : +${algvr_n} dataset_v5_synth_algvr_*_full (local, from algvr-conference.json)"
fi
if [[ "${EVAL3_V17_INCLUDE_PINS30Q5:-0}" == "1" ]]; then
  pins30q5_n=$(echo "$PINS30Q5_NAMES" | grep -c . || true)
  echo "     pins30q5 add-on  : +${pins30q5_n} dataset_v5_synth_pins30q5_*_full (local, from pins top-30 quality)"
fi
echo "   dataset (primary)     : $REPO"
echo "   virtual extras        : $EVAL3_EXTRA_REPOS"
echo "   local-root repos      : ${EVAL3_LOCAL_REPOS:-<none>}"
echo "   policy.path           : $POLICY_PATH  (fresh-from-base)"
echo "   device                : $DEVICE"
echo "   steps / batch         : $STEPS / $BATCH"
echo "   save_freq             : $SAVE_FREQ"
echo "   warmup / decay steps  : $WARMUP_STEPS / $DECAY_STEPS"
echo "   wandb                 : enable=$WANDB_ENABLE project=$WANDB_PROJECT"
echo "   slot loss weight      : $EVAL3_SLOT_LOSS_WEIGHT  (frame0=$EVAL3_SLOT_FRAME0 ce_pregrasp_only=$EVAL3_SLOT_CE_PREGRASP_ONLY)"
echo "   slot token repeat     : K=$EVAL3_SLOT_TOKEN_REPEAT  (Lever A; v16 baseline = 1)"
echo "   grasp grip delta      : $EVAL3_GRASP_GRIP_DELTA"
echo "   cam1 drop             : enable=$EVAL3_CAM1_DROP  ep_p=$EVAL3_CAM1_DROP_EPISODE_P  frame_p=$EVAL3_CAM1_DROP_FRAME_P  postgrasp_mult=$EVAL3_CAM1_DROP_POSTGRASP_MULT"
echo "   rename_map            : $RENAMES"
echo "   output dir            : $OUT"

if [[ "${EVAL3_DRY_RUN:-0}" == "1" ]]; then
  echo ">> EVAL3_DRY_RUN=1; not starting training."
  exit 0
fi

# ---- Optional held-out validation watcher ----------------------------------
# Backgrounds tools/eval3_val_watcher.py which polls $OUT/checkpoints/ and
# scores each new ckpt on EVAL3_VAL_REPOS. JSONL -> $OUT/val_metrics.jsonl.
if [[ "${EVAL3_VAL_WATCH:-0}" == "1" && -n "${EVAL3_VAL_REPOS:-}" ]]; then
  mkdir -p outputs/train/logs
  VAL_LOG="outputs/train/logs/${JOB}_val_watch.log"
  nohup python tools/eval3_val_watcher.py \
    --train-out "$OUT" \
    --final-step "$STEPS" \
    > "$VAL_LOG" 2>&1 &
  echo ">> val watcher backgrounded (pid $!) — log: $VAL_LOG"
elif [[ "${EVAL3_VAL_WATCH:-0}" == "1" ]]; then
  echo "WARN EVAL3_VAL_WATCH=1 but EVAL3_VAL_REPOS is empty — watcher NOT started." >&2
fi

# Same image transforms as v15/v16. Applied to BOTH cam1 and cam2 before
# our cam-drop runs (cam_drop replaces cam1 with noise post-transforms; cam2
# still carries the transformed real frame-0).
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.6,1.4]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.6,1.4]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.7,1.3]}},"hue":{"weight":0.5,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":0.75,"type":"SharpnessJitter","kwargs":{"sharpness":[0.6,1.4]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.04,0.04]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.18,"p":0.4}},"random_erasing":{"weight":1.0,"type":"RandomErasing","kwargs":{"p":0.4,"scale":[0.02,0.08],"ratio":[0.3,3.3],"value":0}}}'

exec python scripts/train_eval3_smolvla.py \
  --policy.path="$POLICY_PATH" \
  --policy.push_to_hub=false \
  --policy.compile_model=false \
  --policy.device="$DEVICE" \
  --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=true \
  --rename_map="$RENAMES" \
  --dataset.repo_id="$REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs="$TFS_JSON" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
  --use_policy_training_preset=false \
  --optimizer.type=adamw \
  --optimizer.lr="$PEAK_LR" \
  --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10.0 \
  --scheduler.type=cosine_decay_with_warmup \
  --scheduler.peak_lr="$PEAK_LR" \
  --scheduler.decay_lr="$DECAY_LR" \
  --scheduler.num_warmup_steps="$WARMUP_STEPS" \
  --scheduler.num_decay_steps="$DECAY_STEPS" \
  --job_name="$JOB" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --batch_size="$BATCH" \
  "$@"

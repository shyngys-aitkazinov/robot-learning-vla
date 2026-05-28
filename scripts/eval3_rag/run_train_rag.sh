#!/usr/bin/env bash
# Eval3 RAG SmolVLA training launcher.
#
# Trains a SmolVLA variant where camera2 = retrieved celebrity reference face
# (from datasets/celeb_refdb/) instead of an empty pad.  The frozen vision
# encoder processes both the live scene (camera1) and the reference face
# (camera2); the action expert learns visual matching rather than celebrity-
# name memorisation, which generalises to OOD celebrities in the DB.
#
# ── PRE-TRAINING CHECKLIST (run once before this script) ──────────────────
#
#   Step 1 — install SmolVLA deps:
#     EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
#
#   Step 2 — build reference face DB from local source datasets:
#     python scripts/eval3_rag/build_celeb_refdb.py
#     # Bulk-download HQ portraits for OOD celebrities (recommended):
#     python scripts/eval3_rag/download_refdb.py \
#         --slugs elon_musk cristiano_ronaldo lionel_messi rihanna \
#                 leonardo_dicaprio tom_cruise dwayne_johnson robert_downey_jr \
#                 scarlett_johansson jennifer_lawrence marc_pollefeys
#     # NOTE: run_train_rag.sh auto-populates TOY celebrity portraits
#     # (taylor_swift, barack_obama, yann_lecun) before training starts.
#
#   Step 3 — generate synthetic training episodes for OOD celebrities and
#   push to HuggingFace (EVAL3_RAG_OOD_REPOS defaults to 1 so these are
#   included in training automatically once uploaded):
#     huggingface-cli login
#     ./scripts/eval3_rag/gen_rag_synth.sh
#   This produces yukk1/dataset_v3_synth_rag1_<celeb>_<pos>_2 on HF.
#
#   Step 4 — train:
#     ./scripts/eval3_rag/run_train_rag.sh 2>&1 | tee outputs/train/logs/test_eval3_v1.log
#
#   Step 5 — push trained model to HF (after training completes):
#     huggingface-cli login
#     python -c "
#     from huggingface_hub import HfApi
#     api = HfApi()
#     api.upload_folder(
#         folder_path='outputs/train/test_eval3_v1/checkpoints/050000/pretrained_model',
#         repo_id='yukk1/test-eval3-v1-smolvla-50k',
#         repo_type='model',
#     )
#     "
#
# ── Knobs (env vars, defaults shown) ──────────────────────────────────────
#   EVAL3_RAG_REFDB=datasets/celeb_refdb   celebrity face image directory
#   EVAL3_RAG_HF_ORG=yukk1                 HF org where synth datasets live
#   EVAL3_RAG_OOD_REPOS=1                  0 to exclude OOD synth repos
#   EVAL3_TRAIN_STEPS=50000
#   EVAL3_EFF_BATCH=64        total gradient batch (per-GPU batch = EFF/NGPU)
#   EVAL3_NUM_GPUS=auto       auto-detected; set to 4 for 4×H200, 8 for 8×H100
#   EVAL3_SAVE_FREQ=5000
#   EVAL3_POLICY_DEVICE=cuda
#   EVAL3_TRAIN_OUT=outputs/train/test_eval3_v1
#   EVAL3_JOB_NAME=test_eval3_v1
#   EVAL3_WANDB=1
#   EVAL3_RESUME_FROM=lerobot/smolvla_base
#
# Quick smoke run (no GPU, no hardware):
#   EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=12 EVAL3_WANDB=0 \
#     EVAL3_TRAIN_OUT=/tmp/eval3_rag_smoke \
#     ./scripts/eval3_rag/run_train_rag.sh --log_freq=2
#
# Full run:
#   ./scripts/eval3_rag/run_train_rag.sh 2>&1 | tee outputs/train/logs/test_eval3_v1.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$ROOT"

# ── keep alive after VSCode / SSH disconnect ──────────────────────────────────
# Launch detached in tmux so training survives a dropped connection.
# Output streams to both the tmux pane and a persistent log file.
# Set EVAL3_NO_TMUX=1 to skip (CI, or when already inside tmux).
if [[ -z "${TMUX:-}" && "${EVAL3_NO_TMUX:-0}" != "1" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    SESSION="eval3_train_$(date +%Y%m%d_%H%M%S)"
    LOG="$ROOT/outputs/logs/${SESSION}.log"
    mkdir -p "$ROOT/outputs/logs"
    _QSELF="$(printf '%q' "$_SELF")"
    _QLOG="$(printf '%q' "$LOG")"
    _QARGS=""
    for _a in "$@"; do _QARGS="${_QARGS} $(printf '%q' "$_a")"; done
    tmux new-session -d -s "$SESSION" \
      "EVAL3_NO_TMUX=1 bash ${_QSELF}${_QARGS} 2>&1 | tee ${_QLOG}"
    echo ">> Training launched in tmux session '$SESSION'"
    echo "   Log        : $LOG"
    echo "   Watch live : tail -f $LOG"
    echo "   Attach     : tmux attach -t $SESSION"
    exit 0
  else
    echo ">> WARNING: tmux not found — install: sudo apt-get install -y tmux"
    echo "   Running directly (job will die if VSCode/SSH disconnects)"
  fi
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ── dataset corpus ─────────────────────────────────────────────────────────
# Base: 9 real v4 datasets (TOY celebs, per-position).
# These alone have the trajectory-identity bug: each dataset has one celeb at
# one fixed slot, so a BC model can ignore camera2 and go to the slot it was
# trained on.  Balanced synth datasets below (Fix A) counter this.
PRIMARY="RobotLearningVLA/dataset_v4_taylor_left"
TOY_EXTRAS="RobotLearningVLA/dataset_v4_taylor_middle,RobotLearningVLA/dataset_v4_taylor_right,RobotLearningVLA/dataset_v4_yann_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_yann_right,RobotLearningVLA/dataset_v4_barack_left,RobotLearningVLA/dataset_v4_barack_middle,RobotLearningVLA/dataset_v4_barack_right"

# Fix A — balanced synthetic datasets for TOY celebs.
# Each dataset has the target celeb at ALL 3 slots, so the action VARIES within
# one dataset and BC cannot shortcut on position alone.
# Set EVAL3_RAG_BALANCED_REPOS=1 after generating them:
#   python tools/eval3_synth_dataset_gen.py --balanced --push-to-hub \
#       --hub-org yukk1 --output-version v4
# This produces yukk1/dataset_v4_synth_<celeb>_balanced_1 on HF.
HF_ORG="${EVAL3_RAG_HF_ORG:-yukk1}"
_BALANCED_REPOS="${HF_ORG}/dataset_v4_synth_taylor_swift_balanced_1,${HF_ORG}/dataset_v4_synth_barack_obama_balanced_1,${HF_ORG}/dataset_v4_synth_yann_lecun_balanced_1"

# OOD synthetic datasets — generated by gen_rag_synth.sh and pushed to HF.
# Only included when EVAL3_RAG_OOD_REPOS=1.
# Name pattern: <HF_ORG>/dataset_rag_synth_<celeb>_<position>_2
# where suffix "rag1" → stored as dataset_v3_synth_rag1_<celeb>_<pos>_2
_OOD_CELEBS="cristiano_ronaldo lionel_messi rihanna elon_musk leonardo_dicaprio tom_cruise dwayne_johnson robert_downey_jr scarlett_johansson jennifer_lawrence marc_pollefeys"
_OOD_REPOS=""
for _c in $_OOD_CELEBS; do
  for _p in left middle right; do
    _OOD_REPOS="${_OOD_REPOS},${HF_ORG}/dataset_v3_synth_rag1_${_c}_${_p}_2"
  done
done
_OOD_REPOS="${_OOD_REPOS#,}"   # strip leading comma

EXTRAS="$TOY_EXTRAS"
if [[ "${EVAL3_RAG_BALANCED_REPOS:-0}" == "1" ]]; then
  EXTRAS="${EXTRAS},${_BALANCED_REPOS}"
  echo ">> Including balanced TOY synth datasets (Fix A — breaks trajectory-identity shortcut)"
fi
if [[ "${EVAL3_RAG_OOD_REPOS:-1}" == "1" ]]; then
  EXTRAS="${EXTRAS},${_OOD_REPOS}"
  echo ">> Including OOD synthetic celebrity datasets from $HF_ORG (needed for runs 7-9)"
fi

export EVAL3_DATASET_REPO="${EVAL3_DATASET_REPO:-$PRIMARY}"
export EVAL3_EXTRA_REPOS="${EVAL3_EXTRA_REPOS:-$EXTRAS}"

# ── RAG face DB ────────────────────────────────────────────────────────────
export EVAL3_RAG_REFDB="${EVAL3_RAG_REFDB:-datasets/celeb_refdb}"
export EVAL3_RAG_HW="${EVAL3_RAG_HW:-480,640}"

if [[ ! -d "$EVAL3_RAG_REFDB" ]]; then
  echo "ERROR: EVAL3_RAG_REFDB=$EVAL3_RAG_REFDB does not exist." >&2
  echo "       Run: python scripts/eval3_rag/build_celeb_refdb.py" >&2
  exit 1
fi

# ── auto-populate celeb_refdb with TOY celebrity portraits ───────────────────
# Without portrait diversity (≥10 images per TOY celeb), Eval3RagWrapper shows
# the same reference photo thousands of times during training → the model
# overfits to those specific photos and fails on held-out reference prints
# (rounds 1-3 / 4-6).  Fetch HQ portraits from CelebA-HQ (1024px) first;
# fall back to Wikipedia for academics not in the entertainment dataset
# (yann_lecun is not in CelebA-HQ).
# Skip with: EVAL3_RAG_SKIP_TOY_PORTRAITS=1
_TOY_MIN_PHOTOS=10
if [[ "${EVAL3_RAG_SKIP_TOY_PORTRAITS:-0}" != "1" ]]; then
  echo ">> Checking celeb_refdb portrait counts for TOY celebs (min=${_TOY_MIN_PHOTOS}) ..."
  _TOY_NEED_CSV=""
  for _c in taylor_swift barack_obama yann_lecun; do
    _n=0
    if [[ -d "$EVAL3_RAG_REFDB/$_c" ]]; then
      _n=$(find "$EVAL3_RAG_REFDB/$_c" -maxdepth 1 \
             \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) \
             2>/dev/null | wc -l | tr -d ' ')
    fi
    if [[ "$_n" -lt "$_TOY_MIN_PHOTOS" ]]; then
      echo "   $_c: $_n / ${_TOY_MIN_PHOTOS} — will fetch"
      _TOY_NEED_CSV="${_TOY_NEED_CSV:+${_TOY_NEED_CSV},}${_c}"
    else
      echo "   $_c: $_n portraits ✓"
    fi
  done

  if [[ -n "$_TOY_NEED_CSV" ]]; then
    echo ""
    echo "   Step 1/2: CelebA-HQ HF stream (entertainment celebrities, 1024px) ..."
    python scripts/eval3_rag/build_celeb_refdb.py \
      --db-root "$EVAL3_RAG_REFDB" \
      --from-hf "leondgarse/celeba_hq_face_recognition" \
      --slugs   "$_TOY_NEED_CSV" \
      --hf-max-per-celeb "$_TOY_MIN_PHOTOS" || true

    # Wikipedia fallback for academics not in CelebA-HQ (e.g. yann_lecun)
    _TOY_WIKI_SLUGS=()
    IFS=',' read -ra _TOY_NEED_ARR <<< "$_TOY_NEED_CSV"
    for _c in "${_TOY_NEED_ARR[@]}"; do
      _n=0
      if [[ -d "$EVAL3_RAG_REFDB/$_c" ]]; then
        _n=$(find "$EVAL3_RAG_REFDB/$_c" -maxdepth 1 \
               \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) \
               2>/dev/null | wc -l | tr -d ' ')
      fi
      [[ "$_n" -lt 1 ]] && _TOY_WIKI_SLUGS+=("$_c")
    done
    if [[ ${#_TOY_WIKI_SLUGS[@]} -gt 0 ]]; then
      echo "   Step 2/2: Wikipedia fallback for: ${_TOY_WIKI_SLUGS[*]}"
      python scripts/eval3_rag/download_refdb.py \
        --db-root "$EVAL3_RAG_REFDB" \
        --slugs "${_TOY_WIKI_SLUGS[@]}" || true
    fi
    echo ""
  fi
fi

# ── GPU count (needed before batch calc) ──────────────────────────────────
# Must be determined here so per-GPU batch can be derived from effective batch.
_NGPU_DEFAULT="$(python3 -c 'import torch; print(max(1, torch.cuda.device_count()))' 2>/dev/null || echo 1)"
NGPU="${EVAL3_NUM_GPUS:-${_NGPU_DEFAULT}}"

# ── training knobs ─────────────────────────────────────────────────────────
# EFF_BATCH is the total gradient batch across all GPUs (DDP averages gradients
# before each optimizer step, so effective_batch = per_gpu_batch × ngpu).
# Keeping EFF_BATCH=64 at any NGPU produces IDENTICAL training dynamics to the
# validated single-GPU run (v4slots-expert-50k, batch=64, LR=8e-4) — 4× H200
# is just 4× faster wall-clock, not a different experiment.
# Override EFF_BATCH to scale up deliberately; override BATCH to fix per-GPU batch.
EFF_BATCH="${EVAL3_EFF_BATCH:-64}"
BATCH="${EVAL3_BATCH:-$(( EFF_BATCH / NGPU ))}"
[[ "$BATCH" -lt 1 ]] && BATCH=1
STEPS="${EVAL3_TRAIN_STEPS:-50000}"
DEVICE="${EVAL3_POLICY_DEVICE:-cuda}"
OUT="${EVAL3_TRAIN_OUT:-outputs/train/test_eval3_v1}"
JOB="${EVAL3_JOB_NAME:-test_eval3_v1}"
SAVE_FREQ="${EVAL3_SAVE_FREQ:-5000}"
POLICY_PATH="${EVAL3_RESUME_FROM:-lerobot/smolvla_base}"

# LR: linear-scaled from EFFECTIVE batch (not per-GPU), capped at 8e-4.
# Using EFF_BATCH here is critical — the gradient step magnitude is determined
# by the effective batch regardless of how many GPUs contribute to it.
# At EFF_BATCH=64: LR = 1e-4 × (64/8) = 8e-4, same as the validated run.
_BASE_BATCH=8
_BASE_LR="1e-4"
_LR_CAP="8e-4"
PEAK_LR="${EVAL3_PEAK_LR:-$(python3 -c "print(min(${_BASE_LR} * (${EFF_BATCH} / ${_BASE_BATCH}), ${_LR_CAP}))")}"
WARMUP="${EVAL3_WARMUP_STEPS:-500}"
DECAY="${EVAL3_DECAY_STEPS:-$(( STEPS - WARMUP ))}"
[[ "$DECAY" -lt 1 ]] && DECAY=1
DECAY_LR="${EVAL3_DECAY_LR:-1e-6}"

if [[ "${EVAL3_WANDB:-1}" == "1" || "${EVAL3_WANDB:-1}" == "true" ]]; then
  WANDB_ENABLE="true"
else
  WANDB_ENABLE="false"
fi
WANDB_PROJECT="${EVAL3_WANDB_PROJECT:-test_eval3_v1}"

# ── dataset prep env vars (passed to eval3_concat_patch) ─────────────────
# camera2 = reference face (not frame-0), so EVAL3_SLOT_FRAME0 must stay off.
export EVAL3_MAX_FRAMES_PER_EP="${EVAL3_MAX_FRAMES_PER_EP:-600}"
# EVAL3_TASK_AUG: keep enabled so TaskAugmenter normalises prompt phrasing.
# EVAL3_TASK_AUG_CANONICAL_P=1.0: use only the two canonical prompt forms
# ("Place the coke on X" / "Place the coke on the X").  Eval3RagWrapper then
# overwrites the task with EVAL3_CANONICAL_TASK (stripping the person name)
# once the reference image is injected — so the model NEVER sees a person
# name during training and must rely on camera2 for identity.
export EVAL3_TASK_AUG="${EVAL3_TASK_AUG:-1}"
export EVAL3_TASK_AUG_CANONICAL_P="${EVAL3_TASK_AUG_CANONICAL_P:-1.0}"
# Task string the model sees at BOTH training and inference (must match).
export EVAL3_CANONICAL_TASK="${EVAL3_CANONICAL_TASK:-Place the coke on the person in the reference photo}"
# Camera2 printed-photo augmentation: simulates the appearance of the
# reference portrait after printing, placing on a table, and photographing.
# Teaches the model to match the same person despite colour shift / blur /
# monochrome print artefacts.  Disable with EVAL3_CAM2_AUG=0 for ablation.
export EVAL3_CAM2_AUG="${EVAL3_CAM2_AUG:-1}"
export EVAL3_CAM2_AUG_P="${EVAL3_CAM2_AUG_P:-0.7}"
export EVAL3_GRIPPER_REPAIR="${EVAL3_GRIPPER_REPAIR:-0}"
export EVAL3_ACTION_SMOOTH_WINDOW="${EVAL3_ACTION_SMOOTH_WINDOW:-3}"
export EVAL3_ACTION_SMOOTH_GRIPPER="${EVAL3_ACTION_SMOOTH_GRIPPER:-0}"
export EVAL3_TRUNCATE_AT_PLACEMENT="${EVAL3_TRUNCATE_AT_PLACEMENT:-1}"
export EVAL3_BG_REPLACE="${EVAL3_BG_REPLACE:-0}"
export EVAL3_PRINT_SHUFFLE="${EVAL3_PRINT_SHUFFLE:-0}"
# Disable v16 frame-0 slot mode — we use reference face for camera2 instead.
unset EVAL3_SLOT_FRAME0
unset EVAL3_SLOT_LOSS_WEIGHT

# ── camera rename map ─────────────────────────────────────────────────────
# camera1 = live scene, camera2 = retrieved reference face, camera3 = empty pad
RENAMES='{"observation.images.front":"observation.images.camera1","observation.images.front_refface":"observation.images.camera2"}'

# ── image augmentation ────────────────────────────────────────────────────
# Identical to the v10 Fix-C / v4slots-expert TFS: face-preserving mild jitter
# with RandomErasing at low strength to prevent texture overfitting.
TFS_JSON='{"brightness":{"weight":2.0,"type":"ColorJitter","kwargs":{"brightness":[0.7,1.3]}},"contrast":{"weight":2.0,"type":"ColorJitter","kwargs":{"contrast":[0.7,1.3]}},"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.5,1.5]}},"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.05,0.05]}},"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},"affine":{"weight":1.0,"type":"RandomAffine","kwargs":{"degrees":[-3.0,3.0],"translate":[0.03,0.03]}},"perspective":{"weight":0.75,"type":"RandomPerspective","kwargs":{"distortion_scale":0.1,"p":0.3}},"resized_crop":{"weight":1.0,"type":"RandomResizedCrop","kwargs":{"size":[480,640],"scale":[0.85,1.0],"ratio":[0.95,1.05]}},"gaussian_blur":{"weight":0.5,"type":"GaussianBlur","kwargs":{"kernel_size":[5,9],"sigma":[0.3,1.0]}},"erase":{"weight":0.25,"type":"RandomErasing","kwargs":{"p":0.15,"scale":[0.02,0.05]}}}'
export EVAL3_TFS_JSON="${EVAL3_TFS_JSON:-$TFS_JSON}"
export EVAL3_NUM_WORKERS="${EVAL3_NUM_WORKERS:-44}"

# ── multi-GPU launcher ────────────────────────────────────────────────────
# torchrun spawns one process per GPU; each applies all monkey-patches
# independently.  NGPU was detected above alongside batch calculation.
MASTER_PORT="${EVAL3_MASTER_PORT:-29500}"

if [[ "$NGPU" -gt 1 ]]; then
  LAUNCHER=(torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT")
  echo ">> Multi-GPU: torchrun with $NGPU GPUs (port $MASTER_PORT)"
else
  LAUNCHER=(python)
fi

echo ">> Eval3 RAG SmolVLA train"
echo "   refdb        : $EVAL3_RAG_REFDB  ($(ls "$EVAL3_RAG_REFDB" 2>/dev/null | wc -l) celebrity dirs)"
echo "   policy       : $POLICY_PATH"
echo "   device       : $DEVICE   gpus=$NGPU"
echo "   steps        : $STEPS / batch=$BATCH/gpu × $NGPU gpus = $(( BATCH * NGPU )) effective  save_freq=$SAVE_FREQ"
echo "   peak_lr      : $PEAK_LR  (from eff_batch=$EFF_BATCH, same dynamics as validated 1-GPU run)"
echo "   warmup/decay : $WARMUP / $DECAY  decay_lr=$DECAY_LR"
echo "   freeze_vision: true  train_expert_only: true  use_amp: true  compile: true"
echo "   out          : $OUT"
echo "   hf_repo      : ${EVAL3_HF_REPO_ID:-${HF_ORG}/${JOB}-smolvla-${STEPS}}"
echo "   rename       : $RENAMES"

HF_REPO_ID="${EVAL3_HF_REPO_ID:-${HF_ORG}/${JOB}-smolvla-${STEPS}}"

exec "${LAUNCHER[@]}" scripts/eval3_rag/train_rag.py \
  --policy.path="$POLICY_PATH" \
  --policy.push_to_hub=true \
  --policy.repo_id="$HF_REPO_ID" \
  --policy.compile_model=true \
  --policy.device="$DEVICE" \
  --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.use_amp=true \
  --policy.pad_language_to=longest \
  --rename_map="$RENAMES" \
  --dataset.repo_id="$EVAL3_DATASET_REPO" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=4 \
  --dataset.image_transforms.tfs="$EVAL3_TFS_JSON" \
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
  --scheduler.num_warmup_steps="$WARMUP" \
  --scheduler.num_decay_steps="$DECAY" \
  --job_name="$JOB" \
  --output_dir="$OUT" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --batch_size="$BATCH" \
  --num_workers="$EVAL3_NUM_WORKERS" \
  "$@"

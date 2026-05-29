#!/usr/bin/env bash
# Kick off the v17 "v4-anchored big-zoo" full training run defined in
# docs/eval3/v17_playbook.md §3.8.
#
# Waits for pins30q5 to be fully built (90 datasets) before launching, then
# starts a backgrounded 50k-step training with the val watcher running
# alongside against the held-out eval set (§3.7).
#
# Usage:
#   ./scripts/start_v17_v4anchored_training.sh                 # standard
#   ./scripts/start_v17_v4anchored_training.sh --no-wait       # don't wait, fail fast if pins30q5 incomplete
#   ./scripts/start_v17_v4anchored_training.sh --dry-run       # print the command, don't run it

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

WAIT=1
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-wait) WAIT=0 ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

# ---- Gate: pins30q5 readiness ---------------------------------------------
# Override if some pins30q5 datasets had to be dropped (e.g. video/parquet
# frame-count mismatch detected by audit). Default 90 = full slate.
need_dirs="${V17_PINS30Q5_NEED_DIRS:-90}"
while true; do
  cur=$(ls -d "$ROOT"/datasets/dataset_v5_synth_pins30q5_*_full 2>/dev/null | wc -l)
  echo "pins30q5 dirs ready: $cur / $need_dirs"
  if [[ "$cur" -ge "$need_dirs" ]]; then break; fi
  if [[ "$WAIT" -eq 0 ]]; then
    echo "pins30q5 not ready and --no-wait set; aborting." >&2
    exit 1
  fi
  sleep 30
done

# ---- Validate held-out eval set + prompts JSON ----------------------------
PROMPTS_JSON="/tmp/holdout_prompts.json"
if [[ ! -f "$PROMPTS_JSON" ]]; then
  echo "rebuilding $PROMPTS_JSON from pool JSONs..."
  python -c "
import json
from pathlib import Path
algvr = json.loads(Path('datasets/algvr-conference.json').read_text())
pins  = json.loads(Path('datasets/pins-face-recognition-top30-quality.json').read_text())
celebs = {c['slug']: c['name'] for c in algvr['celebrities'] + pins['celebrities']}
print(json.dumps({s: f'Place the coke on {n}' for s, n in celebs.items()}))
" > "$PROMPTS_JSON"
fi

HOLDOUT_REPOS=$(ls -d datasets/dataset_v5_synth_holdout_*_full 2>/dev/null \
                  | sed 's|datasets/|RobotLearningVLA/|' | paste -sd, -)
if [[ -z "$HOLDOUT_REPOS" ]]; then
  echo "ERROR: no datasets/dataset_v5_synth_holdout_*_full found." >&2
  echo "       Build them: python tools/eval3_build_holdout_eval_set.py" >&2
  exit 2
fi
PROMPTS=$(cat "$PROMPTS_JSON")

# LeRobot train.cfg.validate() refuses to overwrite an existing output dir
# (resume=false). Do NOT pre-create EVAL3_TRAIN_OUT — let lerobot_train create
# it itself.
mkdir -p outputs/train/logs
if [[ -d outputs/v17 ]]; then
  if [[ -z "$(ls -A outputs/v17 2>/dev/null)" ]]; then
    rmdir outputs/v17  # empty leftover (e.g. from a prior pre-mkdir bug)
  else
    echo "ERROR: outputs/v17 already exists and is non-empty." >&2
    echo "       LeRobot refuses to overwrite. Move or delete it manually." >&2
    exit 3
  fi
fi

CMD=(
  env
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  EVAL3_TRAIN_STEPS=50000
  EVAL3_BATCH=128
  EVAL3_SAVE_FREQ=1000
  EVAL3_WANDB=1
  EVAL3_TRAIN_OUT=./outputs/v17
  EVAL3_JOB_NAME=eval3_v17_v4anchored
  EVAL3_WANDB_PROJECT=eval3-v17-camdrop
  EVAL3_V17_NO_SYNTH=1
  EVAL3_V17_V4_REPLICAS=15
  EVAL3_V17_INCLUDE_ALGVR=1
  EVAL3_V17_INCLUDE_PINS30Q5=1
  EVAL3_STATE_NOISE_SIGMA_MIN=0.05
  EVAL3_VAL_WATCH=1
  EVAL3_VAL_DEVICE=cuda
  EVAL3_VAL_WANDB=1
  EVAL3_VAL_REPOS="$HOLDOUT_REPOS"
  EVAL3_VAL_LOCAL_REPOS="$HOLDOUT_REPOS"
  EVAL3_VAL_PROMPTS="$PROMPTS"
  EVAL3_VAL_EPISODES_PER_REPO=3
  EVAL3_VAL_FRAMES_PER_EPISODE=30
  EVAL3_VAL_POLL_SEC=60
  EVAL3_VAL_IDLE_SEC=1800
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh
  --log_freq=100
)

LOG=outputs/train/logs/eval3_v17_v4anchored.log
echo ">> Command:"
printf '   %q ' "${CMD[@]}"
echo
echo ">> Log: $LOG"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "(--dry-run) not starting"
  exit 0
fi

# Detach into its own session so the training process survives this shell.
nohup setsid "${CMD[@]}" > "$LOG" 2>&1 &
TRAIN_PID=$!
echo ">> Training PID: $TRAIN_PID  (session-detached via setsid)"
echo ">> Tail with: tail -f $LOG"

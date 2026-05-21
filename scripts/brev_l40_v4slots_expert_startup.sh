#!/usr/bin/env bash
# Brev L40 startup: clone, install, start expert-only train (v4 slots).
set -euo pipefail
REPO_URL="${EVAL3_REPO_URL:-https://github.com/shyngys-aitkazinov/robot-learning-vla.git}"
BRANCH="${EVAL3_REPO_BRANCH:-main}"
USER_NAME="${BREV_USER:-ubuntu}"
HOME_DIR="/home/${USER_NAME}"
WORKDIR="${HOME_DIR}/robot-learning-vla"
LOG="${HOME_DIR}/brev_v4slots_expert.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== v4slots EXPERT $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
run_as_user() { sudo -u "$USER_NAME" bash -lc "$*"; }
if ! id "$USER_NAME" &>/dev/null; then USER_NAME="$(ls /home | head -1)"; HOME_DIR="/home/${USER_NAME}"; WORKDIR="${HOME_DIR}/robot-learning-vla"; fi
if [[ ! -d "${WORKDIR}/.git" ]]; then run_as_user "git clone --branch '${BRANCH}' '${REPO_URL}' '${WORKDIR}'"; else run_as_user "cd '${WORKDIR}' && git fetch origin && git checkout '${BRANCH}' && git pull origin '${BRANCH}'"; fi
run_as_user "cd '${WORKDIR}' && EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh"
if [[ -n "${HF_TOKEN:-}" ]]; then run_as_user "cd '${WORKDIR}' && source .venv/bin/activate && huggingface-cli login --token \"\${HF_TOKEN}\" --add-to-git-credential"; fi
TRAIN_CMD='if [[ -x ./scripts/run_eval3_smolvla_v4slots_train.sh ]]; then ./scripts/run_eval3_smolvla_v4slots_train.sh expert; else
  export EVAL3_DATASET_REPO=RobotLearningVLA/dataset_v4_taylor_left
  export EVAL3_EXTRA_REPOS=RobotLearningVLA/dataset_v4_taylor_middle,RobotLearningVLA/dataset_v4_taylor_right,RobotLearningVLA/dataset_v4_yann_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_yann_right,RobotLearningVLA/dataset_v4_barack_left,RobotLearningVLA/dataset_v4_barack_middle,RobotLearningVLA/dataset_v4_barack_right
  export EVAL3_POLICY_DEVICE=cuda EVAL3_BATCH=16 EVAL3_TRAIN_STEPS=50000 EVAL3_TASK_AUG_CANONICAL_P=1.0 EVAL3_GRIPPER_REPAIR=0 EVAL3_BG_REPLACE=0 EVAL3_PRINT_SHUFFLE=0
  export EVAL3_FREEZE_VISION=1 EVAL3_TRAIN_EXPERT_ONLY=1 EVAL3_USE_AMP=1 EVAL3_PEAK_LR=2e-4
  export EVAL3_JOB_NAME=eval3-vla-v6-smolvla-fresh-v4slots-expert-50k EVAL3_TRAIN_OUT=outputs/train/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k
  ./scripts/run_eval3_smolvla_aug_train.sh; fi'
run_as_user "cd '${WORKDIR}' && EVAL3_TRAIN_CMD=\"${TRAIN_CMD}\" EVAL3_JOB_NAME=eval3-v4slots-expert ./scripts/run_eval3_train_daemon.sh start"
echo "=== EXPERT train daemon started — tail ${WORKDIR}/logs/eval3-v4slots-expert.log ==="

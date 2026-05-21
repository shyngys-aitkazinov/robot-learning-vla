#!/usr/bin/env bash
# Shared Brev L40 train startup (upload to ~/ on instance or inline via brev create).
set -euo pipefail

REPO_URL="${EVAL3_REPO_URL:-https://github.com/shyngys-aitkazinov/robot-learning-vla.git}"
BRANCH="${EVAL3_REPO_BRANCH:-main}"
ROLE="${EVAL3_V4SLOTS_ROLE:-full}"
USER_NAME="${BREV_USER:-ubuntu}"
HOME_DIR="/home/${USER_NAME}"
WORKDIR="${HOME_DIR}/robot-learning-vla"
LOG="${HOME_DIR}/brev_v4slots_${ROLE}.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== v4slots train startup role=${ROLE} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

run_as_user() {
  sudo -u "$USER_NAME" bash -lc "$*"
}

if ! id "$USER_NAME" &>/dev/null; then
  USER_NAME="$(ls /home | head -1)"
  HOME_DIR="/home/${USER_NAME}"
  WORKDIR="${HOME_DIR}/robot-learning-vla"
fi

if [[ ! -d "${WORKDIR}/.git" ]]; then
  run_as_user "git clone --branch '${BRANCH}' '${REPO_URL}' '${WORKDIR}'"
else
  run_as_user "cd '${WORKDIR}' && git fetch origin && git checkout '${BRANCH}' && git pull origin '${BRANCH}'"
fi

run_as_user "cd '${WORKDIR}' && EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh"
run_as_user "chmod +x '${WORKDIR}/scripts/run_eval3_smolvla_v4slots_train.sh' '${WORKDIR}/scripts/run_eval3_train_daemon.sh' 2>/dev/null || true"

# Hugging Face login if token present in instance env (Brev secret / user export).
if [[ -n "${HF_TOKEN:-}" ]]; then
  run_as_user "cd '${WORKDIR}' && source .venv/bin/activate && huggingface-cli login --token \"\${HF_TOKEN}\" --add-to-git-credential"
fi

JOB="eval3-v4slots-${ROLE}"
run_as_user "cd '${WORKDIR}' && EVAL3_TRAIN_CMD='./scripts/run_eval3_smolvla_v4slots_train.sh ${ROLE}' EVAL3_JOB_NAME='${JOB}' ./scripts/run_eval3_train_daemon.sh start"

echo "=== started ${ROLE} — log: ${WORKDIR}/logs/${JOB}.log ==="

#!/usr/bin/env bash
# Brev H100 startup script — clone repo + install Eval3 SmolVLA deps.
# Logs: ~/brev_h100_bootstrap.log

set -euo pipefail

REPO_URL="${EVAL3_REPO_URL:-https://github.com/shyngys-aitkazinov/robot-learning-vla.git}"
BRANCH="${EVAL3_REPO_BRANCH:-main}"
USER_NAME="${BREV_USER:-ubuntu}"
HOME_DIR="/home/${USER_NAME}"
WORKDIR="${HOME_DIR}/robot-learning-vla"
LOG="${HOME_DIR}/brev_h100_bootstrap.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== brev H100 bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "repo=${REPO_URL} branch=${BRANCH} workdir=${WORKDIR}"

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

echo "=== bootstrap complete ==="
echo "Next: cd ${WORKDIR} && source .venv/bin/activate"
echo "Train: EVAL3_TRAIN_CMD=./scripts/run_eval3_smolvla_h100_expert.sh EVAL3_JOB_NAME=eval3-h100-expert ./scripts/run_eval3_train_daemon.sh start"

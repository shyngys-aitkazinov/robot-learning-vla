#!/usr/bin/env bash
# Brev L40 startup — clone repo, install SmolVLA deps, print train commands.
# Logs: ~/brev_l40_v4slots_bootstrap.log

set -euo pipefail

REPO_URL="${EVAL3_REPO_URL:-https://github.com/shyngys-aitkazinov/robot-learning-vla.git}"
BRANCH="${EVAL3_REPO_BRANCH:-main}"
USER_NAME="${BREV_USER:-ubuntu}"
HOME_DIR="/home/${USER_NAME}"
WORKDIR="${HOME_DIR}/robot-learning-vla"
LOG="${HOME_DIR}/brev_l40_v4slots_bootstrap.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== brev L40 v4slots bootstrap $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

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
chmod +x "${WORKDIR}/scripts/run_eval3_smolvla_v4slots_train.sh" 2>/dev/null || true

echo "=== bootstrap complete ==="
echo "Full FT (this GPU):"
echo "  cd ${WORKDIR} && source .venv/bin/activate"
echo "  EVAL3_TRAIN_CMD='./scripts/run_eval3_smolvla_v4slots_train.sh full' \\"
echo "  EVAL3_JOB_NAME=eval3-v4slots-full ./scripts/run_eval3_train_daemon.sh start"
echo "Expert (other GPU):"
echo "  EVAL3_TRAIN_CMD='./scripts/run_eval3_smolvla_v4slots_train.sh expert' \\"
echo "  EVAL3_JOB_NAME=eval3-v4slots-expert ./scripts/run_eval3_train_daemon.sh start"

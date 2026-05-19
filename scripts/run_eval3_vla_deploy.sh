#!/usr/bin/env bash
# Wrapper for eval3_vla_deploy.py (SmolVLA closed-loop on SO-101).
#
# Examples:
#   ./scripts/run_eval3_vla_deploy.sh \
#     --robot.type=so101_follower \
#     --robot.port=/dev/tty.usbmodemXXXX \
#     --robot.id=my_awesome_follower_arm \
#     --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
#     --policy.path=RobotLearningVLA/eval3-smolvla-v10-balanced-new66-10k \
#     --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \
#     --rename_map='{"observation.images.front":"observation.images.camera1"}' \
#     --policy.empty_cameras=2 \
#     --task="Place the coke on the Taylor Swift" \
#     --dry_run=true
#
#   # Three rollouts with home return between each:
#   ./scripts/run_eval3_vla_deploy.sh ... \
#     --task="Place the coke on Barack Obama" \
#     --n_rollouts=3 \
#     --episode_time_s=20

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python scripts/eval3_vla_deploy.py "$@"

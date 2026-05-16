#!/usr/bin/env python3
"""Preflight checks for Eval 3 v5 new-data training.

This is intentionally read-only: it verifies the Hugging Face datasets and the
runtime env before the expensive SmolVLA training command starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass

from lerobot.datasets.lerobot_dataset import LeRobotDataset


PRIMARY_REPO = "RobotLearningVLA/dataset_v2_barack_obama_left_1"
EXTRA_REPOS = [
    "RobotLearningVLA/dataset_v2_barack_obama_middle_1",
    "RobotLearningVLA/dataset_v2_barack_obama_right_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_left_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_middle_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_middle_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_right_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_right_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_left_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_left_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_middle_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_middle_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_right_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_right_1",
]

EXPECTED_COLUMNS = {
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
EXPECTED_GRID = {
    ("barack_obama", "left"),
    ("barack_obama", "middle"),
    ("barack_obama", "right"),
    ("yann_lecun", "left"),
    ("yann_lecun", "middle"),
    ("yann_lecun", "right"),
    ("taylor_swift", "left"),
    ("taylor_swift", "middle"),
    ("taylor_swift", "right"),
}
EXPECTED_DEFAULT_RAW_FRAMES = 44_222
EXPECTED_DEFAULT_RAW_EPISODES = 66
EXPECTED_DEFAULT_VIRTUAL_FRAMES = 59_933
EXPECTED_DEFAULT_VIRTUAL_EPISODES = 91
CELEBRITY_CANONICAL = {
    "taylor_swift": "Taylor Swift",
    "yann_lecun": "Yann LeCun",
    "barack_obama": "Barack Obama",
}


@dataclass(frozen=True)
class RepoAudit:
    repo: str
    identity: str
    position: str
    codebase_version: str
    fps: int
    robot_type: str
    episodes: int
    frames: int
    task: str
    canonical_task: str
    columns: list[str]


def _split_repos(raw: str) -> list[str]:
    return [repo.strip() for repo in raw.split(",") if repo.strip()]


def _identity_position(repo: str) -> tuple[str, str]:
    base = repo.rsplit("/", 1)[-1].lower()
    for identity in CELEBRITY_CANONICAL:
        marker = f"{identity}_"
        if marker in base:
            tail = base.split(marker, 1)[1]
            position = tail.split("_", 1)[0]
            return identity, position
    return "unknown", "unknown"


def _task_string(ds: LeRobotDataset) -> str:
    tasks = ds.meta.tasks
    if hasattr(tasks, "index") and len(tasks.index):
        return str(tasks.index[0])
    if isinstance(tasks, dict) and tasks:
        return str(next(iter(tasks)))
    return ""


def _canonical_task(identity: str, task: str) -> str:
    name = CELEBRITY_CANONICAL.get(identity)
    if name and name in task:
        return f"Place the coke on {name}"
    return task


def _audit_repo(repo: str) -> RepoAudit:
    ds = LeRobotDataset(repo)
    identity, position = _identity_position(repo)
    info = ds.meta.info
    task = _task_string(ds)
    return RepoAudit(
        repo=repo,
        identity=identity,
        position=position,
        codebase_version=str(info.get("codebase_version")),
        fps=int(info.get("fps")),
        robot_type=str(info.get("robot_type")),
        episodes=int(info.get("total_episodes")),
        frames=int(info.get("total_frames")),
        task=task,
        canonical_task=_canonical_task(identity, task),
        columns=list(ds.hf_dataset.column_names),
    )


def _check_env(errors: list[str]) -> None:
    expected = {
        "EVAL3_MAX_FRAMES_PER_EP": "0",
        "EVAL3_TASK_AUG": "1",
        "EVAL3_TASK_AUG_CANONICAL_P": "1.0",
        "EVAL3_BG_REPLACE": "0",
        "EVAL3_PRINT_SHUFFLE": "0",
    }
    for key, expected_value in expected.items():
        actual = os.environ.get(key)
        if actual != expected_value:
            errors.append(f"{key}={actual!r}; expected {expected_value!r}")

    for key in ("EVAL3_SWIFT_EPISODE_FILTER", "EVAL3_LECUN_EPISODE_FILTER", "EVAL3_OBAMA_EPISODE_FILTER"):
        if os.environ.get(key):
            errors.append(f"{key} must be unset for v5 new-data training")


def _default_mix(primary_repo: str, extra_repos: list[str]) -> bool:
    return primary_repo == PRIMARY_REPO and extra_repos == EXTRA_REPOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-repo", default=PRIMARY_REPO)
    parser.add_argument("--extra-repos", default=",".join(EXTRA_REPOS))
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--no-strict-default-counts",
        action="store_true",
        help="Do not fail if the default v5 mix no longer matches audited frame/episode counts.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    extra_repos = _split_repos(args.extra_repos)
    virtual_repos = [args.primary_repo, *extra_repos]
    unique_repos = list(dict.fromkeys(virtual_repos))
    errors: list[str] = []

    audits = [_audit_repo(repo) for repo in unique_repos]
    by_repo = {audit.repo: audit for audit in audits}

    observed_grid = {(audit.identity, audit.position) for audit in audits}
    missing_grid = EXPECTED_GRID - observed_grid
    extra_grid = observed_grid - EXPECTED_GRID
    if missing_grid:
        errors.append(f"missing identity/position repos: {sorted(missing_grid)}")
    if extra_grid:
        errors.append(f"unexpected identity/position repos: {sorted(extra_grid)}")

    for audit in audits:
        if audit.codebase_version != "v3.0":
            errors.append(f"{audit.repo}: codebase_version={audit.codebase_version}; expected v3.0")
        if audit.fps != 30:
            errors.append(f"{audit.repo}: fps={audit.fps}; expected 30")
        if audit.robot_type != "so_follower":
            errors.append(f"{audit.repo}: robot_type={audit.robot_type}; expected so_follower")
        missing_columns = EXPECTED_COLUMNS - set(audit.columns)
        if missing_columns:
            errors.append(f"{audit.repo}: missing columns {sorted(missing_columns)}")
        expected_name = CELEBRITY_CANONICAL.get(audit.identity)
        if expected_name and expected_name not in audit.task:
            errors.append(f"{audit.repo}: task={audit.task!r} does not contain {expected_name!r}")

    raw_frames = sum(audit.frames for audit in audits)
    raw_episodes = sum(audit.episodes for audit in audits)
    virtual_frames = sum(by_repo[repo].frames for repo in virtual_repos)
    virtual_episodes = sum(by_repo[repo].episodes for repo in virtual_repos)

    if args.check_env:
        _check_env(errors)

    if _default_mix(args.primary_repo, extra_repos) and not args.no_strict_default_counts:
        expected_pairs = {
            "raw_frames": (raw_frames, EXPECTED_DEFAULT_RAW_FRAMES),
            "raw_episodes": (raw_episodes, EXPECTED_DEFAULT_RAW_EPISODES),
            "virtual_frames": (virtual_frames, EXPECTED_DEFAULT_VIRTUAL_FRAMES),
            "virtual_episodes": (virtual_episodes, EXPECTED_DEFAULT_VIRTUAL_EPISODES),
        }
        for label, (actual, expected) in expected_pairs.items():
            if actual != expected:
                errors.append(f"{label}={actual}; expected {expected}")

    repo_repeats = Counter(virtual_repos)
    summary = {
        "ok": not errors,
        "errors": errors,
        "unique_repos": len(unique_repos),
        "virtual_repos": len(virtual_repos),
        "raw_frames": raw_frames,
        "raw_episodes": raw_episodes,
        "virtual_frames": virtual_frames,
        "virtual_episodes": virtual_episodes,
        "repo_repeats": dict(sorted(repo_repeats.items())),
        "repos": [asdict(audit) for audit in audits],
    }

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("Eval3 v5 dataset preflight")
        print(f"  unique repos     : {len(unique_repos)}")
        print(f"  virtual repos    : {len(virtual_repos)}")
        print(f"  raw frames/eps   : {raw_frames} / {raw_episodes}")
        print(f"  virtual frames/eps: {virtual_frames} / {virtual_episodes}")
        print("  repeats:")
        for repo, count in sorted(repo_repeats.items()):
            print(f"    {count}x {repo}")
        print("  canonical tasks:")
        for audit in audits:
            print(f"    {audit.repo}: {audit.canonical_task}")
        if errors:
            print("  failures:")
            for error in errors:
                print(f"    - {error}")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

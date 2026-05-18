#!/usr/bin/env python3
"""Audit Eval 3 datasets for gripper, timing, action/state lag, and jerk.

This focuses on the suspected data-side cause of poor real-robot grasping:
the policy may be trained on demonstrations where the gripper never opens wide
enough before approach or release.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


DEFAULT_REPOS = [
    "RobotLearningVLA/taylor_swift_1",
    "RobotLearningVLA/yann_lecun_1",
    "RobotLearningVLA/barack_obama_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_taylor_swift_middle_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_taylor_swift_right_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_yann_lecun_left_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_yann_lecun_middle_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_yann_lecun_right_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_barack_obama_left_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_barack_obama_middle_1_v6_truncated",
    "RobotLearningVLA/dataset_v2_barack_obama_right_1_v6_truncated",
]
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def episode_row(episodes: Any, episode: int) -> Any:
    if hasattr(episodes, "iloc"):
        return episodes.iloc[int(episode)]
    return episodes[int(episode)]


def column_to_array(ds: LeRobotDataset, key: str, start: int, end: int) -> np.ndarray:
    values = [ds.hf_dataset[key][i] for i in range(start, end)]
    if values and isinstance(values[0], torch.Tensor):
        return torch.stack([v.detach().float() for v in values]).cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def feature_names(ds: LeRobotDataset, key: str) -> list[str]:
    feature = ds.meta.features.get(key) or {}
    names = feature.get("names") if isinstance(feature, dict) else None
    return list(names or JOINT_NAMES)


def quantiles(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {k: 0.0 for k in ("min", "max", "mean", "std", "q50", "q90", "q95", "q99")}
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "q50": float(np.quantile(flat, 0.50)),
        "q90": float(np.quantile(flat, 0.90)),
        "q95": float(np.quantile(flat, 0.95)),
        "q99": float(np.quantile(flat, 0.99)),
    }


def audit_repo(
    repo_id: str,
    *,
    video_backend: str,
    release_open_threshold: float,
    q90_threshold: float,
    q99_threshold: float,
    lag_p95_threshold: float,
    jerk_p99_threshold: float,
    simulate_gripper_repair: bool,
    repair_open_target: float,
    repair_open_threshold: float,
) -> dict[str, Any]:
    ds = LeRobotDataset(repo_id, video_backend=video_backend)
    action_names = feature_names(ds, "action")
    state_names = feature_names(ds, "observation.state")
    gripper_idx = action_names.index("gripper") if "gripper" in action_names else len(action_names) - 1
    lift_idx = action_names.index("shoulder_lift") if "shoulder_lift" in action_names else 1
    fps = float(ds.fps)

    episode_reports = []
    all_gripper = []
    all_lag = []
    all_jerk = []
    durations = []
    repaired_frames = 0
    total_frames = 0
    for ep in range(int(ds.num_episodes)):
        row = episode_row(ds.meta.episodes, ep)
        start = int(scalar(row["dataset_from_index"]))
        end = int(scalar(row["dataset_to_index"]))
        if end <= start:
            continue
        action = column_to_array(ds, "action", start, end)
        state = column_to_array(ds, "observation.state", start, end)
        action = action.reshape(action.shape[0], -1)
        state = state.reshape(state.shape[0], -1)
        n = action.shape[0]
        total_frames += n
        if simulate_gripper_repair:
            before = action[:, gripper_idx].copy()
            open_mask = before >= float(repair_open_threshold)
            action[open_mask, gripper_idx] = np.maximum(
                action[open_mask, gripper_idx],
                float(repair_open_target),
            )
            repaired_frames += int(np.count_nonzero(np.abs(action[:, gripper_idx] - before) > 1e-6))
        duration_s = n / fps if fps else 0.0
        durations.append(duration_s)

        gripper = action[:, gripper_idx]
        lift = action[:, lift_idx]
        all_gripper.extend(gripper.tolist())

        shared_dim = min(action.shape[1], state.shape[1])
        lag = np.abs(action[:, :shared_dim] - state[:, :shared_dim])
        gripper_lag = lag[:, gripper_idx] if gripper_idx < shared_dim else np.zeros(n)
        all_lag.extend(gripper_lag.tolist())

        if n >= 3:
            jerk = np.diff(np.diff(action, axis=0), axis=0) * (fps**2)
            jerk_l2 = np.linalg.norm(jerk, axis=1)
            all_jerk.extend(jerk_l2.tolist())
        else:
            jerk_l2 = np.zeros(0)

        approach_end = max(1, int(n * 0.25))
        pre_approach_open = float(np.max(gripper[:approach_end]))
        release_window = gripper[max(0, int(n * 0.75)) :]
        release_q90 = float(np.quantile(release_window, 0.90)) if release_window.size else 0.0
        release_q99 = float(np.quantile(release_window, 0.99)) if release_window.size else 0.0
        open_frames = np.where(gripper >= release_open_threshold)[0]
        first_open_s = float(open_frames[0] / fps) if open_frames.size and fps else None
        final_1s_mean = float(np.mean(gripper[max(0, n - int(fps)) :])) if fps else float(np.mean(gripper))
        lift_extension = float(np.max(lift) - np.mean(lift[: max(1, int(fps))]))

        ep_flags = []
        if pre_approach_open < release_open_threshold:
            ep_flags.append("pre_approach_gripper_not_open")
        if release_q90 < q90_threshold:
            ep_flags.append("release_q90_low")
        if release_q99 < q99_threshold:
            ep_flags.append("release_q99_low")
        if first_open_s is None:
            ep_flags.append("never_crossed_open_threshold")
        if duration_s > 20.0:
            ep_flags.append("over_20s")

        episode_reports.append(
            {
                "episode": ep,
                "frames": n,
                "duration_s": duration_s,
                "pre_approach_gripper_max": pre_approach_open,
                "release_gripper_q90": release_q90,
                "release_gripper_q99": release_q99,
                "first_open_s": first_open_s,
                "final_1s_gripper_mean": final_1s_mean,
                "lift_extension": lift_extension,
                "gripper_action_state_lag_p95": float(np.quantile(gripper_lag, 0.95)) if gripper_lag.size else 0.0,
                "jerk_l2_p99": float(np.quantile(jerk_l2, 0.99)) if jerk_l2.size else 0.0,
                "flags": ep_flags,
            }
        )

    gripper_stats = quantiles(np.asarray(all_gripper, dtype=np.float32))
    lag_stats = quantiles(np.asarray(all_lag, dtype=np.float32))
    jerk_stats = quantiles(np.asarray(all_jerk, dtype=np.float32))
    flags = []
    if gripper_stats["q90"] < q90_threshold:
        flags.append(f"dataset_gripper_q90_low:{gripper_stats['q90']:.2f}<{q90_threshold}")
    if gripper_stats["q99"] < q99_threshold:
        flags.append(f"dataset_gripper_q99_low:{gripper_stats['q99']:.2f}<{q99_threshold}")
    if lag_stats["q95"] > lag_p95_threshold:
        flags.append(f"gripper_action_state_lag_high:{lag_stats['q95']:.2f}>{lag_p95_threshold}")
    if jerk_stats["q99"] > jerk_p99_threshold:
        flags.append(f"action_jerk_high:{jerk_stats['q99']:.2f}>{jerk_p99_threshold}")
    over_20 = sum(1 for d in durations if d > 20.0)
    if over_20:
        flags.append(f"episodes_over_20s:{over_20}/{len(durations)}")

    return {
        "repo_id": repo_id,
        "fps": fps,
        "num_episodes": int(ds.num_episodes),
        "num_frames": int(ds.num_frames),
        "action_names": action_names,
        "state_names": state_names,
        "gripper_index": gripper_idx,
        "gripper_stats": gripper_stats,
        "gripper_action_state_lag_stats": lag_stats,
        "action_jerk_l2_stats": jerk_stats,
        "duration_s": quantiles(np.asarray(durations, dtype=np.float32)),
        "episodes_over_20s": over_20,
        "simulated_gripper_repair": bool(simulate_gripper_repair),
        "repair_open_target": float(repair_open_target),
        "repair_open_threshold": float(repair_open_threshold),
        "repair_changed_frames": int(repaired_frames),
        "repair_changed_fraction": (float(repaired_frames) / float(total_frames)) if total_frames else 0.0,
        "episode_reports": episode_reports,
        "flags": flags,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eval 3 Dataset Gripper Audit",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "| repo | episodes | frames | grip q90 | grip q99 | repair changed | median duration | over 20s | flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["datasets"]:
        lines.append(
            f"| `{row['repo_id']}` | {row['num_episodes']} | {row['num_frames']} | "
            f"{row['gripper_stats']['q90']:.2f} | {row['gripper_stats']['q99']:.2f} | "
            f"{row.get('repair_changed_fraction', 0.0) * 100.0:.1f}% | "
            f"{row['duration_s']['q50']:.2f} | {row['episodes_over_20s']} | "
            f"{'; '.join(row['flags']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `dataset_gripper_q90_low` and `dataset_gripper_q99_low` directly support the gripper-not-opening hypothesis.",
            "- `pre_approach_gripper_not_open` in per-episode rows means the gripper was not wide before the first quarter of the trajectory.",
            "- High action-state lag means the commanded gripper is ahead of the measured gripper and may not physically open as far as the dataset action suggests.",
            "- `--simulate-gripper-repair` reports post-repair metrics without changing the Hub dataset; use it as a retrain gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Eval 3 datasets for gripper/action quality.")
    ap.add_argument("--repo", action="append", default=[], help="Dataset repo to audit. Defaults to Eval 3 old+v2 truncated repos.")
    ap.add_argument("--out-dir", default="outputs/eval3_abcd_eval")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--release-open-threshold", type=float, default=45.0)
    ap.add_argument("--gripper-q90-threshold", type=float, default=45.0)
    ap.add_argument("--gripper-q99-threshold", type=float, default=50.0)
    ap.add_argument("--lag-p95-threshold", type=float, default=20.0)
    ap.add_argument("--jerk-p99-threshold", type=float, default=8000.0)
    ap.add_argument("--simulate-gripper-repair", action="store_true", help="Lift already-open gripper commands before computing metrics.")
    ap.add_argument("--repair-open-target", type=float, default=55.0)
    ap.add_argument("--repair-open-threshold", type=float, default=20.0)
    args = ap.parse_args()

    repos = args.repo or DEFAULT_REPOS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for repo in repos:
        print(f">> auditing {repo}", flush=True)
        try:
            datasets.append(
                audit_repo(
                    repo,
                    video_backend=args.video_backend,
                    release_open_threshold=args.release_open_threshold,
                    q90_threshold=args.gripper_q90_threshold,
                    q99_threshold=args.gripper_q99_threshold,
                    lag_p95_threshold=args.lag_p95_threshold,
                    jerk_p99_threshold=args.jerk_p99_threshold,
                    simulate_gripper_repair=args.simulate_gripper_repair,
                    repair_open_target=args.repair_open_target,
                    repair_open_threshold=args.repair_open_threshold,
                )
            )
        except Exception as exc:
            datasets.append({"repo_id": repo, "error": f"{type(exc).__name__}: {exc}", "flags": ["audit_error"]})

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds": {
            "release_open_threshold": args.release_open_threshold,
            "gripper_q90_threshold": args.gripper_q90_threshold,
            "gripper_q99_threshold": args.gripper_q99_threshold,
            "lag_p95_threshold": args.lag_p95_threshold,
            "jerk_p99_threshold": args.jerk_p99_threshold,
            "simulate_gripper_repair": args.simulate_gripper_repair,
            "repair_open_target": args.repair_open_target,
            "repair_open_threshold": args.repair_open_threshold,
        },
        "datasets": datasets,
    }
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(report, indent=2, default=json_default), encoding="utf-8"
    )
    (out_dir / "DATASET_AUDIT.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / 'dataset_audit.json'}")
    print(f"wrote {out_dir / 'DATASET_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

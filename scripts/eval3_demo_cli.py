#!/usr/bin/env python3
"""Interactive Eval 3 demo CLI for TA workstations (Ubuntu / macOS).

Loads the policy and robot **once**, then accepts celebrity prompts in a loop
without reloading the checkpoint between rollouts.

Default model (override anytime)::

    export EVAL3_DEMO_POLICY=RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k
    ./scripts/run_eval3_demo_cli.sh

Quick start on demo day::

    ./scripts/run_eval3_demo_cli.sh
    # at the prompt:  taylor   |   yann   |   obama
    # or full text:   Place the coke on Taylor Swift
    # commands:       help | home | quit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from eval3_prompt_normalize import (  # noqa: E402
    CANONICAL_TASKS,
    normalize_eval3_task,
)

# Presets bundle policy + dataset schema + camera rename defaults.
# Swap models with EVAL3_DEMO_PRESET or EVAL3_DEMO_POLICY (see run_eval3_demo_cli.sh).
DEMO_PRESETS: dict[str, dict[str, Any]] = {
    "v4slots_expert": {
        "policy": "RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k",
        "dataset_repo_id": "RobotLearningVLA/dataset_v4_taylor_left",
        "rename_map": {"observation.images.front": "observation.images.camera1"},
        "empty_cameras": 2,
        "deploy_guards": False,
    },
    "v4slots_full": {
        "policy": "RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-50k",
        "dataset_repo_id": "RobotLearningVLA/dataset_v4_taylor_left",
        "rename_map": {"observation.images.front": "observation.images.camera1"},
        "empty_cameras": 2,
        "deploy_guards": False,
    },
    "v6_new": {
        "policy": "RobotLearningVLA/eval3-vla-v6-smolvla-fresh-new66-50k",
        "dataset_repo_id": "RobotLearningVLA/taylor_swift_1",
        "rename_map": {"observation.images.front": "observation.images.camera1"},
        "empty_cameras": 2,
        "deploy_guards": True,
    },
    "v16": {
        "policy": "RobotLearningVLA/eval3-vla-v16-real-synth-50k-step1k",
        "dataset_repo_id": "RobotLearningVLA/dataset_v4_taylor_left",
        "rename_map": {
            "observation.images.front": "observation.images.camera1",
            "observation.images.front_frame0": "observation.images.camera2",
        },
        "empty_cameras": 1,
        "deploy_guards": False,
    },
}

SHORTCUTS: dict[str, str] = {
    "taylor": "Place the coke on Taylor Swift",
    "swift": "Place the coke on Taylor Swift",
    "yann": "Place the coke on Yann LeCun",
    "lecun": "Place the coke on Yann LeCun",
    "obama": "Place the coke on Barack Obama",
    "barack": "Place the coke on Barack Obama",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Eval 3 interactive demo CLI (load policy once, many prompts).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--preset",
        default=os.environ.get("EVAL3_DEMO_PRESET", "v4slots_expert"),
        choices=sorted(DEMO_PRESETS),
        help="Named deploy recipe (policy + dataset schema + rename_map).",
    )
    ap.add_argument(
        "--policy",
        default=os.environ.get("EVAL3_DEMO_POLICY", "").strip(),
        help="Hub id or local path; overrides the preset's policy only.",
    )
    ap.add_argument(
        "--dataset-repo-id",
        default=os.environ.get("EVAL3_DEMO_DATASET", "").strip(),
        help="Dataset schema source (defaults from preset).",
    )
    ap.add_argument("--robot-port", default=os.environ.get("FOLLOWER_TTY", "/dev/ttyUSB0"))
    ap.add_argument("--robot-id", default=os.environ.get("ROBOT_ID", "my_awesome_follower_arm"))
    ap.add_argument("--camera", type=int, default=int(os.environ.get("CAM_IDX", "0")))
    ap.add_argument(
        "--device",
        default=os.environ.get("EVAL3_POLICY_DEVICE", "auto"),
        help="Policy device: auto (cuda on Linux, mps on Mac), cuda, cpu, mps.",
    )
    ap.add_argument("--episode-time-s", type=float, default=float(os.environ.get("EVAL3_EPISODE_TIME_S", "20")))
    ap.add_argument("--fps", type=int, default=int(os.environ.get("EVAL3_FPS", "30")))
    ap.add_argument("--home-duration-s", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true", help="Load policy only; do not connect hardware.")
    ap.add_argument(
        "--once",
        metavar="TASK",
        default="",
        help="Run a single rollout and exit (non-interactive).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def _resolve_demo_config(args: argparse.Namespace) -> dict[str, Any]:
    preset = DEMO_PRESETS[args.preset]
    policy = (args.policy or preset["policy"]).strip()
    dataset_repo_id = (args.dataset_repo_id or preset["dataset_repo_id"]).strip()
    rename_map = dict(preset["rename_map"])
    empty_cameras = int(preset["empty_cameras"])
    deploy_guards = bool(preset.get("deploy_guards", False))

    # If only --policy is overridden, try to read train_config.json for v16 / rename hints.
    if args.policy:
        inferred = _infer_from_train_config(policy)
        if inferred.get("rename_map"):
            rename_map = inferred["rename_map"]
        if "empty_cameras" in inferred:
            empty_cameras = inferred["empty_cameras"]

    return {
        "policy": policy,
        "dataset_repo_id": dataset_repo_id,
        "rename_map": rename_map,
        "empty_cameras": empty_cameras,
        "deploy_guards": deploy_guards,
        "robot_port": args.robot_port,
        "robot_id": args.robot_id,
        "camera": args.camera,
        "device": args.device,
        "episode_time_s": args.episode_time_s,
        "fps": args.fps,
        "home_duration_s": args.home_duration_s,
        "dry_run": args.dry_run,
    }


def _infer_from_train_config(policy_path: str) -> dict[str, Any]:
    tc: Path | None = None
    local = Path(policy_path) / "train_config.json"
    if local.is_file():
        tc = local
    elif "/" in policy_path and not Path(policy_path).exists():
        try:
            from huggingface_hub import hf_hub_download

            tc = Path(hf_hub_download(policy_path, "train_config.json"))
        except Exception:
            return {}
    if tc is None or not tc.is_file():
        return {}
    try:
        data = json.loads(tc.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, Any] = {}
    rmap = data.get("rename_map")
    if isinstance(rmap, dict) and rmap:
        out["rename_map"] = {str(k): str(v) for k, v in rmap.items()}
    ec = data.get("policy", {}).get("empty_cameras") if isinstance(data.get("policy"), dict) else None
    if ec is None:
        ec = data.get("empty_cameras")
    if ec is not None:
        out["empty_cameras"] = int(ec)
    return out


def _inject_policy_argv(policy_path: str) -> None:
    """eval3_vla_deploy picks slot vs aux head from --policy.path at import time."""
    sys.argv = [sys.argv[0], f"--policy.path={policy_path}"]


def _resolve_device_name(requested: str) -> str:
    from eval3_device import resolve_eval3_device

    dev = resolve_eval3_device(requested)
    return str(dev)


def _print_banner(cfg: dict[str, Any]) -> None:
    guards = "ON (smoothed)" if cfg["deploy_guards"] else "OFF (raw policy)"
    print()
    print("=" * 60)
    print("  Eval 3 Demo CLI")
    print(f"  Model   : {cfg['policy']}")
    print(f"  Dataset : {cfg['dataset_repo_id']}  (schema only)")
    print(f"  Port    : {cfg['robot_port']}   Camera: {cfg['camera']}   Device: {cfg['device']}")
    print(f"  Rollout : {cfg['episode_time_s']:.0f}s @ {cfg['fps']} Hz   Guards: {guards}")
    print("=" * 60)
    print()
    print("  Shortcuts : taylor | yann | obama")
    print("  Commands  : help | home | quit")
    print("  Or type a full instruction, e.g. Place the coke on Taylor Swift")
    print()


def _resolve_user_line(line: str) -> str | None:
    """Return canonical task string, or None for meta-commands."""
    text = line.strip()
    if not text:
        return None
    low = text.lower()
    if low in {"quit", "exit", "q"}:
        raise SystemExit(0)
    if low == "help":
        print("\nCanonical prompts:")
        for t in sorted(CANONICAL_TASKS):
            print(f"  • {t}")
        print("\nShortcuts:", ", ".join(sorted(SHORTCUTS)))
        print("Commands: home (return home now), quit\n")
        return None
    if low == "home":
        return "__HOME__"
    if low in SHORTCUTS:
        return SHORTCUTS[low]
    norm = normalize_eval3_task(text)
    if norm.changed:
        print(f"  (normalized: {norm.normalized!r})")
    return norm.normalized


@dataclass
class DemoSession:
    cfg: dict[str, Any]
    policy: Any
    preprocessor: Any
    postprocessor: Any
    robot: Any
    home_positions: dict[str, float]
    events: dict
    listener: Any
    ds_features: dict
    robot_action_processor: Any
    robot_observation_processor: Any
    interpolator: Any | None


def _load_policy_cfg(cfg: dict[str, Any]):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    policy_cfg = PreTrainedConfig.from_pretrained(cfg["policy"])
    policy_cfg.pretrained_path = cfg["policy"]
    policy_cfg.device = _resolve_device_name(cfg["device"])
    if hasattr(policy_cfg, "empty_cameras"):
        policy_cfg.empty_cameras = int(cfg["empty_cameras"])
    return policy_cfg


def _open_session(cfg: dict[str, Any]) -> DemoSession:
    # Import after argv injection so slot/aux patch matches the checkpoint.
    import eval3_vla_deploy as evd  # noqa: WPS433

    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.rtc import ActionInterpolator
    from lerobot.processor.rename_processor import rename_stats
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.utils.control_utils import init_keyboard_listener
    from lerobot.utils.utils import init_logging
    from lerobot.processor import make_default_processors

    init_logging()
    policy_cfg = _load_policy_cfg(cfg)

    rename_map = dict(cfg["rename_map"])
    is_v16 = evd._FRAME0_RENAME_MAP is not None
    if is_v16 and evd._FRAME0_RENAME_MAP:
        rename_map = dict(evd._FRAME0_RENAME_MAP)

    ds_meta = LeRobotDatasetMetadata(cfg["dataset_repo_id"])
    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg["policy"],
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

    interpolator = None
    if cfg["deploy_guards"]:
        interpolator = ActionInterpolator(multiplier=2)

    robot_cfg = SO101FollowerConfig(
        port=cfg["robot_port"],
        id=cfg["robot_id"],
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=int(cfg["camera"]),
                width=640,
                height=480,
                fps=int(cfg["fps"]),
            )
        },
    )
    _, robot_action_processor, robot_observation_processor = make_default_processors()

    robot = make_robot_from_config(robot_cfg)
    listener, events = init_keyboard_listener()
    robot.connect()
    home_positions = evd._capture_home_positions(robot)
    logging.info("Demo session ready — home captured (%d joints).", len(home_positions))

    return DemoSession(
        cfg=cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot=robot,
        home_positions=home_positions,
        events=events,
        listener=listener,
        ds_features=ds_meta.features,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
        interpolator=interpolator,
    )


def _close_session(session: DemoSession) -> None:
    import eval3_vla_deploy as evd  # noqa: WPS433

    if session.robot.is_connected:
        session.robot.disconnect()
    if session.listener is not None:
        session.listener.stop()
    logging.info("Demo session closed.")


def _run_rollout(session: DemoSession, task: str) -> None:
    import eval3_vla_deploy as evd  # noqa: WPS433
    from datetime import datetime, timezone

    from lerobot.utils.utils import log_say

    cfg = session.cfg
    norm = normalize_eval3_task(task)
    canonical = norm.normalized
    print(f"\n>> Running: {canonical!r}  ({cfg['episode_time_s']:.0f}s)")
    if norm.changed:
        print(f"   (from: {task!r})")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rollout_log_path = Path("outputs/eval3_rollouts") / f"rollout_{ts}.jsonl"
    rollout_header = {
        "instruction": canonical,
        "raw_task": task,
        "episode_time_s": cfg["episode_time_s"],
        "fps": cfg["fps"],
        "policy_path": cfg["policy"],
        "dataset_repo_id": cfg["dataset_repo_id"],
        "demo_cli": True,
    }
    print(f"   Log: {rollout_log_path}")

    alpha = 0.25 if cfg["deploy_guards"] else 0.0
    max_delta = 6.0 if cfg["deploy_guards"] else 0.0
    grip_bias = 5.0 if cfg["deploy_guards"] else 0.0

    log_say(f"Running policy for {cfg['episode_time_s']}s", False)
    session.events["exit_early"] = False
    evd._deploy_loop(
        robot=session.robot,
        policy=session.policy,
        preprocessor=session.preprocessor,
        postprocessor=session.postprocessor,
        ds_features=session.ds_features,
        fps=int(cfg["fps"]),
        episode_time_s=float(cfg["episode_time_s"]),
        robot_observation_processor=session.robot_observation_processor,
        robot_action_processor=session.robot_action_processor,
        display_data=False,
        events=session.events,
        interpolator=session.interpolator,
        single_task=canonical,
        action_smoothing_alpha=alpha,
        max_action_delta_deg=max_delta,
        gripper_open_bias_deg=grip_bias,
        gripper_open_bias_threshold_deg=20.0,
        rollout_log_path=rollout_log_path,
        rollout_header=rollout_header,
    )
    log_say("Episode finished", False)
    if session.events.get("exit_early"):
        session.events["exit_early"] = False
        print("   (stopped early — Esc)")


def _go_home(session: DemoSession) -> None:
    import eval3_vla_deploy as evd  # noqa: WPS433

    print(">> Returning to home …")
    evd._go_home(
        session.robot,
        session.home_positions,
        duration_s=float(session.cfg["home_duration_s"]),
        fps=int(session.cfg["fps"]),
    )


def _dry_run(cfg: dict[str, Any]) -> None:
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_stats

    policy_cfg = _load_policy_cfg(cfg)
    ds_meta = LeRobotDatasetMetadata(cfg["dataset_repo_id"])
    make_policy(policy_cfg, ds_meta=ds_meta, rename_map=cfg["rename_map"])
    make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg["policy"],
        dataset_stats=rename_stats(ds_meta.stats, cfg["rename_map"]),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": cfg["rename_map"]},
        },
    )
    print("Dry run OK — policy and processors loaded; no robot connection.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    cfg = _resolve_demo_config(args)
    cfg["device"] = _resolve_device_name(cfg["device"])

    _inject_policy_argv(cfg["policy"])
    from eval3_lerobot_shim import apply as _shim_apply  # noqa: WPS433

    _shim_apply()
    import eval3_vla_deploy  # noqa: F401, WPS433 — installs head patch

    _print_banner(cfg)

    if cfg["dry_run"]:
        _dry_run(cfg)
        return 0

    if args.once:
        session = _open_session(cfg)
        try:
            task = _resolve_user_line(args.once) or args.once
            if task == "__HOME__":
                _go_home(session)
            else:
                _run_rollout(session, task)
                _go_home(session)
        finally:
            _close_session(session)
        return 0

    session = _open_session(cfg)
    try:
        while True:
            try:
                line = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            try:
                task = _resolve_user_line(line)
            except SystemExit:
                break
            if task is None:
                continue
            if task == "__HOME__":
                _go_home(session)
                continue
            _run_rollout(session, task)
            _go_home(session)
    finally:
        _close_session(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

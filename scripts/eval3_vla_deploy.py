#!/usr/bin/env python3
"""SmolVLA closed-loop control on the real robot (Eval 3).

Loads a ``pretrained_model`` checkpoint, pulls dataset feature/stats schema from the training Hub
repo, runs the standard LeRobot observation → preprocess → policy → postprocess → robot pipeline.

Prereqs (same venv as teleop)::

    uv pip install transformers accelerate sentencepiece

Example::

    python scripts/eval3_vla_deploy.py \\
      --robot.type=so101_follower \\
      --robot.port=/dev/tty.usbmodemXXXX \\
      --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
      --robot.id=my_awesome_follower_arm \\
      --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \\
      --rename_map '{"observation.images.front":"observation.images.camera1"}' \\
      --policy.path=outputs/train/eval3_smolvla/checkpoints/050000/pretrained_model \\
      --policy.device=mps \\
      --task=\"Place the coke on Taylor Swift\" \\
      --episode_time_s=20 \\
      --fps=30

Use ``--dry_run`` to verify checkpoint + metadata load without connecting hardware.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from eval3_prompt_normalize import normalize_eval3_task  # noqa: E402

from eval3_lerobot_shim import apply as _eval3_shim_apply

_eval3_shim_apply()

# Apply the aux-head patch so checkpoints trained with EVAL3_AUX_POS_LOSS_WEIGHT
# > 0 load cleanly (their state_dict carries `position_clf_head.*` keys). Setting
# loss_weight unset / 0 makes it a runtime no-op — the head exists but never
# contributes a gradient, and sample_actions doesn't touch it at all.
from eval3_smolvla_aux_head import apply as _eval3_aux_head_apply  # noqa: E402

_eval3_aux_head_apply()

import torch  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from lerobot.configs import parser  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.feature_utils import build_dataset_frame  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.rtc import ActionInterpolator  # noqa: E402
from lerobot.policies.utils import make_robot_action  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from lerobot.robots import (  # noqa: F401, E402
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.robots import Robot  # noqa: E402
from lerobot.utils.constants import OBS_STR  # noqa: E402
from lerobot.utils.control_utils import init_keyboard_listener, predict_action  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging, log_say  # noqa: E402

from lerobot.processor import (  # noqa: E402
    PolicyProcessorPipeline,
    RobotProcessorPipeline,
    make_default_processors,
)


def _to_float_list(value) -> list[float] | None:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().flatten().tolist()
        else:
            import numpy as _np

            arr = _np.asarray(value).reshape(-1).tolist()
        return [float(x) for x in arr]
    except Exception:
        return None


def _float_mapping(mapping: dict | None) -> dict[str, float] | None:
    if mapping is None:
        return None
    out = {}
    for key in sorted(mapping):
        try:
            out[key] = float(mapping[key])
        except Exception:
            pass
    return out


def _apply_action_guards(
    raw_action: dict[str, float],
    previous_action: dict[str, float] | None,
    *,
    smoothing_alpha: float,
    max_action_delta_deg: float,
    gripper_open_bias_deg: float,
    gripper_open_bias_threshold_deg: float,
) -> dict[str, float]:
    """Apply optional deploy-time action smoothing without hiding raw logs.

    ``smoothing_alpha`` is the weight assigned to the previous command. A value
    of 0.25 means 75% current target + 25% previous target. ``max_action_delta``
    clamps the per-tick change after smoothing; 0 disables it. Gripper open
    bias is intentionally thresholded so a positive bias cannot turn a close
    command into a weak half-open grasp.
    """
    guarded: dict[str, float] = {}
    for key, raw_value in raw_action.items():
        target = float(raw_value)
        is_gripper = "gripper" in key.lower()
        if (
            gripper_open_bias_deg
            and is_gripper
            and target >= float(gripper_open_bias_threshold_deg)
        ):
            target += float(gripper_open_bias_deg)

        if previous_action is not None and key in previous_action and smoothing_alpha > 0:
            prev = float(previous_action[key])
            target = float(smoothing_alpha) * prev + (1.0 - float(smoothing_alpha)) * target

        if previous_action is not None and key in previous_action and max_action_delta_deg > 0:
            prev = float(previous_action[key])
            delta = float(max_action_delta_deg)
            target = min(max(target, prev - delta), prev + delta)

        guarded[key] = target
    return guarded


def _pretrained_file(policy_path: str, filename: str) -> str | None:
    local = Path(policy_path)
    if local.exists():
        candidate = local / filename
        return str(candidate) if candidate.is_file() else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(policy_path, filename)
    except Exception as exc:
        logging.warning("Could not resolve %s from %s: %s", filename, policy_path, exc)
        return None


def _policy_sha(policy_path: str) -> str | None:
    if Path(policy_path).exists():
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(policy_path).sha
    except Exception as exc:
        logging.warning("Could not resolve model sha for %s: %s", policy_path, exc)
        return None


def _checkpoint_stats_counts(policy_path: str) -> dict[str, float | None]:
    path = _pretrained_file(policy_path, "policy_preprocessor_step_5_normalizer_processor.safetensors")
    if not path:
        return {"action": None, "observation.state": None}
    try:
        from safetensors.torch import load_file

        tensors = load_file(path)
        return {
            "action": float(tensors["action.count"].flatten()[0]) if "action.count" in tensors else None,
            "observation.state": (
                float(tensors["observation.state.count"].flatten()[0])
                if "observation.state.count" in tensors
                else None
            ),
        }
    except Exception as exc:
        logging.warning("Could not read checkpoint stats counts from %s: %s", path, exc)
        return {"action": None, "observation.state": None}


@dataclass
class Eval3VLADeployConfig:
    robot: RobotConfig
    # Hub repo whose meta matches training (observation keys + stats).
    dataset_repo_id: str = "RobotLearningVLA/taylor_swift_1"
    task: str = ""
    target_slot: str = ""
    episode_time_s: float = 20.0
    fps: int = 30
    display_data: bool = False
    rename_map: dict[str, str] = field(
        default_factory=lambda: {"observation.images.front": "observation.images.camera1"}
    )
    interpolation_multiplier: int = 1
    action_smoothing_alpha: float = 0.0
    max_action_delta_deg: float = 0.0
    gripper_open_bias_deg: float = 0.0
    gripper_open_bias_threshold_deg: float = 20.0
    dry_run: bool = False
    play_sounds: bool = False
    rollout_log_dir: str = "outputs/eval3_rollouts"
    n_rollouts: int = 1
    go_home_after_rollout: bool = True
    home_duration_s: float = 3.0
    normalize_task: bool = True
    raw_task: str = field(default="", repr=False)
    policy: PreTrainedConfig | None = None

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Provide --policy.path=/path/to/pretrained_model (or Hub id).")
        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path
        if self.target_slot.strip():
            slot = self.target_slot.strip().lower()
            if slot not in {"left", "middle", "right"}:
                raise ValueError("--target_slot must be one of left, middle, right.")
            self.task = f"Place the coke on the {slot} print"
        if not self.task.strip():
            print(
                "Enter Eval 3 instruction (e.g. 'Place the coke on Taylor Swift'), then press Enter:",
                file=sys.stderr,
                flush=True,
            )
            line = sys.stdin.readline()
            if not line.strip():
                raise ValueError(
                    "No task on stdin and --task is empty. "
                    "Provide --task=... or type the instruction at the prompt."
                )
            self.task = line.strip()

        self.raw_task = self.task
        if self.normalize_task and not self.target_slot.strip():
            norm = normalize_eval3_task(self.task)
            if norm.changed:
                logging.info(
                    "Normalized deploy task: %r -> %r",
                    norm.raw,
                    norm.normalized,
                )
            self.task = norm.normalized

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _go_home(
    robot: Robot,
    home_positions: dict[str, float],
    *,
    duration_s: float,
    fps: int,
) -> None:
    """Drive joints back toward the pose captured at connect (vla_eval1 pattern)."""
    if not home_positions:
        return
    logging.info("Returning to home position for %.1fs …", duration_s)
    deadline = time.perf_counter() + float(duration_s)
    interval = 1.0 / max(int(fps), 1)
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        robot.send_action(home_positions)
        precise_sleep(max(0.0, interval - (time.perf_counter() - t0)))
    logging.info("Home return finished.")


def _capture_home_positions(robot: Robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: float(v) for k, v in obs.items() if k.endswith(".pos")}


def _deploy_loop(
    robot: Robot,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    ds_features: dict,
    *,
    fps: int,
    episode_time_s: float,
    robot_observation_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    display_data: bool,
    events: dict,
    interpolator: ActionInterpolator | None,
    single_task: str,
    action_smoothing_alpha: float,
    max_action_delta_deg: float,
    gripper_open_bias_deg: float,
    gripper_open_bias_threshold_deg: float,
    display_compressed_images: bool = False,
    rollout_log_path: Path | None = None,
    rollout_header: dict | None = None,
) -> None:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    if interpolator is not None:
        interpolator.reset()

    dev = get_safe_torch_device(policy.config.device)
    use_interpolation = interpolator is not None and interpolator.enabled
    control_interval = interpolator.get_control_interval(fps) if interpolator else 1.0 / fps
    action_keys = sorted(robot.action_features) if use_interpolation else []
    previous_guarded_action: dict[str, float] | None = None

    log_file = None
    first_frame_path = None
    if rollout_log_path is not None:
        rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = rollout_log_path.open("w", encoding="utf-8")
        log_file.write(json.dumps(rollout_header or {}) + "\n")
        log_file.flush()
        first_frame_path = rollout_log_path.with_suffix(".firstframe.png")

    step_idx = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    try:
        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()

            if events["exit_early"]:
                events["exit_early"] = False
                break

            obs = robot.get_observation()
            # On the very first tick, dump the camera frame(s) so each rollout
            # self-documents which physical camera the deploy actually used.
            if step_idx == 0 and first_frame_path is not None:
                try:
                    from PIL import Image as _PIL_Image
                    for _k, _v in obs.items():
                        if hasattr(_v, "shape") and getattr(_v, "ndim", 0) == 3 and _v.shape[-1] == 3:
                            _PIL_Image.fromarray(_v).save(first_frame_path)
                            logging.info("First-tick camera frame saved to %s (camera key %r)",
                                         first_frame_path, _k)
                            break
                except Exception as _e:
                    logging.warning("Could not save first-tick frame: %s", _e)
            obs_processed = robot_observation_processor(obs)

            observation_frame = build_dataset_frame(ds_features, obs_processed, prefix=OBS_STR)
            is_record_frame = True
            policy_action_raw = None
            policy_action_processed = None
            policy_action_guarded = None

            if use_interpolation:
                ran_inference = False
                if interpolator.needs_new_action():
                    action_values = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=dev,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=single_task,
                        robot_type=robot.robot_type,
                    )
                    act_processed_policy = make_robot_action(action_values, ds_features)
                    robot_action_policy = robot_action_processor((act_processed_policy, obs))
                    policy_action_raw = _to_float_list(action_values)
                    policy_action_processed = _float_mapping(robot_action_policy)
                    robot_action_guarded = _apply_action_guards(
                        robot_action_policy,
                        previous_guarded_action,
                        smoothing_alpha=action_smoothing_alpha,
                        max_action_delta_deg=max_action_delta_deg,
                        gripper_open_bias_deg=gripper_open_bias_deg,
                        gripper_open_bias_threshold_deg=gripper_open_bias_threshold_deg,
                    )
                    previous_guarded_action = robot_action_guarded
                    policy_action_guarded = _float_mapping(robot_action_guarded)
                    action_tensor = torch.tensor([robot_action_guarded[k] for k in action_keys])
                    interpolator.add(action_tensor)
                    ran_inference = True
                interp_action = interpolator.get()
                if interp_action is not None:
                    robot_action_to_send = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                    action_values = robot_action_to_send
                else:
                    continue
                is_record_frame = ran_inference
            else:
                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=dev,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )
                act_processed_policy = make_robot_action(action_values, ds_features)
                robot_action_policy = robot_action_processor((act_processed_policy, obs))
                policy_action_raw = _to_float_list(action_values)
                policy_action_processed = _float_mapping(robot_action_policy)
                robot_action_to_send = _apply_action_guards(
                    robot_action_policy,
                    previous_guarded_action,
                    smoothing_alpha=action_smoothing_alpha,
                    max_action_delta_deg=max_action_delta_deg,
                    gripper_open_bias_deg=gripper_open_bias_deg,
                    gripper_open_bias_threshold_deg=gripper_open_bias_threshold_deg,
                )
                previous_guarded_action = robot_action_to_send
                policy_action_guarded = _float_mapping(robot_action_to_send)
                action_values = robot_action_to_send

            robot.send_action(robot_action_to_send)

            step_idx += 1

            if display_data:
                from lerobot.utils.visualization_utils import log_rerun_data

                log_rerun_data(
                    observation=obs_processed, action=action_values, compress_images=display_compressed_images
                )

            dt_s = time.perf_counter() - start_loop_t
            sleep_time_s = control_interval - dt_s
            if log_file is not None:
                action_keys_out = sorted(robot_action_to_send)
                log_file.write(
                    json.dumps(
                        {
                            "step": step_idx - 1,
                            "t_monotonic": time.perf_counter(),
                            "t_episode_s": time.perf_counter() - start_episode_t,
                            "dt_s": float(dt_s),
                            "loop_hz": float(1.0 / dt_s) if dt_s > 0 else None,
                            "target_fps": float(fps),
                            "sleep_s": float(max(sleep_time_s, 0.0)),
                            "ran_policy_inference": bool(is_record_frame),
                            "state": _to_float_list(observation_frame.get("observation.state")),
                            "joint_keys": action_keys_out,
                            "policy_action_raw": policy_action_raw,
                            "policy_action_processed": policy_action_processed,
                            "policy_action_guarded": policy_action_guarded,
                            "sent_action": _float_mapping(robot_action_to_send),
                            # Backward-compatible field consumed by older audit helpers.
                            "action": [float(robot_action_to_send[k]) for k in action_keys_out],
                        }
                    )
                    + "\n"
                )
            if sleep_time_s < 0:
                logging.warning(
                    "Loop slower than target FPS — inference or cameras may be overloaded "
                    f"(running ~{1 / dt_s:.1f} Hz vs target {fps} Hz)."
                )
            precise_sleep(max(sleep_time_s, 0.0))

            timestamp = time.perf_counter() - start_episode_t
    finally:
        if log_file is not None:
            log_file.close()


@parser.wrap()
def eval3_vla_deploy(cfg: Eval3VLADeployConfig) -> None:
    register_third_party_plugins()
    init_logging()
    logging.info(
        "eval3_vla_deploy dataset_repo_id=%s episode_time_s=%s fps=%s dry_run=%s",
        cfg.dataset_repo_id,
        cfg.episode_time_s,
        cfg.fps,
        cfg.dry_run,
    )

    assert cfg.policy is not None
    if not 0.0 <= cfg.action_smoothing_alpha < 1.0:
        raise ValueError("--action_smoothing_alpha must be in [0, 1).")
    if cfg.max_action_delta_deg < 0:
        raise ValueError("--max_action_delta_deg must be >= 0; use 0 to disable.")
    if cfg.gripper_open_bias_threshold_deg < 0:
        raise ValueError("--gripper_open_bias_threshold_deg must be >= 0.")
    if cfg.rename_map.get("observation.images.front") != "observation.images.camera1":
        logging.warning(
            "Eval3 single-camera deploy normally requires "
            "--rename_map='{\"observation.images.front\":\"observation.images.camera1\"}'. "
            "Current rename_map=%s",
            cfg.rename_map,
        )
    empty_cameras = getattr(cfg.policy, "empty_cameras", None)
    if empty_cameras != 2:
        logging.warning(
            "Eval3 SmolVLA single-camera checkpoints should use policy.empty_cameras=2; got %r.",
            empty_cameras,
        )

    ds_meta = LeRobotDatasetMetadata(cfg.dataset_repo_id)
    ds_features = ds_meta.features

    policy = make_policy(cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(ds_meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    interpolator = None
    if cfg.interpolation_multiplier > 1:
        interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)

    if cfg.dry_run:
        logging.info(
            "Dry run OK — policy and processors loaded; task=%r (raw=%r); skipping robot I/O.",
            cfg.task,
            cfg.raw_task or cfg.task,
        )
        return

    if cfg.n_rollouts < 1:
        raise ValueError("--n_rollouts must be >= 1.")

    if cfg.display_data:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(session_name="eval3_vla_deploy")

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    stats_counts = _checkpoint_stats_counts(str(cfg.policy.pretrained_path))

    robot = make_robot_from_config(cfg.robot)
    listener = None
    try:
        robot.connect()
        listener, events = init_keyboard_listener()
        home_positions = _capture_home_positions(robot)
        logging.info("Home position captured (%d joints).", len(home_positions))

        for rollout_idx in range(cfg.n_rollouts):
            if cfg.n_rollouts > 1:
                print(f"\n[eval3] Rollout {rollout_idx + 1}/{cfg.n_rollouts}")
                if cfg.raw_task and cfg.raw_task != cfg.task:
                    print(f"[eval3] Raw prompt:        {cfg.raw_task!r}")
                    print(f"[eval3] Normalized prompt: {cfg.task!r}")
                else:
                    print(f"[eval3] Prompt: {cfg.task!r}")

            rollout_log_path: Path | None = None
            rollout_header: dict | None = None
            if cfg.rollout_log_dir:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                suffix = f"_{rollout_idx}" if cfg.n_rollouts > 1 else ""
                rollout_log_path = Path(cfg.rollout_log_dir) / f"rollout_{ts}{suffix}.jsonl"
                first_frame_path = rollout_log_path.with_suffix(".firstframe.png")
                rollout_header = {
                    "instruction": cfg.task,
                    "raw_task": cfg.raw_task or cfg.task,
                    "target_slot": cfg.target_slot,
                    "rollout_index": rollout_idx,
                    "n_rollouts": cfg.n_rollouts,
                    "episode_time_s": cfg.episode_time_s,
                    "fps": cfg.fps,
                    "policy_path": str(cfg.policy.pretrained_path),
                    "policy_sha": _policy_sha(str(cfg.policy.pretrained_path)),
                    "checkpoint_stats_count": stats_counts,
                    "dataset_repo_id": cfg.dataset_repo_id,
                    "rename_map": cfg.rename_map,
                    "interpolation_multiplier": cfg.interpolation_multiplier,
                    "action_smoothing_alpha": cfg.action_smoothing_alpha,
                    "max_action_delta_deg": cfg.max_action_delta_deg,
                    "gripper_open_bias_deg": cfg.gripper_open_bias_deg,
                    "gripper_open_bias_threshold_deg": cfg.gripper_open_bias_threshold_deg,
                    "robot_type": getattr(cfg.robot, "type", None),
                    "robot_id": getattr(cfg.robot, "id", None),
                    "policy_device": str(cfg.policy.device),
                    "first_frame_path": str(first_frame_path),
                }
                logging.info("Rollout log: %s", rollout_log_path)

            log_say(f"Running policy for {cfg.episode_time_s}s", cfg.play_sounds)
            _deploy_loop(
                robot=robot,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                ds_features=ds_features,
                fps=cfg.fps,
                episode_time_s=cfg.episode_time_s,
                robot_observation_processor=robot_observation_processor,
                robot_action_processor=robot_action_processor,
                display_data=cfg.display_data,
                events=events,
                interpolator=interpolator,
                single_task=cfg.task,
                action_smoothing_alpha=cfg.action_smoothing_alpha,
                max_action_delta_deg=cfg.max_action_delta_deg,
                gripper_open_bias_deg=cfg.gripper_open_bias_deg,
                gripper_open_bias_threshold_deg=cfg.gripper_open_bias_threshold_deg,
                display_compressed_images=False,
                rollout_log_path=rollout_log_path,
                rollout_header=rollout_header,
            )
            log_say("Episode finished", cfg.play_sounds)

            if events.get("exit_early"):
                events["exit_early"] = False
                break

            if cfg.go_home_after_rollout and rollout_idx + 1 < cfg.n_rollouts:
                _go_home(robot, home_positions, duration_s=cfg.home_duration_s, fps=cfg.fps)

        if cfg.go_home_after_rollout and home_positions:
            _go_home(robot, home_positions, duration_s=cfg.home_duration_s, fps=cfg.fps)
    finally:
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()


def main() -> None:
    eval3_vla_deploy()


if __name__ == "__main__":
    main()

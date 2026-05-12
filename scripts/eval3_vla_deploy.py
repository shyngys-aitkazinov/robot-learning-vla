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
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval3_lerobot_shim import apply as _eval3_shim_apply

_eval3_shim_apply()

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


@dataclass
class Eval3VLADeployConfig:
    robot: RobotConfig
    # Hub repo whose meta matches training (observation keys + stats).
    dataset_repo_id: str = "RobotLearningVLA/taylor_swift_1"
    task: str = ""
    episode_time_s: float = 20.0
    fps: int = 30
    display_data: bool = False
    rename_map: dict[str, str] = field(default_factory=dict)
    interpolation_multiplier: int = 1
    dry_run: bool = False
    play_sounds: bool = False
    rollout_log_dir: str = "outputs/eval3_rollouts"
    policy: PreTrainedConfig | None = None

    def __post_init__(self) -> None:
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Provide --policy.path=/path/to/pretrained_model (or Hub id).")
        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path
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

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


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
                    robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                    action_tensor = torch.tensor([robot_action_to_send[k] for k in action_keys])
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
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                action_values = robot_action_to_send

            robot.send_action(robot_action_to_send)

            if log_file is not None:
                action_keys_out = sorted(robot_action_to_send)
                log_file.write(
                    json.dumps(
                        {
                            "step": step_idx,
                            "t_monotonic": time.perf_counter(),
                            "t_episode_s": time.perf_counter() - start_episode_t,
                            "joint_keys": action_keys_out,
                            "action": [float(robot_action_to_send[k]) for k in action_keys_out],
                        }
                    )
                    + "\n"
                )
            step_idx += 1

            if display_data:
                from lerobot.utils.visualization_utils import log_rerun_data

                log_rerun_data(
                    observation=obs_processed, action=action_values, compress_images=display_compressed_images
                )

            dt_s = time.perf_counter() - start_loop_t
            sleep_time_s = control_interval - dt_s
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
        logging.info("Dry run OK — policy and processors loaded; skipping robot I/O.")
        return

    if cfg.display_data:
        from lerobot.utils.visualization_utils import init_rerun

        init_rerun(session_name="eval3_vla_deploy")

    _, robot_action_processor, robot_observation_processor = make_default_processors()

    rollout_log_path: Path | None = None
    rollout_header: dict | None = None
    if cfg.rollout_log_dir:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rollout_log_path = Path(cfg.rollout_log_dir) / f"rollout_{ts}.jsonl"
        rollout_header = {
            "instruction": cfg.task,
            "episode_time_s": cfg.episode_time_s,
            "fps": cfg.fps,
            "policy_path": str(cfg.policy.pretrained_path),
            "dataset_repo_id": cfg.dataset_repo_id,
            "rename_map": cfg.rename_map,
            "robot_type": getattr(cfg.robot, "type", None),
            "robot_id": getattr(cfg.robot, "id", None),
            "policy_device": str(cfg.policy.device),
        }
        logging.info("Rollout log will be written to %s", rollout_log_path)

    robot = make_robot_from_config(cfg.robot)
    listener = None
    try:
        robot.connect()
        listener, events = init_keyboard_listener()

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
            display_compressed_images=False,
            rollout_log_path=rollout_log_path,
            rollout_header=rollout_header,
        )
        log_say("Episode finished", cfg.play_sounds)
    finally:
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()


def main() -> None:
    eval3_vla_deploy()


if __name__ == "__main__":
    main()

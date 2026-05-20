#!/usr/bin/env python3
"""Closed-loop OpenVLA LoRA deploy for Eval 3 on SO-101.

OpenVLA checkpoints are **not** LeRobot ``PreTrainedPolicy`` artifacts. Use this
adapter instead of ``scripts/eval3_vla_deploy.py``.

Requires the OpenVLA training venv (``.venv_openvla_train``) with ``peft`` and
``openvla`` sources installed — see ``integrations/openvla/README.md``.

Dry-run (loads checkpoint, no motors)::

    source .venv_openvla_train/bin/activate
    OPENVLA_SRC=external/openvla python scripts/eval3_openvla_deploy.py \\
      --robot.type=so101_follower \\
      --robot.port=/dev/tty.usbmodemXXXX \\
      --robot.id=my_awesome_follower_arm \\
      --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
      --checkpoint_path=outputs/train/eval3-openvla-lora-new66-50k/checkpoints/040000 \\
      --openvla_src=external/openvla \\
      --task="Place the coke on Taylor Swift" \\
      --dry_run=true

Guarded 3s hardware smoke test::

    OPENVLA_SRC=external/openvla python scripts/eval3_openvla_deploy.py \\
      ... \\
      --episode_time_s=3 --fps=5 --motion_gain=0.25 \\
      --action_smoothing_alpha=0.35 --max_action_delta_deg=4 \\
      --allow_live_motors=true
"""

# NB: deliberately NO `from __future__ import annotations`. lerobot's
# parser.wrap() reads the config dataclass via inspect.getfullargspec().
# annotations — under PEP 563 lazy annotations that yields the bare string
# "Eval3OpenVLADeployConfig" instead of the class, and draccus then fails
# with "must be called with a dataclass type or instance". Python 3.12
# supports the modern annotation syntax natively without the future import.

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OPENVLA_ROOT = ROOT / "integrations" / "openvla"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_ROOT))

from eval3_lerobot_shim import apply as _eval3_shim_apply

_eval3_shim_apply()

# OpenVLA pins transformers==4.40.1; lerobot's policies/__init__.py eagerly
# imports pi05/smolvla/... which need much newer transformers. The OpenVLA
# deploy uses only lerobot's robot/camera/control utilities (never a lerobot
# policy), so we pre-stub lerobot.policies to skip that import cascade.
# MUST run before any `from lerobot...` import below.
from eval3_openvla_lerobot_compat import apply as _eval3_openvla_lr_apply  # noqa: E402

_eval3_openvla_lr_apply()

import torch  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from lerobot.configs import parser  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401, E402
from lerobot.utils.control_utils import init_keyboard_listener  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging, log_say  # noqa: E402

from eval3_external_vla_data import (  # noqa: E402
    ACTION_NAMES_6,
    IMAGE_KEY,
    STATE_KEY,
    canonicalize_task,
)
from openvla_eval3.checkpoint import (  # noqa: E402
    load_eval3_openvla,
    reshape_eval3_action,
)
from openvla_eval3.inference import (  # noqa: E402
    build_inputs_from_image,
    pick_device,
    pick_dtype,
    predict_action_numpy,
)
from openvla_eval3.mapping import load_embodiment_stats, openvla_action_to_so101
from openvla_eval3.prompts import wrap_openvla_prompt


DEFAULT_CHECKPOINT = "outputs/train/eval3-openvla-lora-new66-50k/checkpoints/040000"


@dataclass
class Eval3OpenVLADeployConfig:
    robot: RobotConfig
    checkpoint_path: str = DEFAULT_CHECKPOINT
    openvla_src: str = ""
    task: str = "Place the coke on Taylor Swift"
    target_slot: str = ""
    episode_time_s: float = 3.0
    fps: int = 5
    device: str = ""
    dtype: str = "auto"
    dry_run: bool = False
    allow_live_motors: bool = False
    play_sounds: bool = False
    rollout_log_dir: str = "outputs/eval3_openvla_rollouts"
    embodiment_stats: str = ""
    motion_gain: float = 0.25
    action_smoothing_alpha: float = 0.35
    max_action_delta_deg: float = 4.0
    gripper_open_bias_deg: float = 0.0
    gripper_open_bias_threshold_deg: float = 20.0
    offline_image: str = ""
    offline_state: str = ""

    def __post_init__(self) -> None:
        if self.target_slot.strip():
            slot = self.target_slot.strip().lower()
            if slot not in {"left", "middle", "right"}:
                raise ValueError("--target_slot must be one of left, middle, right.")
            self.task = f"Place the coke on the {slot} print"
        self.task = canonicalize_task(self.task)


def _ordered_action_keys(robot) -> list[str]:
    available = list(robot.action_features)
    if all(k in available for k in ACTION_NAMES_6):
        return list(ACTION_NAMES_6)
    if len(available) != 6:
        raise ValueError(f"Expected six SO-101 action keys, got {available}")
    logging.warning("Robot action key names differ from Eval3 schema; using robot order: %s", available)
    return available


def _parse_state_csv(value: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("--offline_state must contain six comma-separated joint values.")
    return vals


def _state_map(values: list[float], action_keys: list[str]) -> dict[str, float]:
    return {key: float(values[i]) for i, key in enumerate(action_keys)}


def _current_state_from_obs(obs_processed: dict[str, Any], action_keys: list[str]) -> dict[str, float]:
    if STATE_KEY not in obs_processed:
        raise KeyError(f"Missing {STATE_KEY}; keys={sorted(obs_processed)}")
    vals = torch.as_tensor(obs_processed[STATE_KEY], dtype=torch.float32).flatten().tolist()[:6]
    return _state_map([float(v) for v in vals], action_keys)


def _image_from_obs(obs_processed: dict[str, Any]):
    if IMAGE_KEY not in obs_processed:
        keys = [k for k in obs_processed if "image" in k]
        raise KeyError(f"Missing {IMAGE_KEY}; image keys={keys}")
    return obs_processed[IMAGE_KEY]


def _apply_guards(
    predicted: dict[str, float],
    current_state: dict[str, float],
    previous_action: dict[str, float] | None,
    *,
    motion_gain: float,
    smoothing_alpha: float,
    max_action_delta_deg: float,
    gripper_open_bias_deg: float,
    gripper_open_bias_threshold_deg: float,
) -> dict[str, float]:
    guarded: dict[str, float] = {}
    gain = min(max(float(motion_gain), 0.0), 1.0)
    for key, pred in predicted.items():
        target = float(pred)
        if key in current_state and gain < 1.0:
            target = float(current_state[key]) + gain * (target - float(current_state[key]))
        if "gripper" in key.lower() and gripper_open_bias_deg and target >= gripper_open_bias_threshold_deg:
            target += float(gripper_open_bias_deg)
        if previous_action is not None and key in previous_action and smoothing_alpha > 0:
            target = float(smoothing_alpha) * float(previous_action[key]) + (1.0 - float(smoothing_alpha)) * target
        if previous_action is not None and key in previous_action and max_action_delta_deg > 0:
            prev = float(previous_action[key])
            delta = float(max_action_delta_deg)
            target = min(max(target, prev - delta), prev + delta)
        guarded[key] = target
    return guarded


def _chunk_to_action_dict(
    chunk_row: np.ndarray,
    action_keys: list[str],
    *,
    embodiment_stats: dict[str, Any],
    motion_gain: float,
) -> dict[str, float]:
    raw = np.asarray(chunk_row, dtype=np.float64).reshape(-1)
    mapped = openvla_action_to_so101(raw, stats=embodiment_stats, motion_gain=motion_gain)
    if mapped:
        return {k: mapped[k] for k in action_keys if k in mapped}
    values = raw[: len(action_keys)].tolist()
    return {key: float(values[i]) for i, key in enumerate(action_keys)}


def _predict_chunk(
    bundle,
    *,
    image,
    task: str,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> np.ndarray:
    prompt = wrap_openvla_prompt(task)
    model_inputs = build_inputs_from_image(
        bundle.processor,
        prompt=prompt,
        image=image,
        device=device,
        torch_dtype=torch_dtype,
    )
    raw = predict_action_numpy(
        bundle.model,
        model_inputs,
        unnorm_key=bundle.unnorm_key,
        do_sample=False,
    )
    return reshape_eval3_action(raw, chunk_size=bundle.chunk_size)


def _save_first_frame(obs_processed: dict[str, Any], path: Path) -> None:
    try:
        from PIL import Image

        image = obs_processed.get(IMAGE_KEY)
        if image is None:
            return
        x = torch.as_tensor(image)
        if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
            x = x.permute(1, 2, 0)
        if x.dtype != torch.uint8:
            if float(x.max()) <= 2.0:
                x = x * 255.0
            x = x.round().clamp(0, 255).to(torch.uint8)
        Image.fromarray(x.detach().cpu().numpy()).save(path)
    except Exception as exc:
        logging.warning("Could not save first OpenVLA frame: %s", exc)


def _offline_probe(cfg: Eval3OpenVLADeployConfig, bundle, device: torch.device, torch_dtype: torch.dtype) -> None:
    if not cfg.offline_image:
        logging.info(
            "Dry run OK — OpenVLA checkpoint loaded (unnorm_key=%s, chunk_size=%s, dir=%s).",
            bundle.unnorm_key,
            bundle.chunk_size,
            bundle.checkpoint_dir,
        )
        return
    if not cfg.offline_state:
        raise ValueError("--offline_image requires --offline_state='six,comma,separated,joints'.")
    from PIL import Image

    image = Image.open(cfg.offline_image).convert("RGB")
    chunk = _predict_chunk(bundle, image=image, task=cfg.task, device=device, torch_dtype=torch_dtype)
    stats = load_embodiment_stats(Path(cfg.embodiment_stats) if cfg.embodiment_stats else None)
    print(
        json.dumps(
            {
                "task": cfg.task,
                "unnorm_key": bundle.unnorm_key,
                "chunk_size": bundle.chunk_size,
                "predicted_chunk_6dof": chunk.tolist(),
                "embodiment_mapping_used": bool(stats.get("mapping")),
            },
            indent=2,
        )
    )


def _deploy_loop(
    cfg: Eval3OpenVLADeployConfig,
    *,
    bundle,
    device: torch.device,
    torch_dtype: torch.dtype,
    embodiment_stats: dict[str, Any],
) -> None:
    _, robot_action_processor, robot_observation_processor = make_default_processors()
    robot = make_robot_from_config(cfg.robot)
    listener = None
    log_file = None
    try:
        robot.connect()
        action_keys = _ordered_action_keys(robot)
        listener, events = init_keyboard_listener()

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path(cfg.rollout_log_dir) / f"openvla_rollout_{ts}.jsonl" if cfg.rollout_log_dir else None
        first_frame_path = log_path.with_suffix(".firstframe.png") if log_path else None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(
                json.dumps(
                    {
                        "adapter": "eval3_openvla_deploy",
                        "checkpoint_path": cfg.checkpoint_path,
                        "resolved_checkpoint": str(bundle.checkpoint_dir),
                        "unnorm_key": bundle.unnorm_key,
                        "chunk_size": bundle.chunk_size,
                        "task": cfg.task,
                        "episode_time_s": cfg.episode_time_s,
                        "fps": cfg.fps,
                        "device": str(device),
                        "motion_gain": cfg.motion_gain,
                        "action_smoothing_alpha": cfg.action_smoothing_alpha,
                        "max_action_delta_deg": cfg.max_action_delta_deg,
                        "first_frame_path": str(first_frame_path),
                    }
                )
                + "\n"
            )
            log_file.flush()
            logging.info("OpenVLA rollout log: %s", log_path)

        log_say(f"Running OpenVLA for {cfg.episode_time_s}s", cfg.play_sounds)
        control_interval = 1.0 / max(int(cfg.fps), 1)
        predicted_chunk: np.ndarray | None = None
        chunk_index = 0
        previous_guarded: dict[str, float] | None = None
        step = 0
        start_t = time.perf_counter()

        while time.perf_counter() - start_t < cfg.episode_time_s:
            loop_t = time.perf_counter()
            if events["exit_early"]:
                events["exit_early"] = False
                break

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            if step == 0 and first_frame_path:
                _save_first_frame(obs_processed, first_frame_path)

            image = _image_from_obs(obs_processed)
            current_state = _current_state_from_obs(obs_processed, action_keys)

            ran_inference = False
            if predicted_chunk is None or chunk_index >= int(predicted_chunk.shape[0]):
                predicted_chunk = _predict_chunk(
                    bundle,
                    image=image,
                    task=cfg.task,
                    device=device,
                    torch_dtype=torch_dtype,
                )
                chunk_index = 0
                ran_inference = True

            raw_action = _chunk_to_action_dict(
                predicted_chunk[chunk_index],
                action_keys,
                embodiment_stats=embodiment_stats,
                motion_gain=1.0,
            )
            guarded_action = _apply_guards(
                raw_action,
                current_state,
                previous_guarded,
                motion_gain=cfg.motion_gain,
                smoothing_alpha=cfg.action_smoothing_alpha,
                max_action_delta_deg=cfg.max_action_delta_deg,
                gripper_open_bias_deg=cfg.gripper_open_bias_deg,
                gripper_open_bias_threshold_deg=cfg.gripper_open_bias_threshold_deg,
            )
            previous_guarded = guarded_action
            robot.send_action(robot_action_processor((guarded_action, obs)))

            if log_file:
                log_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "chunk_index": chunk_index,
                            "ran_inference": ran_inference,
                            "raw_action": raw_action,
                            "guarded_action": guarded_action,
                            "current_state": current_state,
                            "dt_s": time.perf_counter() - loop_t,
                        }
                    )
                    + "\n"
                )
                log_file.flush()

            chunk_index += 1
            step += 1
            sleep_s = control_interval - (time.perf_counter() - loop_t)
            if sleep_s < 0:
                logging.warning(
                    "OpenVLA loop slower than target FPS (%.2f Hz vs %s).",
                    1.0 / max(time.perf_counter() - loop_t, 1e-6),
                    cfg.fps,
                )
            precise_sleep(max(sleep_s, 0.0))

        log_say("OpenVLA episode finished", cfg.play_sounds)
    finally:
        if log_file:
            log_file.close()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()


@parser.wrap()
def eval3_openvla_deploy(cfg: Eval3OpenVLADeployConfig) -> None:
    register_third_party_plugins()
    init_logging()
    if not 0.0 <= cfg.action_smoothing_alpha < 1.0:
        raise ValueError("--action_smoothing_alpha must be in [0, 1).")
    if cfg.max_action_delta_deg < 0:
        raise ValueError("--max_action_delta_deg must be >= 0.")
    if not 0.0 < cfg.motion_gain <= 1.0:
        raise ValueError("--motion_gain must be in (0, 1].")
    if cfg.fps <= 0:
        raise ValueError("--fps must be positive.")

    device = pick_device(cfg.device)
    torch_dtype = pick_dtype(device, cfg.dtype)
    bundle = load_eval3_openvla(
        cfg.checkpoint_path,
        openvla_src=cfg.openvla_src,
        device=str(device),
        dtype=cfg.dtype,
    )
    embodiment_stats = load_embodiment_stats(Path(cfg.embodiment_stats) if cfg.embodiment_stats else None)

    if cfg.dry_run:
        _offline_probe(cfg, bundle, device, torch_dtype)
        return

    if not cfg.allow_live_motors:
        raise RuntimeError(
            "Live OpenVLA motor control is experimental. Re-run with --allow_live_motors=true "
            "after a dry run succeeds."
        )

    _deploy_loop(cfg, bundle=bundle, device=device, torch_dtype=torch_dtype, embodiment_stats=embodiment_stats)


def main() -> None:
    eval3_openvla_deploy()


if __name__ == "__main__":
    main()

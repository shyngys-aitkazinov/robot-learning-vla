#!/usr/bin/env python3
"""Experimental FlowerVLA closed-loop deploy adapter for Eval 3.

FlowerVLA is not a LeRobot ``PreTrainedPolicy`` checkpoint, so it cannot go
through ``eval3_vla_deploy.py`` directly. This adapter mirrors the FlowerVLA
training-time image/proprio packing, q01/q99 normalization, 7-D padded action
format, then strips the padded channel before sending SO-101 commands.

Start with ``--dry_run`` or a short ``--episode_time_s=3`` rollout. Live motor
control requires ``--allow_live_motors`` on purpose.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval3_lerobot_shim import apply as _eval3_shim_apply

_eval3_shim_apply()

# FlowerVLA's attention (external/flower_vla_calvin) passes both an explicit
# attn_mask and is_causal=True to F.scaled_dot_product_attention, which
# torch>=2.x rejects. This shim merges is_causal into the mask. Must run
# before the FlowerVLA model executes a forward pass.
from eval3_flower_sdpa_compat import apply as _eval3_flower_sdpa_apply  # noqa: E402

_eval3_flower_sdpa_apply()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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
    ACTION_NAMES_7,
    IMAGE_KEY,
    PADDED_DIM,
    STATE_KEY,
    canonicalize_task,
    denormalize_q01_q99,
    normalize_q01_q99,
)
from train_eval3_flower import (  # noqa: E402
    _normalize_flower_conditioning,
    build_flower_model,
)


DEFAULT_FLOWER_REPO = "RobotLearningVLA/eval3-flower-new66-50k"
DEFAULT_PRETRAINED_REPO = "mbreuss/flower_vla_pret"
DEFAULT_PRETRAINED_FILE = "360000_model_weights.pt"
DEFAULT_VLM_PATH = "microsoft/Florence-2-large"


@dataclass
class Eval3FlowerDeployConfig:
    robot: RobotConfig
    checkpoint_path: str = DEFAULT_FLOWER_REPO
    task: str = "Place the coke on Taylor Swift"
    target_slot: str = ""
    episode_time_s: float = 3.0
    fps: int = 5
    device: str = "auto"
    dry_run: bool = False
    allow_live_motors: bool = False
    play_sounds: bool = False
    rollout_log_dir: str = "outputs/eval3_flower_rollouts"

    # FlowerVLA source/model construction. Leave these empty to use checkpoint
    # metadata where possible.
    flower_src: str = ""
    pretrained_checkpoint: str = ""
    pretrained_repo: str = ""
    pretrained_file: str = ""
    vlm_path: str = ""
    chunk_size: int = 0
    image_size: int = 0
    num_sampling_steps: int = 0
    dit_dim: int = 0
    n_heads: int = 0
    n_layers: int = 0
    query_seq_len: int = 0

    # Safety guards. ``motion_gain`` blends from the observed current joint
    # state toward Flower's absolute target, which is safer than scaling target
    # angles toward zero.
    motion_gain: float = 0.25
    action_smoothing_alpha: float = 0.35
    max_action_delta_deg: float = 4.0
    gripper_open_bias_deg: float = 0.0
    gripper_open_bias_threshold_deg: float = 20.0

    # Optional offline probe without connecting hardware.
    offline_image: str = ""
    offline_state: str = ""

    def __post_init__(self) -> None:
        if self.target_slot.strip():
            slot = self.target_slot.strip().lower()
            if slot not in {"left", "middle", "right"}:
                raise ValueError("--target_slot must be one of left, middle, right.")
            self.task = f"Place the coke on the {slot} print"
        self.task = canonicalize_task(self.task)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _hf_download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename))


def _resolve_checkpoint(checkpoint_path: str) -> tuple[Path, Path | None, Path | None]:
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        base = path.parent
        return path, base / "train_config.json", base / "dataset_statistics.json"
    if path.is_dir():
        direct = path / "checkpoint.pt"
        if direct.is_file():
            return direct, path / "train_config.json", path / "dataset_statistics.json"
        candidates = sorted(path.glob("checkpoints/*/checkpoint.pt"))
        if candidates:
            ckpt = candidates[-1]
            return ckpt, ckpt.parent / "train_config.json", ckpt.parent / "dataset_statistics.json"
        raise FileNotFoundError(f"No checkpoint.pt found under {path}")

    ckpt = _hf_download(checkpoint_path, "checkpoint.pt")
    train_cfg = None
    stats_path = None
    try:
        train_cfg = _hf_download(checkpoint_path, "train_config.json")
    except Exception as exc:
        logging.warning("No train_config.json in %s: %s", checkpoint_path, exc)
    try:
        stats_path = _hf_download(checkpoint_path, "dataset_statistics.json")
    except Exception as exc:
        logging.warning("No dataset_statistics.json in %s: %s", checkpoint_path, exc)
    return ckpt, train_cfg, stats_path


def _unwrap_stats(payload: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    stats = checkpoint.get("dataset_statistics")
    if isinstance(stats, dict) and "action" in stats and "proprio" in stats:
        return stats

    if isinstance(payload, dict):
        if "action" in payload and "proprio" in payload:
            return payload
        for value in payload.values():
            if isinstance(value, dict) and "action" in value and "proprio" in value:
                return value

    raise ValueError("Could not find FlowerVLA dataset statistics with action/proprio q01/q99.")


def _pick_int(cfg_value: int, train_config: dict[str, Any], key: str, default: int) -> int:
    if cfg_value:
        return int(cfg_value)
    return int(train_config.get(key, default))


def _pick_str(cfg_value: str, train_config: dict[str, Any], key: str, default: str) -> str:
    if cfg_value:
        return cfg_value
    value = train_config.get(key, default)
    return str(value) if value is not None else default


def _make_flower_args(cfg: Eval3FlowerDeployConfig, train_config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        flower_src=cfg.flower_src,
        pretrained_checkpoint=_pick_str(
            cfg.pretrained_checkpoint, train_config, "pretrained_checkpoint", ""
        ),
        pretrained_repo=_pick_str(cfg.pretrained_repo, train_config, "pretrained_repo", DEFAULT_PRETRAINED_REPO),
        pretrained_file=_pick_str(cfg.pretrained_file, train_config, "pretrained_file", DEFAULT_PRETRAINED_FILE),
        vlm_path=_pick_str(cfg.vlm_path, train_config, "vlm_path", DEFAULT_VLM_PATH),
        chunk_size=_pick_int(cfg.chunk_size, train_config, "chunk_size", 10),
        image_size=_pick_int(cfg.image_size, train_config, "image_size", 224),
        num_sampling_steps=_pick_int(cfg.num_sampling_steps, train_config, "num_sampling_steps", 4),
        dit_dim=_pick_int(cfg.dit_dim, train_config, "dit_dim", 1024),
        n_heads=_pick_int(cfg.n_heads, train_config, "n_heads", 16),
        n_layers=_pick_int(cfg.n_layers, train_config, "n_layers", 18),
        query_seq_len=_pick_int(cfg.query_seq_len, train_config, "query_seq_len", 100),
    )


def _select_device(name: str) -> torch.device:
    key = str(name).strip().lower()
    if key == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(key)


def _load_model_and_stats(cfg: Eval3FlowerDeployConfig):
    ckpt_path, train_config_path, stats_path = _resolve_checkpoint(cfg.checkpoint_path)
    checkpoint = _torch_load(ckpt_path)

    train_config = checkpoint.get("train_config") if isinstance(checkpoint.get("train_config"), dict) else {}
    if train_config_path and train_config_path.is_file():
        train_config = {**_read_json(train_config_path), **train_config}

    stats_payload = _read_json(stats_path) if stats_path and stats_path.is_file() else {}
    stats = _unwrap_stats(stats_payload, checkpoint)
    flower_args = _make_flower_args(cfg, train_config)
    device = _select_device(cfg.device)

    logging.info("Loading FlowerVLA checkpoint=%s device=%s", ckpt_path, device)
    model = build_flower_model(flower_args, device)
    state = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.warning("FlowerVLA load missing keys: %s", missing[:20])
    if unexpected:
        logging.warning("FlowerVLA load unexpected keys: %s", unexpected[:20])
    model.eval()
    return model, stats, flower_args, device, ckpt_path


def _to_chw_float(image: Any, image_size: int) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        from PIL import Image

        img = Image.open(image).convert("RGB")
        x = torch.as_tensor(list(img.getdata()), dtype=torch.uint8).reshape(img.height, img.width, 3)
    else:
        x = torch.as_tensor(image)
    if x.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {tuple(x.shape)}")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        chw = x
    else:
        chw = x.permute(2, 0, 1)
    chw = chw.to(dtype=torch.float32)
    if float(chw.max()) > 2.0:
        chw = chw / 255.0
    if image_size and (chw.shape[-2] != image_size or chw.shape[-1] != image_size):
        chw = F.interpolate(
            chw.unsqueeze(0), size=(int(image_size), int(image_size)), mode="bilinear", align_corners=False
        )[0]
    return chw.clamp(0.0, 1.0)


def _parse_state_csv(value: str) -> torch.Tensor:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != 6:
        raise ValueError("--offline_state must contain six comma-separated SO-101 joint values.")
    return torch.tensor(vals, dtype=torch.float32)


def _pad_7(x: torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32).flatten()
    if x.numel() > PADDED_DIM:
        raise ValueError(f"Expected <= {PADDED_DIM} values, got {x.numel()}")
    if x.numel() == PADDED_DIM:
        return x
    return torch.cat([x, torch.zeros(PADDED_DIM - x.numel(), dtype=x.dtype, device=x.device)], dim=0)


def _state_from_observation(obs_processed: dict[str, Any]) -> torch.Tensor:
    # Dataset-format key is preferred; fall back to assembling from raw per-joint
    # `<motor>.pos` keys (what so_follower.get_observation() actually returns).
    if STATE_KEY in obs_processed:
        return _pad_7(torch.as_tensor(obs_processed[STATE_KEY], dtype=torch.float32))
    # ACTION_NAMES_6 already include the ".pos" suffix
    if all(k in obs_processed for k in ACTION_NAMES_6):
        vec = torch.tensor(
            [float(obs_processed[k]) for k in ACTION_NAMES_6], dtype=torch.float32,
        )
        return _pad_7(vec)
    raise KeyError(
        f"Robot observation is missing {STATE_KEY} and per-joint {list(ACTION_NAMES_6)}; "
        f"available keys={sorted(obs_processed)}"
    )


def _image_from_observation(obs_processed: dict[str, Any], image_size: int) -> torch.Tensor:
    # Dataset-format key is preferred; fall back to the raw camera key
    # ('front' for --robot.cameras='{front: ...}') which is what
    # so_follower.get_observation() actually returns.
    img = obs_processed.get(IMAGE_KEY)
    bare = IMAGE_KEY.removeprefix("observation.images.")
    if img is None:
        img = obs_processed.get(bare)
    if img is None:
        raise KeyError(
            f"Robot observation is missing {IMAGE_KEY!r} or bare {bare!r}; "
            f"available keys={sorted(obs_processed)}"
        )
    return _to_chw_float(img, image_size=image_size)


def _flower_batch(
    *,
    image_chw: torch.Tensor,
    state_7: torch.Tensor,
    task: str,
    stats: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    images = image_chw.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    states = state_7.to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    states = normalize_q01_q99(states, {"action": stats["proprio"]})
    zero_gripper_view = torch.zeros_like(images)
    return {
        "rgb_obs": {
            "rgb_static": images,
            "rgb_gripper": zero_gripper_view,
        },
        "obs": {
            "proprio": states,
            "robot_obs": states,
        },
        "robot_obs": states,
        "lang_text": [task],
    }


def _predict_chunk(
    model,
    *,
    image_chw: torch.Tensor,
    state_7: torch.Tensor,
    task: str,
    stats: dict[str, Any],
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    batch = _flower_batch(image_chw=image_chw, state_7=state_7, task=task, stats=stats, device=device)
    with torch.no_grad():
        features = model.encode_observations(batch)
        # ``rf_loss`` adds these during training. Sampling needs the same shape
        # normalization when upstream Flower returns scalar action_type fields.
        pseudo_actions = torch.zeros(1, chunk_size, PADDED_DIM, device=device, dtype=torch.float32)
        _normalize_flower_conditioning(features, {"actions": pseudo_actions})
        noise = torch.randn(1, chunk_size, PADDED_DIM, device=device, dtype=torch.float32)
        norm_actions = model.sample_actions(noise, features, inference=True)
        raw_actions = denormalize_q01_q99(norm_actions.detach().cpu(), stats)
    return raw_actions[0, :, :6].to(dtype=torch.float32)


def _as_float_list(x: Any) -> list[float]:
    return [float(v) for v in torch.as_tensor(x, dtype=torch.float32).flatten().tolist()]


def _ordered_action_keys(robot) -> list[str]:
    available = list(robot.action_features)
    if all(k in available for k in ACTION_NAMES_6):
        return list(ACTION_NAMES_6)
    if len(available) != 6:
        raise ValueError(f"Expected six SO-101 action keys, got {available}")
    logging.warning("Robot action key names differ from Eval3 schema; using robot order: %s", available)
    return available


def _state_map(state_7: torch.Tensor, action_keys: list[str]) -> dict[str, float]:
    values = _as_float_list(state_7[:6])
    return {key: values[i] for i, key in enumerate(action_keys)}


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


def _save_first_frame(obs_processed: dict[str, Any], path: Path) -> None:
    try:
        from PIL import Image

        image = obs_processed.get(IMAGE_KEY)
        if image is None:
            # raw-robot fallback: bare camera key like 'front'
            image = obs_processed.get(IMAGE_KEY.removeprefix("observation.images."))
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
        logging.warning("Could not save first FlowerVLA frame: %s", exc)


def _write_log_header(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8")
    handle.write(json.dumps(payload) + "\n")
    handle.flush()
    return handle


def _offline_probe(cfg: Eval3FlowerDeployConfig, model, stats: dict[str, Any], flower_args, device: torch.device) -> None:
    if not cfg.offline_image:
        logging.info("Dry run OK: FlowerVLA checkpoint and statistics loaded; no offline image supplied.")
        return
    if not cfg.offline_state:
        raise ValueError("--offline_image requires --offline_state='six,comma,separated,joints'.")
    image = _to_chw_float(cfg.offline_image, image_size=flower_args.image_size)
    state = _pad_7(_parse_state_csv(cfg.offline_state))
    chunk = _predict_chunk(
        model,
        image_chw=image,
        state_7=state,
        task=cfg.task,
        stats=stats,
        device=device,
        chunk_size=flower_args.chunk_size,
    )
    print(json.dumps({"task": cfg.task, "action_names": list(ACTION_NAMES_6), "predicted_chunk": chunk.tolist()}, indent=2))


def _deploy_loop(
    cfg: Eval3FlowerDeployConfig,
    *,
    model,
    stats: dict[str, Any],
    flower_args,
    device: torch.device,
    ckpt_path: Path,
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
        log_path = Path(cfg.rollout_log_dir) / f"flower_rollout_{ts}.jsonl" if cfg.rollout_log_dir else None
        first_frame_path = log_path.with_suffix(".firstframe.png") if log_path else None
        if log_path:
            log_file = _write_log_header(
                log_path,
                {
                    "adapter": "eval3_flower_deploy",
                    "checkpoint_path": cfg.checkpoint_path,
                    "resolved_checkpoint": str(ckpt_path),
                    "task": cfg.task,
                    "episode_time_s": cfg.episode_time_s,
                    "fps": cfg.fps,
                    "device": str(device),
                    "chunk_size": flower_args.chunk_size,
                    "image_size": flower_args.image_size,
                    "action_names_7": list(ACTION_NAMES_7),
                    "sent_action_names": action_keys,
                    "motion_gain": cfg.motion_gain,
                    "action_smoothing_alpha": cfg.action_smoothing_alpha,
                    "max_action_delta_deg": cfg.max_action_delta_deg,
                    "first_frame_path": str(first_frame_path),
                },
            )
            logging.info("FlowerVLA rollout log will be written to %s", log_path)

        log_say(f"Running FlowerVLA for {cfg.episode_time_s}s", cfg.play_sounds)
        control_interval = 1.0 / max(int(cfg.fps), 1)
        predicted_chunk: torch.Tensor | None = None
        chunk_index = 0
        previous_guarded_action: dict[str, float] | None = None
        step = 0
        start_t = time.perf_counter()

        # NOTE on smoothness: FlowerVLA re-infers a fresh action chunk every
        # `chunk_size` steps, and each inference (Florence-2 backbone) takes
        # ~1-2 s. Async prefetch was considered but does NOT work here: with
        # chunk_size=10 and inference ≈ one chunk-duration, a prefetched chunk
        # is conditioned on a ~9-step-stale state, so its early actions point
        # backward and jerk the arm at chunk boundaries. The synchronous
        # re-conditioning below is the correct receding-horizon behavior.
        # The per-chunk pause is inherent to the slow VLM backbone; lower
        # `num_sampling_steps` (DiT integration steps) trims it modestly.

        while time.perf_counter() - start_t < cfg.episode_time_s:
            loop_t = time.perf_counter()
            if events["exit_early"]:
                events["exit_early"] = False
                break

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            if step == 0 and first_frame_path:
                _save_first_frame(obs_processed, first_frame_path)

            image = _image_from_observation(obs_processed, image_size=flower_args.image_size)
            state_7 = _state_from_observation(obs_processed)
            current_state = _state_map(state_7, action_keys)

            ran_inference = False
            if predicted_chunk is None or chunk_index >= int(predicted_chunk.shape[0]):
                predicted_chunk = _predict_chunk(
                    model,
                    image_chw=image,
                    state_7=state_7,
                    task=cfg.task,
                    stats=stats,
                    device=device,
                    chunk_size=flower_args.chunk_size,
                )
                chunk_index = 0
                ran_inference = True

            raw_values = _as_float_list(predicted_chunk[chunk_index])
            raw_action = {key: raw_values[i] for i, key in enumerate(action_keys)}
            guarded_action = _apply_guards(
                raw_action,
                current_state,
                previous_guarded_action,
                motion_gain=cfg.motion_gain,
                smoothing_alpha=cfg.action_smoothing_alpha,
                max_action_delta_deg=cfg.max_action_delta_deg,
                gripper_open_bias_deg=cfg.gripper_open_bias_deg,
                gripper_open_bias_threshold_deg=cfg.gripper_open_bias_threshold_deg,
            )
            previous_guarded_action = guarded_action
            robot_action_to_send = robot_action_processor((guarded_action, obs))
            robot.send_action(robot_action_to_send)

            dt = time.perf_counter() - loop_t
            if log_file:
                log_file.write(
                    json.dumps(
                        {
                            "step": step,
                            "t_episode_s": time.perf_counter() - start_t,
                            "dt_s": float(dt),
                            "loop_hz": float(1.0 / dt) if dt > 0 else None,
                            "ran_policy_inference": ran_inference,
                            "chunk_index": chunk_index,
                            "state": _as_float_list(state_7[:6]),
                            "policy_action_raw": raw_values,
                            "policy_action_guarded": [float(guarded_action[k]) for k in action_keys],
                            "sent_action": {k: float(robot_action_to_send[k]) for k in sorted(robot_action_to_send)},
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
                    "FlowerVLA loop slower than target FPS (%.2f Hz vs target %s).",
                    1.0 / max(time.perf_counter() - loop_t, 1e-6),
                    cfg.fps,
                )
            precise_sleep(max(sleep_s, 0.0))

        log_say("FlowerVLA episode finished", cfg.play_sounds)
    finally:
        if log_file:
            log_file.close()
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()


@parser.wrap()
def eval3_flower_deploy(cfg: Eval3FlowerDeployConfig) -> None:
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

    model, stats, flower_args, device, ckpt_path = _load_model_and_stats(cfg)
    if cfg.dry_run:
        _offline_probe(cfg, model, stats, flower_args, device)
        return

    if not cfg.allow_live_motors:
        raise RuntimeError(
            "Live FlowerVLA motor control is experimental. Re-run with --allow_live_motors=true "
            "after a dry run succeeds."
        )

    _deploy_loop(cfg, model=model, stats=stats, flower_args=flower_args, device=device, ckpt_path=ckpt_path)


def main() -> None:
    eval3_flower_deploy()


if __name__ == "__main__":
    main()

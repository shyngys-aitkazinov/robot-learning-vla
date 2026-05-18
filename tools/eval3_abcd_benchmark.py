#!/usr/bin/env python3
"""Offline A/B/C/D checkpoint benchmark for Eval 3.

The script ranks only the canonical A/B/C/D model checkpoints and writes:
  - model_manifest.json
  - offline_scores.json
  - OFFLINE_REPORT.md

It is intentionally hardware-free. Use the shortlist it produces for the
real-robot trials described in docs/eval3/abcd_model_eval.md.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import fields
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402


PROMPTS = {
    "swift": "Place the coke on Taylor Swift",
    "lecun": "Place the coke on Yann LeCun",
    "obama": "Place the coke on Barack Obama",
}
PROBE_DATASETS = {
    "swift": "RobotLearningVLA/taylor_swift_1",
    "lecun": "RobotLearningVLA/yann_lecun_1",
    "obama": "RobotLearningVLA/barack_obama_1",
}
RENAME_MAP = {"observation.images.front": "observation.images.camera1"}
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
EXPECTED_FILES = {
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
}


def abcd_model_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for step in (10, 20, 30, 40, 50):
        specs.append(
            {
                "label": f"A-{step}k",
                "category": "A",
                "checkpoint": f"{step}k",
                "repo_id": f"RobotLearningVLA/eval3-vla-v7-A-smolvla-new-{step}k",
            }
        )
    for step in (10, 20, 30, 40, 50):
        specs.append(
            {
                "label": f"B-{step}k",
                "category": "B",
                "checkpoint": f"{step}k",
                "repo_id": f"RobotLearningVLA/eval3-vla-v7-B-smolvla-new-old-{step}k",
            }
        )
    for step in (3, 6, 9, 12):
        specs.append(
            {
                "label": f"C-{step}k",
                "category": "C",
                "checkpoint": f"{step}k",
                "repo_id": f"RobotLearningVLA/eval3-vla-v7-C-warm-v3-{step}k",
            }
        )
    for step in (2500, 5000, 7500, 10000):
        specs.append(
            {
                "label": f"D-{step}",
                "category": "D",
                "checkpoint": str(step),
                "repo_id": f"RobotLearningVLA/eval3-vla-v7-D-obama-only-{step}",
            }
        )
    return specs


def v8_model_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for step in (10, 20, 30, 40, 50):
        specs.append(
            {
                "label": f"V8-{step}k",
                "category": "V8",
                "checkpoint": f"{step}k",
                "repo_id": f"RobotLearningVLA/eval3-vla-v8-gripper-repair-smooth-{step}k",
            }
        )
    return specs


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
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


def episode_bounds(ds: LeRobotDataset, episode: int = 0) -> tuple[int, int]:
    row = episode_row(ds.meta.episodes, int(episode))
    return int(scalar(row["dataset_from_index"])), int(scalar(row["dataset_to_index"]))


def sample_indices(ds: LeRobotDataset, n: int) -> list[int]:
    if n <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, len(ds) - 1, n)))


def final_phase_indices(ds: LeRobotDataset, n_episodes: int = 3, frac: float = 0.95) -> list[int]:
    out: list[int] = []
    total_eps = int(getattr(ds, "num_episodes", 1))
    for ep in range(min(total_eps, n_episodes)):
        f0, f1 = episode_bounds(ds, ep)
        if f1 <= f0:
            continue
        out.append(min(f1 - 1, f0 + int((f1 - f0 - 1) * frac)))
    return out


def sequence_indices(ds: LeRobotDataset, window: int) -> list[int]:
    f0, f1 = episode_bounds(ds, 0)
    if f1 <= f0:
        return [0]
    start = f0 + max(0, int((f1 - f0) * 0.25))
    end = min(f1, start + max(2, window))
    return list(range(start, end))


def pretrained_file(policy_path: str, filename: str) -> str | None:
    local = Path(policy_path)
    if local.exists():
        candidate = local / filename
        return str(candidate) if candidate.is_file() else None
    try:
        return hf_hub_download(policy_path, filename)
    except Exception:
        return None


def load_json_file(policy_path: str, filename: str) -> dict[str, Any] | None:
    path = pretrained_file(policy_path, filename)
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def restore_features(raw_features: dict | None) -> dict[str, PolicyFeature]:
    out = {}
    for key, value in (raw_features or {}).items():
        out[key] = PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
    return out


def load_policy_bundle(policy_path: str, device: str, dataset_repo_for_stats: str):
    cfg_path = pretrained_file(policy_path, "config.json")
    if not cfg_path:
        raise FileNotFoundError(f"Could not resolve config.json for {policy_path}")
    raw_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    raw_cfg.pop("type", None)

    valid_keys = {f.name for f in fields(SmolVLAConfig)}
    init_kwargs = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    policy_cfg = SmolVLAConfig(**init_kwargs)
    policy_cfg.input_features = restore_features(raw_cfg.get("input_features"))
    policy_cfg.output_features = restore_features(raw_cfg.get("output_features"))
    policy_cfg.device = device
    policy_cfg.push_to_hub = False
    policy_cfg.compile_model = False
    policy_cfg.pretrained_path = policy_path

    ds_meta = LeRobotDatasetMetadata(dataset_repo_for_stats)
    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=RENAME_MAP)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=rename_stats(ds_meta.stats, RENAME_MAP),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
    )
    return policy, preprocessor, postprocessor, get_safe_torch_device(policy.config.device)


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def apply_camera_variant(image: np.ndarray, variant: str) -> np.ndarray:
    if variant == "original":
        return image
    from PIL import Image, ImageEnhance, ImageFilter

    pil = Image.fromarray(image)
    if variant == "dark":
        return np.asarray(ImageEnhance.Brightness(pil).enhance(0.65))
    if variant == "bright":
        return np.asarray(ImageEnhance.Brightness(pil).enhance(1.35))
    if variant == "center_crop":
        w, h = pil.size
        crop = pil.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)))
        return np.asarray(crop.resize((w, h), Image.BILINEAR))
    if variant == "blur":
        return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.2)))
    raise ValueError(f"unknown camera variant: {variant}")


def row_to_observation(row: dict, image_variant: str = "original") -> dict[str, Any]:
    arr = image_to_hwc_uint8(row["observation.images.front"])
    arr = apply_camera_variant(arr, image_variant)
    state = row["observation.state"]
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    else:
        state = np.asarray(state)
    return {
        "observation.images.front": arr,
        "observation.state": state,
    }


def predict(bundle, row: dict, prompt: str, image_variant: str = "original") -> np.ndarray:
    policy, preprocessor, postprocessor, device = bundle
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    with torch.no_grad():
        action = predict_action(
            observation=row_to_observation(row, image_variant=image_variant),
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=prompt,
        )
    arr = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    if arr.ndim >= 2:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32).reshape(-1)[:6]


def model_info_record(api: HfApi, spec: dict[str, Any]) -> dict[str, Any]:
    repo_id = spec["repo_id"]
    record = dict(spec)
    try:
        info = api.model_info(repo_id, files_metadata=True)
        files = {s.rfilename for s in (info.siblings or [])}
        record.update(
            {
                "exists": True,
                "commit_sha": info.sha,
                "missing_files": sorted(EXPECTED_FILES - files),
                "has_model_safetensors": "model.safetensors" in files,
                "all_files": sorted(files),
            }
        )
    except Exception as exc:
        record.update(
            {
                "exists": False,
                "commit_sha": None,
                "missing_files": sorted(EXPECTED_FILES),
                "has_model_safetensors": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return record

    train_cfg = load_json_file(repo_id, "train_config.json") or {}
    policy_cfg = load_json_file(repo_id, "config.json") or {}
    preproc_path = pretrained_file(repo_id, "policy_preprocessor_step_5_normalizer_processor.safetensors")
    stats_count = {"action": None, "observation.state": None}
    if preproc_path:
        try:
            from safetensors.torch import load_file

            tensors = load_file(preproc_path)
            if "action.count" in tensors:
                stats_count["action"] = float(tensors["action.count"].flatten()[0])
            if "observation.state.count" in tensors:
                stats_count["observation.state"] = float(tensors["observation.state.count"].flatten()[0])
        except Exception as exc:
            stats_count["error"] = f"{type(exc).__name__}: {exc}"

    record["stats_count"] = stats_count
    record["train_config"] = {
        "job_name": train_cfg.get("job_name"),
        "steps": train_cfg.get("steps"),
        "batch_size": train_cfg.get("batch_size"),
        "dataset_repo_id": (train_cfg.get("dataset") or {}).get("repo_id"),
        "policy_pretrained_path": (train_cfg.get("policy") or {}).get("pretrained_path"),
        "policy_empty_cameras": (train_cfg.get("policy") or {}).get("empty_cameras"),
        "policy_n_action_steps": (train_cfg.get("policy") or {}).get("n_action_steps"),
        "rename_map": train_cfg.get("rename_map"),
    }
    record["policy_config"] = {
        "empty_cameras": policy_cfg.get("empty_cameras"),
        "n_action_steps": policy_cfg.get("n_action_steps"),
        "chunk_size": policy_cfg.get("chunk_size"),
        "input_features": sorted((policy_cfg.get("input_features") or {}).keys()),
    }
    return record


def score_from_mae(mae: float, fail_at: float = 12.0) -> float:
    return max(0.0, min(1.0, 1.0 - mae / fail_at))


def score_from_minimum(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return max(0.0, min(1.0, value / target))


def evaluate_model(
    spec: dict[str, Any],
    *,
    device: str,
    datasets: dict[str, LeRobotDataset],
    sample_map: dict[str, list[int]],
    final_map: dict[str, list[int]],
    sequence_map: dict[str, list[int]],
    dataset_repo_for_stats: str,
    camera_variants: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": spec["label"],
        "category": spec["category"],
        "checkpoint": spec["checkpoint"],
        "repo_id": spec["repo_id"],
        "metrics": {},
        "failures": [],
    }
    bundle = load_policy_bundle(spec["repo_id"], device, dataset_repo_for_stats)

    matched_errs = []
    per_joint_errs = []
    prompt_pair_values: dict[str, list[float]] = {
        f"{a}-{b}": [] for a, b in combinations(PROMPTS, 2)
    }
    camera_pair_values: dict[str, list[float]] = {variant: [] for variant in camera_variants}
    final_errs = []
    gripper_preds = []
    smooth_deltas = []
    smooth_jerks = []

    for src, ds in datasets.items():
        for idx in sample_map[src]:
            row = ds[idx]
            gt = row["action"].detach().cpu().numpy() if isinstance(row["action"], torch.Tensor) else np.asarray(row["action"])
            pred = predict(bundle, row, PROMPTS[src])
            err = np.abs(pred - np.asarray(gt, dtype=np.float32).reshape(-1)[:6])
            matched_errs.append(float(err.mean()))
            per_joint_errs.append(err)

            actions = {name: predict(bundle, row, prompt) for name, prompt in PROMPTS.items()}
            for a, b in combinations(PROMPTS, 2):
                prompt_pair_values[f"{a}-{b}"].append(float(np.linalg.norm(actions[a] - actions[b])))

            for variant in camera_variants:
                variant_actions = {
                    name: predict(bundle, row, prompt, image_variant=variant) for name, prompt in PROMPTS.items()
                }
                pair_distances = [
                    float(np.linalg.norm(variant_actions[a] - variant_actions[b]))
                    for a, b in combinations(PROMPTS, 2)
                ]
                camera_pair_values[variant].append(float(min(pair_distances)))

        for idx in final_map[src]:
            row = ds[idx]
            gt = row["action"].detach().cpu().numpy() if isinstance(row["action"], torch.Tensor) else np.asarray(row["action"])
            pred = predict(bundle, row, PROMPTS[src])
            final_errs.append(float(np.abs(pred - np.asarray(gt, dtype=np.float32).reshape(-1)[:6]).mean()))
            gripper_preds.append(float(pred[5]))

        seq_preds = [predict(bundle, ds[idx], PROMPTS[src]) for idx in sequence_map[src]]
        if len(seq_preds) >= 2:
            diffs = np.diff(np.stack(seq_preds), axis=0)
            smooth_deltas.extend(np.linalg.norm(diffs, axis=1).tolist())
        if len(seq_preds) >= 3:
            jerks = np.diff(np.diff(np.stack(seq_preds), axis=0), axis=0)
            smooth_jerks.extend(np.linalg.norm(jerks, axis=1).tolist())

    per_joint = np.stack(per_joint_errs) if per_joint_errs else np.zeros((0, 6), dtype=np.float32)
    matched_mae = float(np.mean(matched_errs)) if matched_errs else math.inf
    per_joint_mae = {
        JOINT_NAMES[i]: float(per_joint[:, i].mean()) if len(per_joint) else None for i in range(len(JOINT_NAMES))
    }
    pair_means = {k: float(np.mean(v)) if v else 0.0 for k, v in prompt_pair_values.items()}
    min_prompt_pair = float(min(pair_means.values())) if pair_means else 0.0
    camera_variant_min = {
        k: float(np.mean(v)) if v else 0.0 for k, v in camera_pair_values.items()
    }
    min_camera_pair = float(min(camera_variant_min.values())) if camera_variant_min else 0.0
    final_mae = float(np.mean(final_errs)) if final_errs else math.inf
    gripper_q90_pred = float(np.quantile(gripper_preds, 0.90)) if gripper_preds else 0.0
    mean_delta = float(np.mean(smooth_deltas)) if smooth_deltas else 0.0
    p95_delta = float(np.quantile(smooth_deltas, 0.95)) if smooth_deltas else 0.0
    p95_jerk = float(np.quantile(smooth_jerks, 0.95)) if smooth_jerks else 0.0

    component_scores = {
        "matched_mae": score_from_mae(matched_mae, fail_at=12.0),
        "prompt_distinction": score_from_minimum(min_prompt_pair, target=30.0),
        "gripper_open": score_from_minimum(gripper_q90_pred, target=45.0),
        "smoothness": max(0.0, min(1.0, 1.0 - p95_delta / 80.0)),
        "completion_proxy": score_from_mae(final_mae, fail_at=15.0),
        "camera_robustness": score_from_minimum(min_camera_pair, target=20.0),
    }
    weights = {
        "matched_mae": 0.30,
        "prompt_distinction": 0.25,
        "gripper_open": 0.15,
        "smoothness": 0.15,
        "completion_proxy": 0.10,
        "camera_robustness": 0.05,
    }
    offline_score = float(sum(component_scores[k] * weights[k] for k in weights))

    if min_prompt_pair < 10.0:
        result["failures"].append(f"prompt distinction collapse: min pair L2 {min_prompt_pair:.2f} < 10")
    if gripper_q90_pred < 45.0:
        result["failures"].append(f"weak gripper opening: final q90 {gripper_q90_pred:.2f} < 45")
    if p95_delta > 80.0:
        result["failures"].append(f"shaky action deltas: p95 L2 {p95_delta:.2f} > 80")
    if min_camera_pair < 10.0:
        result["failures"].append(f"camera perturbation collapse: min pair L2 {min_camera_pair:.2f} < 10")

    result["metrics"] = {
        "offline_score": offline_score,
        "component_scores": component_scores,
        "weights": weights,
        "matched_mae": matched_mae,
        "per_joint_mae": per_joint_mae,
        "prompt_pair_l2_mean": pair_means,
        "min_prompt_pair_l2": min_prompt_pair,
        "camera_variant_min_pair_l2_mean": camera_variant_min,
        "min_camera_variant_pair_l2": min_camera_pair,
        "final_phase_mae": final_mae,
        "gripper_final_q90_pred": gripper_q90_pred,
        "action_delta_mean_l2": mean_delta,
        "action_delta_p95_l2": p95_delta,
        "action_jerk_p95_l2": p95_jerk,
    }
    return result


def select_shortlist(results: list[dict[str, Any]], max_models: int = 5) -> list[dict[str, Any]]:
    valid = [r for r in results if "metrics" in r and r["metrics"].get("offline_score") is not None]
    by_label = {r["label"]: r for r in valid}
    selected_labels: list[str] = []
    for category in ("A", "B", "C", "D"):
        group = [r for r in valid if r["category"] == category]
        if group:
            best = max(group, key=lambda r: r["metrics"]["offline_score"])
            selected_labels.append(best["label"])
    if valid:
        best_score = max(r["metrics"]["offline_score"] for r in valid)
        for r in sorted(valid, key=lambda row: row["metrics"]["offline_score"], reverse=True):
            if r["metrics"]["offline_score"] >= best_score * 0.95 and r["label"] not in selected_labels:
                selected_labels.append(r["label"])
            if len(selected_labels) >= max_models:
                break
    return [by_label[label] for label in selected_labels[:max_models] if label in by_label]


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('title', 'Eval 3 A/B/C/D Offline Report')}",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "## Ranked Models",
        "",
        "| rank | model | repo | score | matched MAE | min prompt L2 | gripper q90 | delta p95 | failures |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    ranked = sorted(report["results"], key=lambda r: r.get("metrics", {}).get("offline_score", -1), reverse=True)
    for i, row in enumerate(ranked, start=1):
        metrics = row.get("metrics", {})
        failures = "; ".join(row.get("failures", [])) or "none"
        lines.append(
            f"| {i} | {row['label']} | `{row['repo_id']}` | "
            f"{metrics.get('offline_score', 0):.3f} | {metrics.get('matched_mae', 0):.2f} | "
            f"{metrics.get('min_prompt_pair_l2', 0):.2f} | "
            f"{metrics.get('gripper_final_q90_pred', 0):.2f} | "
            f"{metrics.get('action_delta_p95_l2', 0):.2f} | {failures} |"
        )

    lines.extend(["", "## Shortlist", ""])
    for row in report["shortlist"]:
        lines.append(f"- `{row['label']}`: `{row['repo_id']}` score={row['metrics']['offline_score']:.3f}")

    lines.extend(
        [
            "",
            "## Hardware Stage 1 Command Template",
            "",
            "```bash",
            "python scripts/eval3_vla_deploy.py \\",
            "  --robot.type=so101_follower \\",
            "  --robot.port=<follower_tty> \\",
            "  --robot.id=my_awesome_follower_arm \\",
            "  --robot.cameras='{front: {type: opencv, index_or_path: <cam_idx>, width: 640, height: 480, fps: 30}}' \\",
            "  --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \\",
            "  --rename_map='{\"observation.images.front\":\"observation.images.camera1\"}' \\",
            "  --policy.path=<MODEL_REPO> \\",
            "  --policy.device=mps \\",
            "  --policy.n_action_steps=25 \\",
            "  --interpolation_multiplier=2 \\",
            "  --action_smoothing_alpha=0.25 \\",
            "  --episode_time_s=20 \\",
            "  --fps=30",
            "```",
            "",
            "## Notes",
            "",
            "- This report is an offline shortlist, not a replacement for real-robot scoring.",
            "- `D-12500` and duplicate D names are intentionally excluded.",
            "- The live winner should be chosen after Stage 1/Stage 2 hardware labels are merged with these scores.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Rank Eval 3 A/B/C/D checkpoints with offline probes.")
    ap.add_argument("--model-set", choices=("abcd", "v8"), default="abcd")
    ap.add_argument("--out-dir", default="outputs/eval3_abcd_eval")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--dataset-repo-for-stats", default="RobotLearningVLA/taylor_swift_1")
    ap.add_argument("--n-frames-per-dataset", type=int, default=4)
    ap.add_argument("--jerk-window-frames", type=int, default=12)
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--max-models", type=int, default=None, help="Debug/smoke: evaluate only first N models.")
    ap.add_argument(
        "--camera-variants",
        default="original,dark,bright,center_crop,blur",
        help="Comma-separated camera perturbations for robustness scoring.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    specs = v8_model_specs() if args.model_set == "v8" else abcd_model_specs()
    excluded = []
    title = "Eval 3 v8 Offline Report" if args.model_set == "v8" else "Eval 3 A/B/C/D Offline Report"
    if args.model_set == "abcd":
        excluded = [
            "RobotLearningVLA/eval3-vla-v7-D-obama-only-12500",
            "RobotLearningVLA/eval3-vla-v7-D-obama-only-2.5k",
            "RobotLearningVLA/eval3-vla-v7-D-obama-only-5k",
            "RobotLearningVLA/eval3-vla-v7-D-obama-only-7.5k",
            "RobotLearningVLA/eval3-vla-v7-D-obama-only-10k",
        ]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_set": args.model_set,
        "excluded": excluded,
        "models": [model_info_record(api, spec) for spec in specs],
    }
    (out_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    if args.manifest_only:
        print(f"wrote {out_dir / 'model_manifest.json'}")
        return 0

    eval_specs = [m for m in manifest["models"] if m.get("exists") and not m.get("missing_files")]
    if args.max_models is not None:
        eval_specs = eval_specs[: max(0, args.max_models)]
    if not eval_specs:
        raise RuntimeError(f"No complete {args.model_set} model repos available for evaluation.")

    datasets = {
        label: LeRobotDataset(repo, video_backend=args.video_backend) for label, repo in PROBE_DATASETS.items()
    }
    sample_map = {
        label: sample_indices(ds, args.n_frames_per_dataset) for label, ds in datasets.items()
    }
    final_map = {label: final_phase_indices(ds) for label, ds in datasets.items()}
    sequence_map = {label: sequence_indices(ds, args.jerk_window_frames) for label, ds in datasets.items()}
    camera_variants = [x.strip() for x in args.camera_variants.split(",") if x.strip()]

    results = []
    for spec in eval_specs:
        print(f">> evaluating {spec['label']}: {spec['repo_id']}", flush=True)
        try:
            results.append(
                evaluate_model(
                    spec,
                    device=args.device,
                    datasets=datasets,
                    sample_map=sample_map,
                    final_map=final_map,
                    sequence_map=sequence_map,
                    dataset_repo_for_stats=args.dataset_repo_for_stats,
                    camera_variants=camera_variants,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "label": spec["label"],
                    "category": spec["category"],
                    "checkpoint": spec["checkpoint"],
                    "repo_id": spec["repo_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "failures": [f"evaluation error: {type(exc).__name__}: {exc}"],
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    shortlist = select_shortlist(results)
    report = {
        "title": title,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {
            "device": args.device,
            "video_backend": args.video_backend,
            "n_frames_per_dataset": args.n_frames_per_dataset,
            "jerk_window_frames": args.jerk_window_frames,
            "dataset_repo_for_stats": args.dataset_repo_for_stats,
            "camera_variants": camera_variants,
            "sample_indices": sample_map,
            "final_phase_indices": final_map,
            "sequence_indices": sequence_map,
        },
        "results": results,
        "shortlist": shortlist,
    }
    (out_dir / "offline_scores.json").write_text(
        json.dumps(report, indent=2, default=json_default), encoding="utf-8"
    )
    (out_dir / "OFFLINE_REPORT.md").write_text(render_report(report), encoding="utf-8")
    print(f"wrote {out_dir / 'offline_scores.json'}")
    print(f"wrote {out_dir / 'OFFLINE_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

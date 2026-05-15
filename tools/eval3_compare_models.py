#!/usr/bin/env python3
"""Compare Eval 3 SmolVLA checkpoints on the same offline probes.

The report is intentionally diagnostic, not a replacement for hardware success:
it checks matched-prompt MAE, prompt-swap sensitivity, final-phase action targets,
known per-joint failures, and any saved live rollout summaries.
"""
from __future__ import annotations

import argparse
import json
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
DATASETS = {
    "swift": "RobotLearningVLA/taylor_swift_1",
    "lecun": "RobotLearningVLA/yann_lecun_1",
    "obama": "RobotLearningVLA/barack_obama_1",
}
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
RENAME_MAP = {"observation.images.front": "observation.images.camera1"}
DEFAULT_MODELS = {
    "v1_aug": "RobotLearningVLA/eval3-smolvla-3way-50k-aug-v1",
    "v3_fresh": "RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _model_config_path(policy_path: str) -> str:
    p = Path(policy_path)
    if p.exists():
        return str(p / "config.json")
    return hf_hub_download(policy_path, "config.json")


def _model_sha(policy_path: str) -> str | None:
    if Path(policy_path).exists():
        return None
    try:
        return HfApi().model_info(policy_path).sha
    except Exception:
        return None


def _restore_features(raw_features: dict | None) -> dict[str, PolicyFeature]:
    out = {}
    for key, value in (raw_features or {}).items():
        out[key] = PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
    return out


def _load_policy_bundle(policy_path: str, device: str, dataset_repo_for_stats: str):
    with open(_model_config_path(policy_path), encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg.pop("type", None)

    valid_keys = {f.name for f in fields(SmolVLAConfig)}
    init_kwargs = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    policy_cfg = SmolVLAConfig(**init_kwargs)
    policy_cfg.input_features = _restore_features(raw_cfg.get("input_features"))
    policy_cfg.output_features = _restore_features(raw_cfg.get("output_features"))
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
    device_obj = get_safe_torch_device(policy.config.device)
    return policy, preprocessor, postprocessor, device_obj


def _row_to_observation(row: dict) -> dict[str, Any]:
    img = row["observation.images.front"]
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = arr.clip(0, 255).astype(np.uint8)
    else:
        arr = np.asarray(img)

    state = row["observation.state"]
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    else:
        state = np.asarray(state)

    return {
        "observation.images.front": arr,
        "observation.state": state,
    }


def _predict_action(
    bundle,
    row: dict,
    prompt: str,
) -> np.ndarray:
    policy, preprocessor, postprocessor, device = bundle
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    with torch.no_grad():
        action = predict_action(
            observation=_row_to_observation(row),
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


def _sample_indices(ds: LeRobotDataset, n: int) -> list[int]:
    if n <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, len(ds) - 1, n)))


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1)[0].item()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _episode_row(episodes: Any, episode: int) -> Any:
    if hasattr(episodes, "iloc"):
        return episodes.iloc[int(episode)]
    return episodes[int(episode)]


def _final_phase_index(ds: LeRobotDataset, episode: int = 0, frac: float = 0.95) -> int:
    ep_df = ds.meta.episodes
    row = _episode_row(ep_df, int(episode))
    f0 = int(_scalar(row["dataset_from_index"]))
    f1 = int(_scalar(row["dataset_to_index"]))
    return min(f1 - 1, f0 + int((f1 - f0 - 1) * frac))


def _mae_summary(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    if not preds:
        return {"overall_mae": None, "per_joint_mae": None, "n_frames": 0}
    err = np.abs(np.stack(preds) - np.stack(gts))
    return {
        "overall_mae": float(err.mean()),
        "per_joint_mae": {joint: float(err[:, i].mean()) for i, joint in enumerate(JOINT_NAMES)},
        "n_frames": int(err.shape[0]),
    }


def _load_live_summaries() -> dict:
    out: dict[str, Any] = {}
    paths = {
        "legacy_live": _REPO / "outputs" / "eval3_live_analysis" / "live_rollout_summary.json",
        "v2_live": _REPO / "outputs" / "eval3_live_analysis_v2" / "live_rollout_summary.json",
    }
    for label, path in paths.items():
        if path.is_file():
            try:
                out[label] = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                out[label] = {"error": f"Could not parse {path}: {exc}"}
    return out


def evaluate_model(
    label: str,
    policy_path: str,
    *,
    device: str,
    datasets: dict[str, LeRobotDataset],
    sample_indices: dict[str, list[int]],
    dataset_repo_for_stats: str,
    fail_mae_deg: float,
    fail_prompt_l2_deg: float,
) -> dict:
    bundle = _load_policy_bundle(policy_path, device, dataset_repo_for_stats)
    result: dict[str, Any] = {
        "label": label,
        "policy_path": policy_path,
        "sha": _model_sha(policy_path),
        "matched_mae": {},
        "prompt_swap_l2": {},
        "final_phase": {},
        "failures": [],
    }

    for src, ds in datasets.items():
        preds = []
        gts = []
        swap_rows = []
        for idx in sample_indices[src]:
            row = ds[idx]
            gt = row["action"].detach().cpu().numpy() if isinstance(row["action"], torch.Tensor) else np.asarray(row["action"])
            pred = _predict_action(bundle, row, PROMPTS[src])
            preds.append(pred)
            gts.append(np.asarray(gt, dtype=np.float32).reshape(-1)[:6])
            swap_rows.append(row)

        matched = _mae_summary(preds, gts)
        result["matched_mae"][src] = matched
        for joint, mae in matched["per_joint_mae"].items():
            if mae > fail_mae_deg:
                result["failures"].append(
                    f"{src}: matched-prompt {joint} MAE {mae:.2f} deg > {fail_mae_deg:.2f}"
                )

        pair_values: dict[str, list[float]] = {
            f"{a}-{b}": [] for a, b in combinations(PROMPTS.keys(), 2)
        }
        for row in swap_rows:
            actions = {name: _predict_action(bundle, row, prompt) for name, prompt in PROMPTS.items()}
            for a, b in combinations(PROMPTS.keys(), 2):
                pair_values[f"{a}-{b}"].append(float(np.linalg.norm(actions[a] - actions[b])))

        pair_means = {k: float(np.mean(v)) for k, v in pair_values.items()}
        result["prompt_swap_l2"][src] = pair_means
        for pair, value in pair_means.items():
            if value < fail_prompt_l2_deg:
                result["failures"].append(
                    f"{src}: prompt-swap {pair} L2 {value:.2f} deg < {fail_prompt_l2_deg:.2f}"
                )

        final_row = ds[_final_phase_index(ds)]
        final_actions = {
            prompt_name: _predict_action(bundle, final_row, prompt).tolist()
            for prompt_name, prompt in PROMPTS.items()
        }
        result["final_phase"][src] = {
            "frame_index": int(final_row["index"]),
            "actions": final_actions,
            "wrist_roll": {k: float(v[4]) for k, v in final_actions.items()},
            "shoulder_lift": {k: float(v[1]) for k, v in final_actions.items()},
        }

    return result


def _render_markdown(report: dict) -> str:
    lines = [
        "# Eval 3 Model Comparison",
        "",
        f"Generated: `{report['generated_utc']}`",
        "",
        "> Randomized TOY print order is still not guaranteed without data where every identity appears in every target position. Current known target positions are Swift=middle, LeCun=right, Obama=right.",
        "",
        "## Models",
        "",
        "| label | policy | sha |",
        "|---|---|---|",
    ]
    for model in report["models"]:
        lines.append(f"| {model['label']} | `{model['policy_path']}` | `{model.get('sha') or 'local/unknown'}` |")

    lines.extend(["", "## Matched-Prompt MAE", "", "| model | source | overall | shoulder_lift | wrist_roll | gripper |", "|---|---:|---:|---:|---:|---:|"])
    for model in report["models"]:
        for src, summary in model["matched_mae"].items():
            pj = summary["per_joint_mae"]
            lines.append(
                f"| {model['label']} | {src} | {summary['overall_mae']:.2f} | "
                f"{pj['shoulder_lift']:.2f} | {pj['wrist_roll']:.2f} | {pj['gripper']:.2f} |"
            )

    lines.extend(["", "## Prompt-Swap Sensitivity", "", "| model | source image | swift-lecun | swift-obama | lecun-obama |", "|---|---:|---:|---:|---:|"])
    for model in report["models"]:
        for src, pairs in model["prompt_swap_l2"].items():
            lines.append(
                f"| {model['label']} | {src} | {pairs['swift-lecun']:.2f} | "
                f"{pairs['swift-obama']:.2f} | {pairs['lecun-obama']:.2f} |"
            )

    lines.extend(["", "## Final-Phase Wrist Roll", "", "| model | source frame | swift prompt | lecun prompt | obama prompt |", "|---|---:|---:|---:|---:|"])
    for model in report["models"]:
        for src, final in model["final_phase"].items():
            wr = final["wrist_roll"]
            lines.append(f"| {model['label']} | {src} | {wr['swift']:.1f} | {wr['lecun']:.1f} | {wr['obama']:.1f} |")

    lines.extend(["", "## Failures", ""])
    any_failures = False
    for model in report["models"]:
        if model["failures"]:
            any_failures = True
            lines.append(f"### {model['label']}")
            for failure in model["failures"]:
                lines.append(f"- {failure}")
            lines.append("")
    if not any_failures:
        lines.append("No threshold failures under the configured offline gates.")

    if report.get("live_rollouts"):
        lines.extend(["", "## Saved Live Rollout Summaries", ""])
        for label, rows in report["live_rollouts"].items():
            lines.append(f"### {label}")
            if isinstance(rows, list):
                for row in rows:
                    if "error" in row:
                        lines.append(f"- `{row.get('repo', 'unknown')}`: {row['error']}")
                    else:
                        wr = row.get("wr_final_1s", row.get("wr_final"))
                        sl = row.get("sl_final", row.get("live_shoulder_lift"))
                        lines.append(
                            f"- `{row.get('repo', 'unknown')}` celeb={row.get('celeb')} "
                            f"wr={wr} sl={sl} verdict={row.get('verdict', row.get('wr_correctness'))}"
                        )
            else:
                lines.append(f"- {rows}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="label=policy_path. May be repeated.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--n-frames-per-dataset", type=int, default=8)
    ap.add_argument("--dataset-repo-for-stats", default="RobotLearningVLA/taylor_swift_1")
    ap.add_argument("--out-dir", default="outputs/eval3_model_compare")
    ap.add_argument("--fail-mae-deg", type=float, default=5.0)
    ap.add_argument("--fail-prompt-l2-deg", type=float, default=10.0)
    ap.add_argument("--skip-live-rollouts", action="store_true")
    args = ap.parse_args()

    model_specs = dict(DEFAULT_MODELS)
    for raw in args.model:
        if "=" not in raw:
            raise ValueError("--model must be label=policy_path")
        label, path = raw.split("=", 1)
        model_specs[label.strip()] = path.strip()

    datasets = {
        label: LeRobotDataset(repo, video_backend=args.video_backend)
        for label, repo in DATASETS.items()
    }
    sample_indices = {
        label: _sample_indices(ds, args.n_frames_per_dataset)
        for label, ds in datasets.items()
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settings": {
            "device": args.device,
            "n_frames_per_dataset": args.n_frames_per_dataset,
            "dataset_repo_for_stats": args.dataset_repo_for_stats,
            "fail_mae_deg": args.fail_mae_deg,
            "fail_prompt_l2_deg": args.fail_prompt_l2_deg,
            "sample_indices": sample_indices,
        },
        "models": [],
        "live_rollouts": {} if args.skip_live_rollouts else _load_live_summaries(),
        "unsupported_layout_warning": (
            "Current data only has Swift target=middle, LeCun target=right, Obama target=right; "
            "left-target randomized TOY layouts are unsupported without new demonstrations."
        ),
    }

    for label, policy_path in model_specs.items():
        print(f">> evaluating {label}: {policy_path}", flush=True)
        report["models"].append(
            evaluate_model(
                label,
                policy_path,
                device=args.device,
                datasets=datasets,
                sample_indices=sample_indices,
                dataset_repo_for_stats=args.dataset_repo_for_stats,
                fail_mae_deg=args.fail_mae_deg,
                fail_prompt_l2_deg=args.fail_prompt_l2_deg,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval3_model_compare.json"
    md_path = out_dir / "EVAL3_MODEL_COMPARE.md"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

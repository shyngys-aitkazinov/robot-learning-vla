#!/usr/bin/env python3
"""Offline Eval 3 checkpoint sweep for SmolVLA.

This intentionally mirrors the deploy-time policy path:
LeRobotDataset frame -> live-style numpy observation -> policy preprocessor ->
policy.select_action -> postprocessor.  The metrics are diagnostics, not a
hardware-success guarantee.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval3_lerobot_shim import apply as _eval3_shim_apply

_eval3_shim_apply()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402


DEFAULT_REPOS = [
    "RobotLearningVLA/dataset_v2_barack_obama_left_1",
    "RobotLearningVLA/dataset_v2_barack_obama_middle_1",
    "RobotLearningVLA/dataset_v2_barack_obama_right_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_left_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_middle_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_right_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_left_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_middle_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_right_1",
]

CANONICAL_PROMPTS = {
    "barack_obama": "Place the coke on Barack Obama",
    "taylor_swift": "Place the coke on Taylor Swift",
    "yann_lecun": "Place the coke on Yann LeCun",
}

DISPLAY_NAMES = {
    "barack_obama": "Barack Obama",
    "taylor_swift": "Taylor Swift",
    "yann_lecun": "Yann LeCun",
}

ACTION_KEY = "action"
STATE_KEY = "observation.state"
DEFAULT_CAMERA_KEY = "observation.images.front"


@dataclass(frozen=True)
class EvalSample:
    repo_id: str
    identity: str
    position: str
    episode_index: int
    frame_index: int
    phase: str


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train/eval3_3way_50k_v5_newdata_balanced"),
        help="Training output directory containing checkpoints/<step>/pretrained_model.",
    )
    ap.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated checkpoint steps. Empty means all checkpoint dirs in --train-dir.",
    )
    ap.add_argument(
        "--dataset-repos",
        default=",".join(DEFAULT_REPOS),
        help="Comma-separated dataset repos to evaluate.",
    )
    ap.add_argument("--revision", default="v3.0", help="HF dataset revision/tag.")
    ap.add_argument("--device", default="cuda", help="Policy device, usually cuda on Brev.")
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    ap.add_argument(
        "--rename-map",
        default='{"observation.images.front":"observation.images.camera1"}',
        help="JSON rename map used by training/deploy.",
    )
    ap.add_argument("--robot-type", default="so101_follower")
    ap.add_argument(
        "--frames-per-episode",
        type=int,
        default=2,
        help="Frames sampled per episode. Default samples mid and final phase.",
    )
    ap.add_argument(
        "--max-samples-per-repo",
        type=int,
        default=0,
        help="Optional cap after episode sampling. 0 disables cap.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval3_reports/eval3_v5_checkpoint_sweep.json"),
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/eval3_reports/eval3_v5_checkpoint_sweep.md"),
    )
    ap.add_argument(
        "--fail-under-prompt-acc",
        type=float,
        default=0.0,
        help="Exit non-zero if best prompt-nearest accuracy is below this fraction. 0 disables.",
    )
    ap.add_argument(
        "--meta-repo-id",
        default="",
        help=(
            "LeRobot Hub dataset id for dataset metadata/stats used to build preprocessors "
            "(must match the checkpoint's training dataset for correct normalization). "
            "Default: first repo in --dataset-repos (evaluation order)."
        ),
    )
    return ap.parse_args()


def _repo_identity_position(repo_id: str) -> tuple[str, str]:
    lower = repo_id.lower()
    if "barack_obama" in lower:
        identity = "barack_obama"
    elif "taylor_swift" in lower:
        identity = "taylor_swift"
    elif "yann_lecun" in lower:
        identity = "yann_lecun"
    else:
        raise ValueError(f"Could not infer identity from repo_id={repo_id!r}")

    position = ""
    for candidate in ("left", "middle", "right"):
        if re.search(rf"(^|_){candidate}(_|$)", lower):
            position = candidate
            break
    if not position:
        raise ValueError(f"Could not infer target position from repo_id={repo_id!r}")
    return identity, position


def _checkpoint_paths(train_dir: Path, checkpoints_csv: str) -> list[tuple[str, Path]]:
    if checkpoints_csv.strip():
        steps = [x.strip() for x in checkpoints_csv.split(",") if x.strip()]
    else:
        ckpt_dir = train_dir / "checkpoints"
        steps = sorted(p.name for p in ckpt_dir.iterdir() if p.is_dir())
    out = []
    for step in steps:
        path = train_dir / "checkpoints" / step / "pretrained_model"
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint pretrained_model not found: {path}")
        out.append((step, path))
    return out


def _episode_columns(ds: LeRobotDataset) -> tuple[list[int], list[int], list[int]]:
    eps = ds.meta.episodes
    episode_indices = [int(x) for x in eps["episode_index"]]
    from_indices = [int(x) for x in eps["dataset_from_index"]]
    to_indices = [int(x) for x in eps["dataset_to_index"]]
    return episode_indices, from_indices, to_indices


def _sample_indices_for_episode(start: int, stop: int, n: int) -> list[tuple[int, str]]:
    length = max(stop - start, 1)
    if n <= 1:
        fractions = [0.88]
    elif n == 2:
        fractions = [0.50, 0.88]
    else:
        fractions = np.linspace(0.25, 0.90, n).tolist()

    sampled: list[tuple[int, str]] = []
    seen = set()
    for frac in fractions:
        offset = min(max(int(math.floor((length - 1) * float(frac))), 0), length - 1)
        idx = start + offset
        if idx in seen:
            continue
        seen.add(idx)
        phase = "final" if frac >= 0.75 else "mid"
        sampled.append((idx, phase))
    return sampled


def _make_samples(ds: LeRobotDataset, repo_id: str, frames_per_episode: int, max_samples: int) -> list[EvalSample]:
    identity, position = _repo_identity_position(repo_id)
    ep_indices, starts, stops = _episode_columns(ds)
    samples: list[EvalSample] = []
    for ep_idx, start, stop in zip(ep_indices, starts, stops):
        for frame_idx, phase in _sample_indices_for_episode(start, stop, frames_per_episode):
            samples.append(
                EvalSample(
                    repo_id=repo_id,
                    identity=identity,
                    position=position,
                    episode_index=ep_idx,
                    frame_index=frame_idx,
                    phase=phase,
                )
            )
    if max_samples > 0 and len(samples) > max_samples:
        keep = np.linspace(0, len(samples) - 1, max_samples).round().astype(int).tolist()
        samples = [samples[i] for i in keep]
    return samples


def _row_to_live_observation(row: dict[str, Any], camera_key: str) -> dict[str, np.ndarray]:
    image = row[camera_key]
    if isinstance(image, torch.Tensor):
        img_t = image.detach().cpu()
        if img_t.ndim == 3 and img_t.shape[0] in (1, 3):
            img_t = img_t.permute(1, 2, 0)
        if img_t.dtype.is_floating_point:
            img_t = (img_t.clamp(0, 1) * 255).to(torch.uint8)
        image_np = img_t.numpy()
    else:
        image_np = np.asarray(image)
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    state = row[STATE_KEY]
    if isinstance(state, torch.Tensor):
        state_np = state.detach().cpu().numpy().astype(np.float32)
    else:
        state_np = np.asarray(state, dtype=np.float32)

    return {camera_key: image_np, STATE_KEY: state_np}


def _action_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _load_datasets(
    repo_ids: list[str], *, revision: str, video_backend: str, frames_per_episode: int, max_samples: int
) -> tuple[dict[str, LeRobotDataset], list[EvalSample]]:
    datasets: dict[str, LeRobotDataset] = {}
    samples: list[EvalSample] = []
    for repo_id in repo_ids:
        ds = LeRobotDataset(repo_id, revision=revision, video_backend=video_backend)
        datasets[repo_id] = ds
        repo_samples = _make_samples(ds, repo_id, frames_per_episode, max_samples)
        samples.extend(repo_samples)
        print(
            f"[dataset] {repo_id}: frames={ds.num_frames} episodes={ds.num_episodes} "
            f"samples={len(repo_samples)}",
            flush=True,
        )
    return datasets, samples


def _load_policy_bundle(
    policy_path: Path,
    *,
    device: str,
    meta_repo_id: str,
    revision: str,
    rename_map: dict[str, str],
):
    ds_meta = LeRobotDatasetMetadata(meta_repo_id, revision=revision)
    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.device = device
    cfg.pretrained_path = str(policy_path)
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(policy_path),
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return cfg, policy, preprocessor, postprocessor, get_safe_torch_device(cfg.device)


def _predict(
    obs: dict[str, np.ndarray],
    *,
    task: str,
    cfg,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    robot_type: str,
) -> np.ndarray:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    obs_copy = {k: np.array(v, copy=True) for k, v in obs.items()}
    action = predict_action(
        observation=obs_copy,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=cfg.use_amp,
        task=task,
        robot_type=robot_type,
    )
    return _action_array(action)


def _pairwise_l2(actions: dict[str, np.ndarray]) -> dict[str, float]:
    keys = sorted(actions)
    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            out[f"{a}__{b}"] = float(np.linalg.norm(actions[a] - actions[b]))
    return out


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summarize_checkpoint(
    rows: list[dict[str, Any]],
    *,
    action_names: list[str],
) -> dict[str, Any]:
    true_err = np.stack([r["true_abs_err"] for r in rows], axis=0)
    true_l2 = [float(r["true_l2"]) for r in rows]
    final_rows = [r for r in rows if r["phase"] == "final"]
    final_true_err = np.stack([r["true_abs_err"] for r in final_rows], axis=0) if final_rows else true_err
    prompt_nearest_acc = float(np.mean([r["nearest_prompt"] == r["identity"] for r in rows]))
    final_prompt_nearest_acc = (
        float(np.mean([r["nearest_prompt"] == r["identity"] for r in final_rows])) if final_rows else None
    )

    pairwise_values = [v for r in rows for v in r["prompt_pairwise_l2"].values()]
    true_other_values = [v for r in rows for v in r["true_vs_other_l2"]]
    wrist_values = [v for r in rows for v in r["prompt_pairwise_wrist_abs"].values()]

    per_joint_mae = {
        action_names[i] if i < len(action_names) else f"joint_{i}": float(true_err[:, i].mean())
        for i in range(true_err.shape[1])
    }
    final_per_joint_mae = {
        action_names[i] if i < len(action_names) else f"joint_{i}": float(final_true_err[:, i].mean())
        for i in range(final_true_err.shape[1])
    }
    worst_joint = max(per_joint_mae.items(), key=lambda kv: kv[1])

    by_identity_position: dict[str, dict[str, Any]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["identity"], row["position"])].append(row)
    for (identity, position), group in sorted(grouped.items()):
        gerr = np.stack([r["true_abs_err"] for r in group], axis=0)
        gfinal = [r for r in group if r["phase"] == "final"]
        by_identity_position[f"{identity}/{position}"] = {
            "samples": len(group),
            "mae": float(gerr.mean()),
            "prompt_nearest_acc": float(np.mean([r["nearest_prompt"] == r["identity"] for r in group])),
            "final_samples": len(gfinal),
            "final_true_wrist_roll_mean": _mean([r["true_prompt_action"][4] for r in gfinal]),
            "final_gt_wrist_roll_mean": _mean([r["gt_action"][4] for r in gfinal]),
            "final_mae": _mean([float(np.mean(r["true_abs_err"])) for r in gfinal]),
        }

    confusion = Counter(f"{r['identity']}->{r['nearest_prompt']}" for r in rows)

    return {
        "samples": len(rows),
        "final_samples": len(final_rows),
        "mae": float(true_err.mean()),
        "l2": float(np.mean(true_l2)),
        "final_mae": float(final_true_err.mean()),
        "prompt_nearest_acc": prompt_nearest_acc,
        "final_prompt_nearest_acc": final_prompt_nearest_acc,
        "prompt_pairwise_l2_mean": _mean(pairwise_values),
        "true_vs_other_l2_mean": _mean(true_other_values),
        "prompt_pairwise_wrist_abs_mean": _mean(wrist_values),
        "per_joint_mae": per_joint_mae,
        "final_per_joint_mae": final_per_joint_mae,
        "worst_joint": {"name": worst_joint[0], "mae": worst_joint[1]},
        "by_identity_position": by_identity_position,
        "nearest_prompt_confusion": dict(sorted(confusion.items())),
    }


def _evaluate_checkpoint(
    step: str,
    policy_path: Path,
    *,
    datasets: dict[str, LeRobotDataset],
    samples: list[EvalSample],
    action_names: list[str],
    camera_key: str,
    device: str,
    revision: str,
    rename_map: dict[str, str],
    robot_type: str,
    meta_repo_id: str,
) -> dict[str, Any]:
    print(f"[checkpoint] loading {step}: {policy_path}", flush=True)
    t0 = time.time()
    cfg, policy, preprocessor, postprocessor, torch_device = _load_policy_bundle(
        policy_path,
        device=device,
        meta_repo_id=meta_repo_id,
        revision=revision,
        rename_map=rename_map,
    )
    print(f"[checkpoint] {step}: loaded in {time.time() - t0:.1f}s", flush=True)

    rows: list[dict[str, Any]] = []
    for i, sample in enumerate(samples, start=1):
        ds = datasets[sample.repo_id]
        row = ds[sample.frame_index]
        obs = _row_to_live_observation(row, camera_key)
        gt = _action_array(row[ACTION_KEY])

        prompt_actions = {}
        for identity, prompt in CANONICAL_PROMPTS.items():
            prompt_actions[identity] = _predict(
                obs,
                task=prompt,
                cfg=cfg,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=torch_device,
                robot_type=robot_type,
            )

        distances_to_gt = {identity: float(np.linalg.norm(action - gt)) for identity, action in prompt_actions.items()}
        nearest_prompt = min(distances_to_gt.items(), key=lambda kv: kv[1])[0]
        true_action = prompt_actions[sample.identity]
        pairwise_l2 = _pairwise_l2(prompt_actions)
        pairwise_wrist = {
            key: float(abs(prompt_actions[key.split("__")[0]][4] - prompt_actions[key.split("__")[1]][4]))
            for key in pairwise_l2
        }
        true_vs_other_l2 = [
            float(np.linalg.norm(true_action - action))
            for identity, action in prompt_actions.items()
            if identity != sample.identity
        ]
        rows.append(
            {
                "repo_id": sample.repo_id,
                "identity": sample.identity,
                "position": sample.position,
                "episode_index": sample.episode_index,
                "frame_index": sample.frame_index,
                "phase": sample.phase,
                "gt_action": gt.tolist(),
                "true_prompt_action": true_action.tolist(),
                "true_abs_err": np.abs(true_action - gt),
                "true_l2": float(np.linalg.norm(true_action - gt)),
                "distances_to_gt": distances_to_gt,
                "nearest_prompt": nearest_prompt,
                "prompt_pairwise_l2": pairwise_l2,
                "prompt_pairwise_wrist_abs": pairwise_wrist,
                "true_vs_other_l2": true_vs_other_l2,
            }
        )
        if i % 25 == 0 or i == len(samples):
            print(f"[checkpoint] {step}: evaluated {i}/{len(samples)} samples", flush=True)

    summary = _summarize_checkpoint(rows, action_names=action_names)
    del policy, preprocessor, postprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "step": step,
        "policy_path": str(policy_path),
        "summary": summary,
        "rows": [
            {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in row.items()
                if k != "true_abs_err"
            }
            for row in rows
        ],
    }


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    s = result["summary"]
    return (
        float(s["prompt_nearest_acc"]),
        float(s["final_prompt_nearest_acc"] or 0.0),
        float(s["true_vs_other_l2_mean"] or 0.0),
        -float(s["final_mae"]),
    )


def _write_report(report: dict[str, Any], output_md: Path) -> None:
    results = report["results"]
    best = report["best_checkpoint"]
    lines = [
        "# Eval 3 SmolVLA Checkpoint Sweep",
        "",
        "Offline diagnostics only. These numbers do not prove hardware success.",
        "",
        f"- Train dir: `{report['train_dir']}`",
        f"- Samples: `{report['num_samples']}` frames from `{len(report['dataset_repos'])}` repos",
        f"- Best checkpoint by offline ranking: `{best}`",
        "",
        "## Checkpoint Summary",
        "",
        "| checkpoint | MAE | final MAE | prompt-nearest acc | final prompt acc | true-vs-other L2 | prompt pair L2 | wrist prompt diff | worst joint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        s = result["summary"]
        lines.append(
            "| {step} | {mae:.4f} | {final_mae:.4f} | {acc:.1%} | {facc:.1%} | "
            "{tv:.3f} | {pl2:.3f} | {wr:.3f} | {wj}={wjv:.3f} |".format(
                step=result["step"],
                mae=s["mae"],
                final_mae=s["final_mae"],
                acc=s["prompt_nearest_acc"],
                facc=s["final_prompt_nearest_acc"] or 0.0,
                tv=s["true_vs_other_l2_mean"] or 0.0,
                pl2=s["prompt_pairwise_l2_mean"] or 0.0,
                wr=s["prompt_pairwise_wrist_abs_mean"] or 0.0,
                wj=s["worst_joint"]["name"],
                wjv=s["worst_joint"]["mae"],
            )
        )

    best_result = next(r for r in results if r["step"] == best)
    lines.extend(
        [
            "",
            "## Best Checkpoint Identity/Position Breakdown",
            "",
            "| identity/position | samples | MAE | final MAE | prompt acc | final predicted wrist | final GT wrist |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in best_result["summary"]["by_identity_position"].items():
        lines.append(
            "| {key} | {samples} | {mae:.4f} | {final_mae:.4f} | {acc:.1%} | {pw:.2f} | {gw:.2f} |".format(
                key=key,
                samples=value["samples"],
                mae=value["mae"],
                final_mae=value["final_mae"] or 0.0,
                acc=value["prompt_nearest_acc"],
                pw=value["final_true_wrist_roll_mean"] or 0.0,
                gw=value["final_gt_wrist_roll_mean"] or 0.0,
            )
        )

    lines.extend(["", "## Best Checkpoint Prompt Confusion", ""])
    for key, count in best_result["summary"]["nearest_prompt_confusion"].items():
        lines.append(f"- `{key}`: {count}")

    lines.extend(
        [
            "",
            "## Ranking Rule",
            "",
            "Sorted by prompt-nearest accuracy, then final prompt-nearest accuracy, then true-vs-other prompt L2, then lower final MAE.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    repo_ids = [x.strip() for x in args.dataset_repos.split(",") if x.strip()]
    rename_map = json.loads(args.rename_map)
    ckpts = _checkpoint_paths(args.train_dir, args.checkpoints)
    meta_repo_id = (args.meta_repo_id or "").strip() or repo_ids[0]

    print(f"[sweep] checkpoints={[step for step, _ in ckpts]}", flush=True)
    datasets, samples = _load_datasets(
        repo_ids,
        revision=args.revision,
        video_backend=args.video_backend,
        frames_per_episode=args.frames_per_episode,
        max_samples=args.max_samples_per_repo,
    )
    if not samples:
        raise RuntimeError("No evaluation samples selected.")
    action_names = datasets[repo_ids[0]].meta.features[ACTION_KEY].get("names") or []
    print(f"[sweep] total_samples={len(samples)} action_names={action_names}", flush=True)
    print(f"[sweep] meta_repo_id={meta_repo_id}", flush=True)

    results = []
    for step, policy_path in ckpts:
        results.append(
            _evaluate_checkpoint(
                step,
                policy_path,
                datasets=datasets,
                samples=samples,
                action_names=action_names,
                camera_key=args.camera_key,
                device=args.device,
                revision=args.revision,
                rename_map=rename_map,
                robot_type=args.robot_type,
                meta_repo_id=meta_repo_id,
            )
        )

    results = sorted(results, key=_rank_key, reverse=True)
    best = results[0]["step"]
    report = {
        "created_unix_s": time.time(),
        "train_dir": str(args.train_dir),
        "dataset_repos": repo_ids,
        "num_samples": len(samples),
        "sample_config": {
            "frames_per_episode": args.frames_per_episode,
            "max_samples_per_repo": args.max_samples_per_repo,
            "revision": args.revision,
            "video_backend": args.video_backend,
            "camera_key": args.camera_key,
            "rename_map": rename_map,
            "device": args.device,
            "meta_repo_id": meta_repo_id,
        },
        "best_checkpoint": best,
        "results": results,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report(report, args.output_md)
    print(f"[sweep] wrote {args.output_json}", flush=True)
    print(f"[sweep] wrote {args.output_md}", flush=True)
    print(f"[sweep] best_checkpoint={best}", flush=True)

    if args.fail_under_prompt_acc > 0 and results[0]["summary"]["prompt_nearest_acc"] < args.fail_under_prompt_acc:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

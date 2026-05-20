"""Held-out validation — behavior-cloning loss of saved checkpoints.

Training runs on the real `dataset_v4` corpus; this scores every saved
checkpoint's BC loss on held-out **synthetic** `dataset_v3` data. Train and
validation are different domains (real teleop vs synthetic composites), so the
number is a cross-domain generalization signal: it rises once the policy
overfits the real-data quirks instead of learning the task.

lerobot's training loop has no validation hook, so this runs per-checkpoint
*after* training (train.py calls it automatically), or standalone on a
finished run directory:

    python fresh_start/validate.py
    python fresh_start/validate.py --training.output_dir=/ephemeral/outputs/<run>
    python fresh_start/validate.py --validation.max_episodes_per_dataset=8
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap: make sibling modules importable when run as a script ---
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import groot_shim  # noqa: E402

groot_shim.apply()  # MUST run before any `lerobot.policies` import

import draccus  # noqa: E402

from config import PipelineConfig, ValidationConfig  # noqa: E402
from merge_datasets import resolve_source_root  # noqa: E402


def _build_val_datasets(val_cfg: ValidationConfig, chunk_size: int, video_backend: str):
    """Build one capped `LeRobotDataset` per validation source root."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    datasets = []
    for root_rel in val_cfg.source_roots:
        root = resolve_source_root(root_rel)
        if not (root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Validation dataset not found: {root}\n"
                f"Pull it from the Hub first (e.g. RobotLearningVLA/{root.name})."
            )
        repo_id = f"local/{root.name}"
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        n_episodes = min(val_cfg.max_episodes_per_dataset, meta.total_episodes)
        # delta_timestamps: current obs + the full action chunk the policy predicts.
        delta_timestamps = {
            "observation.state": [0.0],
            "observation.images.front": [0.0],
            "action": [i / meta.fps for i in range(chunk_size)],
        }
        dataset = LeRobotDataset(
            repo_id,
            root=str(root),
            delta_timestamps=delta_timestamps,
            episodes=list(range(n_episodes)),
            video_backend=video_backend,
        )
        datasets.append((root.name, dataset))
    return datasets


def evaluate_checkpoint(
    checkpoint_model_dir: str | Path,
    val_datasets: list,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict:
    """Mean BC loss of one checkpoint over the validation datasets.

    `checkpoint_model_dir` is a `.../checkpoints/<step>/pretrained_model` dir.
    Returns ``{"overall": float, "per_dataset": {name: float}, "frames": int}``.
    """
    import torch
    from torch.utils.data import DataLoader

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    checkpoint_model_dir = Path(checkpoint_model_dir)
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_model_dir)
    policy_cfg.pretrained_path = checkpoint_model_dir
    policy_cfg.device = device

    # Build the policy with weights from the checkpoint; the saved preprocessor
    # already carries the training-set normalization stats (the correct ones).
    policy = make_policy(cfg=policy_cfg, ds_meta=val_datasets[0][1].meta, rename_map={})
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(checkpoint_model_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    per_dataset: dict[str, float] = {}
    total_sum, total_count = 0.0, 0
    for name, dataset in val_datasets:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
        ds_sum, ds_count = 0.0, 0
        for raw_batch in loader:
            count = int(raw_batch["action"].shape[0])
            batch = preprocessor(raw_batch)
            with torch.no_grad():
                loss, _ = policy.forward(batch)
            ds_sum += float(loss) * count
            ds_count += count
        per_dataset[name] = ds_sum / max(ds_count, 1)
        total_sum += ds_sum
        total_count += ds_count

    del policy, preprocessor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "overall": total_sum / max(total_count, 1),
        "per_dataset": per_dataset,
        "frames": total_count,
    }


def evaluate_run(cfg: PipelineConfig, output_dir: str | Path | None = None) -> list[dict]:
    """Score every checkpoint in a run directory; print a ranked table."""
    out = Path(output_dir or cfg.training.output_dir)
    checkpoints = sorted(
        (p for p in out.glob("checkpoints/*/pretrained_model") if p.parent.name.isdigit()),
        key=lambda p: int(p.parent.name),
    )
    if not checkpoints:
        print(f"[validate] no checkpoints under {out}/checkpoints — nothing to validate")
        return []

    from lerobot.configs.policies import PreTrainedConfig

    chunk_size = int(getattr(PreTrainedConfig.from_pretrained(checkpoints[0]), "chunk_size", 50))
    val_datasets = _build_val_datasets(cfg.validation, chunk_size, cfg.data.video_backend)
    names = [name for name, _ in val_datasets]
    total_frames = sum(len(ds) for _, ds in val_datasets)
    print(
        f"[validate] held-out validation on {len(val_datasets)} synthetic dataset(s), "
        f"{total_frames} frames (<= {cfg.validation.max_episodes_per_dataset} ep each)"
    )

    results: list[dict] = []
    for checkpoint in checkpoints:
        step = int(checkpoint.parent.name)
        result = evaluate_checkpoint(
            checkpoint,
            val_datasets,
            cfg.training.device,
            cfg.validation.batch_size,
            cfg.validation.num_workers,
        )
        results.append({"step": step, **result})
        per = "  ".join(f"{n}={result['per_dataset'][n]:.4f}" for n in names)
        print(f"[validate]   step {step:>6}: val_loss={result['overall']:.4f}   {per}")

    best = min(results, key=lambda r: r["overall"])
    print(
        f"[validate] best checkpoint: step {best['step']} "
        f"(val_loss={best['overall']:.4f}) — lowest cross-domain BC loss"
    )
    return results


def main() -> None:
    cfg = draccus.parse(PipelineConfig)
    if not cfg.validation.enable:
        print("[validate] validation.enable=false — nothing to do")
        return
    evaluate_run(cfg)


if __name__ == "__main__":
    main()

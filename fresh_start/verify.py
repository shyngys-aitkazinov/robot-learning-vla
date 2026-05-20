"""Staged verification for the SmolVLA pipeline.

Runs cheap-to-expensive checks; each stage gates the next. Run all stages, or
one at a time:

    python fresh_start/verify.py                 # stages 1-6
    python fresh_start/verify.py --stage 5       # just the augmentation preview
    python fresh_start/verify.py --smoke-steps 20

Stages:
    1  shim         — `import lerobot.policies` works in a fresh interpreter
    2  preflight    — the 9 source datasets exist and share one schema
    3  merge        — aggregate them into the single training corpus
    4  dataset      — the merged dataset loads; frames are float [0,1] CHW
    5  augmentation — apply the configured transforms, save before/after PNGs
    6  smoke train  — a short SmolVLA fine-tune + held-out validation
                      (downloads smolvla_base)
"""

from __future__ import annotations

import argparse
import copy
import random
import subprocess
import sys
import time
from pathlib import Path

# --- bootstrap: make sibling modules importable when run as a script ---
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import groot_shim  # noqa: E402

groot_shim.apply()

import transforms_ext  # noqa: E402

transforms_ext.register()

from augmentation import build_image_transforms, summarize  # noqa: E402
from config import PipelineConfig  # noqa: E402
from merge_datasets import ensure_merged, resolve_source_root  # noqa: E402

_PREVIEW_DIR = Path("/ephemeral/outputs/aug_preview")


def _banner(n: int, name: str) -> None:
    print(f"\n{'=' * 70}\n[stage {n}] {name}\n{'=' * 70}")


def stage1_shim() -> None:
    _banner(1, "GROOT shim — import lerobot.policies in a fresh interpreter")
    result = subprocess.run(
        [sys.executable, str(_HERE / "groot_shim.py")],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"shim selftest failed:\n{result.stderr}")
    print("PASS")


def stage2_preflight(cfg: PipelineConfig) -> None:
    _banner(2, "Source-dataset preflight")
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    roots = [resolve_source_root(p) for p in cfg.data.source_roots]
    metas = []
    for r in roots:
        meta = LeRobotDatasetMetadata(f"local/{r.name}", root=r)
        metas.append(meta)
        tasks = sorted(set(meta.tasks.index.tolist()))
        print(f"  {r.name:32s} {meta.total_episodes:3d} ep  {meta.total_frames:6d} frames  {tasks}")

    fps = {m.fps for m in metas}
    robots = {m.robot_type for m in metas}
    features = {tuple(sorted(m.features)) for m in metas}
    if len(fps) != 1:
        raise AssertionError(f"fps mismatch across datasets: {fps}")
    if len(robots) != 1:
        raise AssertionError(f"robot_type mismatch across datasets: {robots}")
    if len(features) != 1:
        raise AssertionError("feature-schema mismatch across datasets")
    total_ep = sum(m.total_episodes for m in metas)
    total_fr = sum(m.total_frames for m in metas)
    print(f"  consistent: fps={fps.pop()} robot={robots.pop()} | total {total_ep} ep, {total_fr} frames")
    print("PASS")


def stage3_merge(cfg: PipelineConfig) -> None:
    _banner(3, "Merge datasets")
    ensure_merged(cfg.data)
    print("PASS")


def stage4_dataset(cfg: PipelineConfig) -> None:
    _banner(4, "Merged-dataset sanity")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        cfg.data.merged_repo_id, root=cfg.data.merged_root, video_backend=cfg.data.video_backend
    )
    print(f"  {ds.num_episodes} episodes, {ds.num_frames} frames")
    for idx in (0, ds.num_frames // 2, ds.num_frames - 1):
        item = ds[idx]
        img = item["observation.images.front"]
        if img.dtype != torch.float32:
            raise AssertionError(f"frame {idx}: image dtype {img.dtype}, expected float32")
        if img.ndim != 3 or img.shape[0] != 3:
            raise AssertionError(f"frame {idx}: image shape {tuple(img.shape)}, expected (3,H,W)")
        if not (0.0 <= float(img.min()) and float(img.max()) <= 1.0):
            raise AssertionError(f"frame {idx}: image range [{img.min()},{img.max()}] not in [0,1]")
        if item["observation.state"].shape[-1] != 6 or item["action"].shape[-1] != 6:
            raise AssertionError(f"frame {idx}: state/action are not 6-dim")
    print(f"  image={tuple(ds[0]['observation.images.front'].shape)} float32 [0,1]  "
          f"state/action=6d  task={ds[0]['task']!r}")
    print("PASS")


def stage5_augmentation(cfg: PipelineConfig, n_samples: int = 4) -> None:
    _banner(5, "Augmentation preview")
    import torch
    import torchvision.utils as vutils
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"  {summarize(cfg.augmentation)}")
    ds = LeRobotDataset(
        cfg.data.merged_repo_id, root=cfg.data.merged_root, video_backend=cfg.data.video_backend
    )
    transforms = build_image_transforms(cfg.augmentation)

    random.seed(0)
    idxs = random.sample(range(ds.num_frames), k=min(n_samples, ds.num_frames))
    rows: list[torch.Tensor] = []
    for idx in idxs:
        original = ds[idx]["observation.images.front"]
        augmented = transforms(original.clone())
        rows.extend([original, augmented.clamp(0.0, 1.0)])

    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    out = _PREVIEW_DIR / "aug_preview.png"
    vutils.save_image(vutils.make_grid(torch.stack(rows), nrow=2, padding=4), out)
    print(f"  wrote {out}  (left column = original, right column = augmented)")
    print("PASS")


def stage6_smoke(cfg: PipelineConfig, steps: int, batch: int) -> None:
    _banner(6, f"Smoke train + validation ({steps} steps, batch {batch})")
    import train as train_module

    smoke = copy.deepcopy(cfg)
    smoke.training.steps = steps
    smoke.training.batch_size = batch
    smoke.training.save_freq = steps
    smoke.training.log_freq = 1
    smoke.training.num_workers = 2
    smoke.training.wandb_enable = False
    smoke.training.job_name = "smolvla_smoke"
    smoke.training.output_dir = f"/ephemeral/outputs/smoke_{int(time.time())}"
    # keep the post-train validation tiny so the smoke stays fast
    smoke.validation.max_episodes_per_dataset = 1

    out_dir = train_module.run(smoke)
    checkpoints = sorted(Path(out_dir).glob("checkpoints/*/pretrained_model"))
    if not checkpoints:
        raise AssertionError(f"smoke train wrote no checkpoint under {out_dir}")
    print(f"  checkpoint written: {checkpoints[-1]}")
    print("PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 7), help="run only this stage")
    parser.add_argument("--smoke-steps", type=int, default=10, help="steps for stage 6")
    parser.add_argument("--smoke-batch", type=int, default=4, help="batch size for stage 6")
    args = parser.parse_args()

    cfg = PipelineConfig()  # defaults; verification doesn't need CLI config overrides
    stages = {
        1: lambda: stage1_shim(),
        2: lambda: stage2_preflight(cfg),
        3: lambda: stage3_merge(cfg),
        4: lambda: stage4_dataset(cfg),
        5: lambda: stage5_augmentation(cfg),
        6: lambda: stage6_smoke(cfg, args.smoke_steps, args.smoke_batch),
    }
    to_run = [args.stage] if args.stage else sorted(stages)
    for n in to_run:
        stages[n]()
    print(f"\nAll requested stages passed: {to_run}")


if __name__ == "__main__":
    main()

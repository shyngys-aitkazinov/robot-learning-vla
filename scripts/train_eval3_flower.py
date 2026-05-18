#!/usr/bin/env python3
"""Train FlowerVLA on Eval 3 exact-data recipes.

Run this inside a separate FlowerVLA environment; do not install FlowerVLA into
the main LeRobot/SmolVLA `.venv`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval3_external_vla_data import (  # noqa: E402
    RECIPES,
    Eval3ExternalVLADataset,
    collate_external_vla,
    compute_recipe_statistics,
    get_recipe,
    normalize_q01_q99,
)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def build_flower_model(args: argparse.Namespace, device: torch.device):
    _patch_optional_flash_attn_check()

    if args.flower_src:
        src = Path(args.flower_src).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"--flower-src does not exist: {src}")
        sys.path.insert(0, str(src))

    try:
        from flower.models.flower import FLOWERVLA
    except Exception as exc:  # pragma: no cover - depends on external checkout
        raise RuntimeError(
            "Could not import FlowerVLA. Clone intuitive-robots/flower_vla_calvin or "
            "flower_vla_pret and pass --flower-src=/path/to/checkout."
        ) from exc

    pretrained_path = args.pretrained_checkpoint
    if not pretrained_path:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("huggingface_hub is required to download mbreuss/flower_vla_pret") from exc
        pretrained_path = hf_hub_download(args.pretrained_repo, args.pretrained_file)

    model = FLOWERVLA(
        vlm_path=args.vlm_path,
        use_second_view=False,
        second_view_key=None,
        lowdim_obs_dim=7,
        action_dim=7,
        act_window_size=args.chunk_size,
        multistep=args.chunk_size,
        num_sampling_steps=args.num_sampling_steps,
        use_proprio=True,
        return_act_chunk=False,
        dit_dim=args.dit_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_rope=True,
        query_seq_len=args.query_seq_len,
        sampling_type="uniform",
        load_pretrained=bool(pretrained_path),
        pretrained_model_path=pretrained_path,
    )

    # The upstream model has had two slightly different proprio batch contracts.
    # This makes encode_observations read batch["obs"]["proprio"] when needed.
    if getattr(model, "use_proprio", False) and not isinstance(getattr(model, "obs_modalities", None), str):
        model.obs_modalities = "obs"

    return model.to(device)


def _patch_optional_flash_attn_check() -> None:
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
    except Exception:
        return

    original_check_imports = dynamic_module_utils.check_imports
    if getattr(original_check_imports, "_eval3_flower_flash_attn_optional", False):
        return

    def patched_check_imports(filename):
        try:
            return original_check_imports(filename)
        except ImportError as exc:
            if "flash_attn" not in str(exc):
                raise

            missing_packages = []
            for imp in dynamic_module_utils.get_imports(filename):
                if imp == "flash_attn":
                    continue
                try:
                    importlib.import_module(imp)
                except ImportError:
                    missing_packages.append(imp)

            if missing_packages:
                raise ImportError(
                    "This modeling file requires the following packages that were not found in your environment: "
                    f"{', '.join(missing_packages)}. Run `pip install {' '.join(missing_packages)}`"
                ) from exc

            print("FlowerVLA: treating optional flash_attn import as unavailable; using standard attention.", flush=True)
            return dynamic_module_utils.get_relative_imports(filename)

    patched_check_imports._eval3_flower_flash_attn_optional = True
    dynamic_module_utils.check_imports = patched_check_imports


def make_flower_batch(batch: dict[str, Any], *, device: torch.device, stats: dict[str, Any]) -> dict[str, Any]:
    images = batch["images"].to(device=device, dtype=torch.float32).unsqueeze(1)
    states = batch["states"].to(device=device, dtype=torch.float32).unsqueeze(1)
    actions = batch["actions"].to(device=device, dtype=torch.float32)

    states = normalize_q01_q99(states, {"action": stats["proprio"]})
    actions = normalize_q01_q99(actions, stats)
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
        "lang_text": batch["tasks"],
        "actions": actions,
    }


def flower_loss(model, flower_batch: dict[str, Any]) -> torch.Tensor:
    features = model.encode_observations(flower_batch)
    out = model.rf_loss(features, flower_batch["actions"])
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, dict) and "loss" in out:
        return out["loss"]
    if torch.is_tensor(out):
        return out
    raise TypeError(f"Unsupported FlowerVLA rf_loss return type: {type(out)!r}")


def save_checkpoint(
    *,
    model,
    optimizer,
    out_dir: Path,
    step: int,
    args: argparse.Namespace,
    stats: dict[str, Any],
    final: bool = False,
) -> Path:
    ckpt_dir = out_dir / "checkpoints" / f"{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "step": int(step),
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": _jsonable(vars(args)),
            "dataset_statistics": stats,
            "final": bool(final),
        },
        ckpt_dir / "checkpoint.pt",
    )
    (ckpt_dir / "train_config.json").write_text(json.dumps(_jsonable(vars(args)), indent=2) + "\n")
    (ckpt_dir / "dataset_statistics.json").write_text(json.dumps({stats["unnorm_key"]: stats}, indent=2) + "\n")
    return ckpt_dir


def maybe_upload(ckpt_dir: Path, repo_id: str, *, private: bool = True) -> None:
    if not repo_id:
        return
    from huggingface_hub import HfApi, create_repo, upload_folder

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api = HfApi()
    sha = api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Eval3 FlowerVLA checkpoint {ckpt_dir.name}",
    )
    print(f"uploaded {ckpt_dir} -> {repo_id} ({sha})", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", choices=RECIPES.keys(), default="new66")
    ap.add_argument("--job-name", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--flower-src", default="", help="Local FlowerVLA checkout containing flower/models/flower.py")
    ap.add_argument("--pretrained-repo", default="mbreuss/flower_vla_pret")
    ap.add_argument("--pretrained-file", default="360000_model_weights.pt")
    ap.add_argument("--pretrained-checkpoint", default="")
    ap.add_argument("--vlm-path", default="microsoft/Florence-2-large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--save-freq", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--task-mode", choices=("raw", "canonical", "mixed"), default="mixed")
    ap.add_argument("--task-canonical-p", type=float, default=0.8)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--num-sampling-steps", type=int, default=4)
    ap.add_argument("--dit-dim", type=int, default=1024)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--n-layers", type=int, default=18)
    ap.add_argument("--query-seq-len", type=int, default=100)
    ap.add_argument("--log-freq", type=int, default=20)
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--push-intermediate", action="store_true")
    ap.add_argument("--hub-repo-id", default="")
    ap.add_argument("--public", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    recipe = get_recipe(args.recipe)
    if not args.job_name:
        args.job_name = recipe.job_ids["flower"]
    if not args.output_dir:
        args.output_dir = str(ROOT / "outputs" / "train" / args.job_name)
    if not args.hub_repo_id and args.push_to_hub:
        args.hub_repo_id = f"RobotLearningVLA/{args.job_name}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f">> Eval3 FlowerVLA train: recipe={recipe.name} job={args.job_name}", flush=True)
    print(f"   output={out_dir}", flush=True)
    print("   data labels: exact raw labels, no gripper repair, no action smoothing, no extra cap", flush=True)

    stats = compute_recipe_statistics(recipe, revision=args.revision, video_backend=args.video_backend)
    if stats["num_transitions"] != recipe.expected_frames:
        raise RuntimeError(f"{recipe.name} has {stats['num_transitions']} frames, expected {recipe.expected_frames}")
    (out_dir / "dataset_statistics.json").write_text(json.dumps({recipe.unnorm_key: stats}, indent=2) + "\n")

    ds = Eval3ExternalVLADataset(
        recipe,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        task_mode=args.task_mode,
        task_canonical_p=args.task_canonical_p,
        revision=args.revision,
        video_backend=args.video_backend,
        download_videos=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_external_vla,
    )

    model = build_flower_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler_enabled = device.type == "cuda"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    data_iter = cycle(loader)
    last_loss = math.nan
    for step in range(1, args.steps + 1):
        batch = next(data_iter)
        flower_batch = make_flower_batch(batch, device=device, stats=stats)
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if scaler_enabled else nullcontext()
        with autocast_ctx:
            loss = flower_loss(model, flower_batch)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        last_loss = float(loss.detach().cpu().item())

        if step == 1 or step % args.log_freq == 0:
            print(f"step={step:06d} loss={last_loss:.6f}", flush=True)
        if step % args.save_freq == 0:
            ckpt_dir = save_checkpoint(model=model, optimizer=optimizer, out_dir=out_dir, step=step, args=args, stats=stats)
            print(f"saved {ckpt_dir}", flush=True)
            if args.push_to_hub and args.push_intermediate:
                maybe_upload(ckpt_dir, f"{args.hub_repo_id}-{step//1000}k", private=not args.public)

    final_dir = save_checkpoint(model=model, optimizer=optimizer, out_dir=out_dir, step=args.steps, args=args, stats=stats, final=True)
    (out_dir / "train_summary.json").write_text(
        json.dumps({"job_name": args.job_name, "recipe": recipe.name, "steps": args.steps, "last_loss": last_loss}, indent=2) + "\n"
    )
    if args.push_to_hub:
        maybe_upload(final_dir, args.hub_repo_id, private=not args.public)
    print(f"done: {final_dir}", flush=True)


if __name__ == "__main__":
    main()

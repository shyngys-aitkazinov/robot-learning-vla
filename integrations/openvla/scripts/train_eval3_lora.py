#!/usr/bin/env python3
"""LoRA fine-tune OpenVLA on Eval 3 exact-data recipes.

This is a local PyTorch Dataset path for LeRobot Hub data. It follows the
upstream OpenVLA fine-tune contract but avoids RLDS conversion for these first
Eval 3 external-VLA runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval3_external_vla_data import (  # noqa: E402
    RECIPES,
    Eval3ExternalVLADataset,
    compute_openvla_chunk_statistics,
    get_recipe,
    normalize_q01_q99,
    tensor_chw_to_pil,
)


IGNORE_INDEX = -100


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _stats_payload(stats: dict[str, Any]) -> dict[str, Any]:
    key = stats["unnorm_key"]
    body = {k: v for k, v in stats.items() if k != "unnorm_key"}
    return {key: body}


def register_openvla(openvla_src: str = "") -> dict[str, Any]:
    if openvla_src:
        src = Path(openvla_src).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"--openvla-src does not exist: {src}")
        sys.path.insert(0, str(src))

    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - depends on external OpenVLA env
        raise RuntimeError(
            "OpenVLA training dependencies are missing. Activate the OpenVLA venv and pass "
            "--openvla-src=/path/to/openvla checkout if prismatic is not importable."
        ) from exc

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    return {
        "ActionTokenizer": ActionTokenizer,
        "AutoModelForVision2Seq": AutoModelForVision2Seq,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LoraConfig": LoraConfig,
        "PaddedCollatorForActionPrediction": PaddedCollatorForActionPrediction,
        "PurePromptBuilder": PurePromptBuilder,
        "VicunaV15ChatPromptBuilder": VicunaV15ChatPromptBuilder,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


class Eval3OpenVLADataset(Dataset):
    def __init__(
        self,
        *,
        recipe: str,
        stats: dict[str, Any],
        action_tokenizer,
        base_tokenizer,
        image_transform,
        prompt_builder_fn,
        chunk_size: int,
        task_mode: str,
        task_canonical_p: float,
        image_aug: bool,
        revision: str | None,
        video_backend: str,
    ):
        self.base = Eval3ExternalVLADataset(
            recipe,
            chunk_size=chunk_size,
            image_size=None,
            task_mode=task_mode,
            task_canonical_p=task_canonical_p,
            revision=revision,
            video_backend=video_backend,
            download_videos=True,
        )
        self.stats = stats
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.image_aug = bool(image_aug)
        self.dataset_statistics = _stats_payload(stats)
        self._augment = self._build_augment() if image_aug else None

    def _build_augment(self):
        try:
            import torchvision.transforms as T
        except Exception:
            return None
        return T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0), antialias=True),
                T.ColorJitter(brightness=0.2, contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=0.05),
            ]
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]
        image = tensor_chw_to_pil(sample["image"])
        if self._augment is not None:
            image = self._augment(image)

        action_flat = sample["actions"].reshape(-1)
        normalized_action = normalize_q01_q99(action_flat, self.stats).detach().cpu().numpy().astype(np.float32)
        instruction = sample["task"].strip().lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(normalized_action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        labels_t[: -(len(normalized_action) + 1)] = IGNORE_INDEX

        return {
            "pixel_values": self.image_transform(image),
            "input_ids": input_ids_t,
            "labels": labels_t,
            "dataset_name": self.stats["unnorm_key"],
        }


def move_batch(batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if key == "pixel_values":
            if isinstance(value, torch.Tensor):
                out[key] = value.to(device=device, dtype=dtype)
            elif isinstance(value, dict):
                out[key] = {k: v.to(device=device, dtype=dtype) for k, v in value.items()}
            else:
                out[key] = value
        elif isinstance(value, torch.Tensor):
            out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def build_model(args: argparse.Namespace, deps: dict[str, Any], stats: dict[str, Any], device: torch.device):
    quantization_config = None
    if args.use_quantization:
        if device.type != "cuda":
            raise ValueError("--use-quantization requires a CUDA device")
        quantization_config = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = deps["AutoProcessor"].from_pretrained(args.vla_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model_kwargs = {
        "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.use_quantization:
        model_kwargs["device_map"] = "auto"
    model = deps["AutoModelForVision2Seq"].from_pretrained(args.vla_path, **model_kwargs)

    norm_stats = _stats_payload(stats)
    model.config.norm_stats = norm_stats
    if hasattr(model, "norm_stats"):
        model.norm_stats = norm_stats

    if args.use_quantization:
        model = deps["prepare_model_for_kbit_training"](model)
    else:
        model = model.to(device)

    lora_config = deps["LoraConfig"](
        r=args.lora_rank,
        lora_alpha=min(args.lora_rank, 16),
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = deps["get_peft_model"](model, lora_config)
    model.print_trainable_parameters()
    return processor, model


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    model,
    processor,
    optimizer,
    step: int,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
    torch.save({"step": int(step), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_dir / "optimizer.pt")
    (checkpoint_dir / "dataset_statistics.json").write_text(json.dumps(_stats_payload(stats), indent=2) + "\n")
    (checkpoint_dir / "train_config.json").write_text(json.dumps(_jsonable(vars(args)), indent=2) + "\n")


def maybe_upload(folder: Path, repo_id: str, *, private: bool) -> None:
    if not repo_id:
        return
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    sha = HfApi().upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Eval3 OpenVLA LoRA checkpoint {folder.name}",
    )
    print(f"uploaded {folder} -> {repo_id} ({sha})", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", choices=RECIPES.keys(), default="new66")
    ap.add_argument("--job-name", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--openvla-src", default="")
    ap.add_argument("--vla-path", default="openvla/openvla-7b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--save-steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=5e-4)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--use-quantization", action="store_true")
    ap.add_argument("--image-aug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--task-mode", choices=("raw", "canonical", "mixed"), default="mixed")
    ap.add_argument("--task-canonical-p", type=float, default=0.8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--video-backend", default="pyav")
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="Load model/dataset and run one forward/backward microbatch.")
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument("--push-intermediate", action="store_true")
    ap.add_argument("--hub-repo-id", default="")
    ap.add_argument("--public", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    recipe = get_recipe(args.recipe)
    if not args.job_name:
        args.job_name = recipe.job_ids["openvla"]
    if not args.output_dir:
        args.output_dir = str(ROOT / "outputs" / "train" / args.job_name)
    if args.push_to_hub and not args.hub_repo_id:
        args.hub_repo_id = f"RobotLearningVLA/{args.job_name}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    deps = register_openvla(args.openvla_src)

    print(f">> Eval3 OpenVLA LoRA train: recipe={recipe.name} job={args.job_name}", flush=True)
    print(f"   output={out_dir}", flush=True)
    print(f"   chunk action dim={args.chunk_size}x7 flattened to {args.chunk_size * 7} generated tokens", flush=True)
    print("   data labels: exact raw labels, no gripper repair, no action smoothing, no extra cap", flush=True)

    stats = compute_openvla_chunk_statistics(recipe, chunk_size=args.chunk_size, revision=args.revision, video_backend=args.video_backend)
    if stats["num_transitions"] != recipe.expected_frames:
        raise RuntimeError(f"{recipe.name} has {stats['num_transitions']} frames, expected {recipe.expected_frames}")
    (out_dir / "dataset_statistics.json").write_text(json.dumps(_stats_payload(stats), indent=2) + "\n")

    processor, model = build_model(args, deps, stats, device)
    action_tokenizer = deps["ActionTokenizer"](processor.tokenizer)
    prompt_builder_fn = deps["PurePromptBuilder"] if "v01" not in args.vla_path else deps["VicunaV15ChatPromptBuilder"]
    dataset = Eval3OpenVLADataset(
        recipe=recipe.name,
        stats=stats,
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder_fn,
        chunk_size=args.chunk_size,
        task_mode=args.task_mode,
        task_canonical_p=args.task_canonical_p,
        image_aug=args.image_aug,
        revision=args.revision,
        video_backend=args.video_backend,
    )
    collator = deps["PaddedCollatorForActionPrediction"](
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    model.train()
    data_iter = cycle(loader)
    global_step = 0
    last_loss = float("nan")
    target_steps = 1 if args.dry_run else args.steps
    while global_step < target_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accumulation_steps):
            batch = move_batch(next(data_iter), device=device, dtype=dtype)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = output.loss / args.grad_accumulation_steps
            loss.backward()
            accum_loss += float(loss.detach().cpu().item())
        optimizer.step()
        global_step += 1
        last_loss = accum_loss

        if global_step == 1 or global_step % args.log_steps == 0:
            print(f"step={global_step:06d} loss={last_loss:.6f}", flush=True)
        if not args.dry_run and global_step % args.save_steps == 0:
            ckpt_dir = out_dir / "checkpoints" / f"{global_step:06d}"
            save_checkpoint(checkpoint_dir=ckpt_dir, model=model, processor=processor, optimizer=optimizer, step=global_step, args=args, stats=stats)
            print(f"saved {ckpt_dir}", flush=True)
            if args.push_to_hub and args.push_intermediate:
                maybe_upload(ckpt_dir, f"{args.hub_repo_id}-{global_step//1000}k", private=not args.public)

    final_dir = out_dir / "checkpoints" / f"{global_step:06d}"
    save_checkpoint(checkpoint_dir=final_dir, model=model, processor=processor, optimizer=optimizer, step=global_step, args=args, stats=stats)
    (out_dir / "train_summary.json").write_text(
        json.dumps({"job_name": args.job_name, "recipe": recipe.name, "steps": global_step, "last_loss": last_loss}, indent=2) + "\n"
    )
    if args.push_to_hub and not args.dry_run:
        maybe_upload(final_dir, args.hub_repo_id, private=not args.public)
    print(f"done: {final_dir}", flush=True)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()

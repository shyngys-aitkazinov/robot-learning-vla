#!/usr/bin/env python3
"""Minimal BC training loop for Eval 3 *overfit gate* (pipeline debugging).

This is **not** a compliant VLA by itself — use SmolVLA-class OSS + VL backbone for the course.
It proves: LeRobotDataset → resize → proprio + RGB → MSE on recorded actions.

Example:
  python scripts/train_eval3_bc_overfit.py \\
    --repo-id RobotLearningVLA/taylor_swift_1 --episodes 0 --steps 2000 --device cpu --video-backend pyav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize as tv_resize
from tqdm import tqdm

from eval3_models import Eval3TinyBC, find_first_image_key


class PreprocessedSubset(torch.utils.data.Dataset):
    def __init__(self, base_ds, image_key: str, state_key: str | None, image_size: int):
        self.base_ds = base_ds
        self.image_key = image_key
        self.state_key = state_key
        self.image_size = image_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        row = self.base_ds[idx]
        img = row[self.image_key]
        if img.dim() == 3 and img.shape[0] in (1, 3):
            pass
        elif img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        img = img.float() / 255.0 if img.dtype != torch.float32 else img
        if img.max() > 1.5:
            img = img / 255.0
        img = tv_resize(img, [self.image_size, self.image_size])
        if self.state_key and self.state_key in row:
            state = row[self.state_key].float().flatten()
        else:
            state = torch.zeros(1, dtype=torch.float32)
        action = row["action"].float().flatten()
        return img, state, action


def collate(batch):
    imgs, states, acts = zip(*batch, strict=True)
    return torch.stack(imgs), torch.stack(states), torch.stack(acts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/eval3_bc_overfit"))
    ap.add_argument(
        "--video-backend",
        default="pyav",
        help="LeRobot video decoder (pyav avoids torchcodec+FFmpeg dylib issues on many Macs)",
    )
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, episodes=args.episodes, video_backend=args.video_backend)
    sample = ds[0]
    image_key = find_first_image_key(sample)
    if not image_key:
        raise RuntimeError("No observation.images.* key in dataset sample")
    state_key = "observation.state" if "observation.state" in sample else None

    action_dim = int(sample["action"].numel())
    state_dim = int(sample[state_key].numel()) if state_key else 1

    wrapped = PreprocessedSubset(ds, image_key, state_key, args.image_size)
    loader = DataLoader(
        wrapped,
        batch_size=min(args.batch_size, len(wrapped)),
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )

    device = torch.device(args.device)
    model = Eval3TinyBC(
        action_dim=action_dim,
        state_dim=state_dim,
        image_size=args.image_size,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.functional.mse_loss

    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "repo_id": args.repo_id,
        "episodes": args.episodes,
        "image_key": image_key,
        "state_key": state_key,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "image_size": args.image_size,
        "model_class": "Eval3TinyBC",
    }
    meta_path = args.output_dir / "eval3_bc_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    step = 0
    pbar = tqdm(total=args.steps)
    it = iter(loader)
    while step < args.steps:
        try:
            imgs, states, acts = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, states, acts = next(it)
        imgs = imgs.to(device)
        states = states.to(device)
        acts = acts.to(device)

        pred = model(imgs, states)
        loss = loss_fn(pred, acts)
        opt.zero_grad()
        loss.backward()
        opt.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.6f}")
    pbar.close()

    ckpt_path = args.output_dir / "best.pt"
    torch.save({"model_state": model.cpu().state_dict(), "meta": meta}, ckpt_path)
    print(f"wrote {ckpt_path}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()

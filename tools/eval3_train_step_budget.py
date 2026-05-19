#!/usr/bin/env python3
"""Estimate SmolVLA training steps from frame count, batch size, and target epochs.

Mirrors the budgeting block in TongxiHu/vla_eval1 ``train_eval1_h100.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# Rough frame counts for v10 recipes (post filters / caps).
RECIPE_FRAMES: dict[str, int] = {
    "v4_balanced_only": 3 * 750 * 500,  # upper bound before synth cap
    "v4_balanced_new66": 37_000 + 3 * 75 * 500,  # new66 + capped v4 synth
    "new66": 36_865,
}


@dataclass(frozen=True)
class StepBudget:
    num_frames: int
    batch_size: int
    aug_multiplier: float
    target_epochs: float
    steps_per_epoch: int
    recommended_steps: int

    def format_table(self) -> str:
        eff_epochs = (
            self.recommended_steps / max(self.steps_per_epoch, 1) * self.aug_multiplier
        )
        return f"""
Eval3 SmolVLA step budget
-------------------------
  Frames (est.)     : {self.num_frames:,}
  Batch size        : {self.batch_size}
  Aug multiplier    : {self.aug_multiplier:.2f}
  Steps / epoch     : ~{self.steps_per_epoch}
  Target eff. epochs: {self.target_epochs:.1f}
  Recommended steps : {self.recommended_steps:,}  (~{eff_epochs:.0f} effective epochs)
"""


def compute_budget(
    *,
    num_frames: int,
    batch_size: int,
    aug_multiplier: float = 2.2,
    target_epochs: float = 67.0,
) -> StepBudget:
    batch_size = max(int(batch_size), 1)
    steps_per_epoch = max(int(num_frames) // batch_size, 1)
    recommended = int(steps_per_epoch * float(target_epochs) / max(float(aug_multiplier), 1e-6))
    recommended = max(recommended, steps_per_epoch)
    return StepBudget(
        num_frames=int(num_frames),
        batch_size=batch_size,
        aug_multiplier=float(aug_multiplier),
        target_epochs=float(target_epochs),
        steps_per_epoch=steps_per_epoch,
        recommended_steps=recommended,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", choices=sorted(RECIPE_FRAMES), default="v4_balanced_new66")
    ap.add_argument("--num-frames", type=int, default=0, help="Override recipe frame estimate.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--aug-multiplier", type=float, default=2.2)
    ap.add_argument("--target-epochs", type=float, default=67.0)
    ap.add_argument("--step-rate", type=float, default=0.0, help="Optional steps/s for ETA line.")
    args = ap.parse_args()

    frames = int(args.num_frames) if args.num_frames > 0 else RECIPE_FRAMES[args.recipe]
    budget = compute_budget(
        num_frames=frames,
        batch_size=args.batch_size,
        aug_multiplier=args.aug_multiplier,
        target_epochs=args.target_epochs,
    )
    print(budget.format_table())
    if args.step_rate > 0:
        hours = budget.recommended_steps / args.step_rate / 3600.0
        print(f"  ETA @ {args.step_rate:.1f} step/s : ~{hours:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

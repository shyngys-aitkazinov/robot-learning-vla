#!/usr/bin/env python3
"""Watch a slot-bottleneck training run and evaluate every checkpoint.

Polls ``<output_dir>/checkpoints/``. For each new step-checkpoint it loads the
policy and, on the three Taylor scenes (left / middle / right), with the
proprioceptive state **zeroed** (so the policy cannot ride proprio-continuation
— exactly the deploy cold-start condition), predicts the action under all three
celebrity prompts. It reports, per scene:

  * predicted shoulder_pan under each prompt  (+pan = LEFT, -pan = RIGHT)
  * the slot the correct-celebrity prompt rotates toward, vs the true slot
  * cross-prompt Δ_max (healthy >= ~20 deg)
  * the slot-classifier argmax (from the prefix head)

Output goes to stdout only — the launcher redirects it to a watch log. This
watcher never creates or writes inside the training output dir (doing so would
trip lerobot's "output dir already exists" guard).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set BEFORE importing the patch so apply() installs the SlotClassifier
# (otherwise the checkpoint's slot_clf.* weights have nowhere to load).
os.environ.setdefault("EVAL3_SLOT_LOSS_WEIGHT", "0.5")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

import eval3_lerobot_shim  # noqa: E402

eval3_lerobot_shim.apply()
import eval3_smolvla_slot_bottleneck  # noqa: E402

eval3_smolvla_slot_bottleneck.apply()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval3_abcd_benchmark import load_policy_bundle, predict  # noqa: E402

POS = ["left", "middle", "right"]
PROMPTS = {
    "taylor": "Place the coke on Taylor Swift",
    "obama": "Place the coke on Barack Obama",
    "lecun": "Place the coke on Yann LeCun",
}
STATS_REPO = "RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"


def classify_pan(p: float) -> str:
    if p > 12.0:
        return "left"
    if p < -12.0:
        return "right"
    return "middle"


def log(msg: str) -> None:
    # stdout only — the launcher redirects this to the watch log file.
    print(msg, flush=True)


def eval_checkpoint(ckpt: str, device: str, datasets_root: Path) -> None:
    bundle = load_policy_bundle(ckpt, device, STATS_REPO)
    policy = bundle[0]
    cap: dict = {}
    h = None
    if hasattr(policy.model, "slot_clf"):
        h = policy.model.slot_clf.register_forward_hook(
            lambda m, i, o: cap.__setitem__("logits", o[0].detach())
        )

    log(f"  {'scene':>8s} {'true':>7s} | "
        + "  ".join(f"{c:>14s}" for c in PROMPTS)
        + f" | {'Δmax':>6s}  {'corr->slot':>12s}  {'slotclf':>8s}")
    log("  " + "-" * 92)

    deltas = []
    correct_hits = 0
    for pos in POS:
        name = f"dataset_v3_synth_pinned_idood_taylor_swift_{pos}_2"
        ds = LeRobotDataset(
            f"RobotLearningVLA/{name}", root=str(datasets_root / name),
            episodes=[0], video_backend="pyav",
        )
        row = dict(ds[0])
        # Zero the proprio: the policy must choose direction from image+prompt.
        row["observation.state"] = torch.zeros_like(ds[0]["observation.state"])
        pans = {}
        slotclf = "?"
        for celeb, prompt in PROMPTS.items():
            cap.pop("logits", None)
            pans[celeb] = float(predict(bundle, row, prompt)[0])
            if celeb == "taylor" and "logits" in cap:
                slotclf = POS[int(cap["logits"][0].argmax())]
        spread = max(pans.values()) - min(pans.values())
        deltas.append(spread)
        corr_slot = classify_pan(pans["taylor"])  # correct prompt for a taylor scene
        hit = "OK" if corr_slot == pos else "XX"
        if corr_slot == pos:
            correct_hits += 1
        log(f"  {pos:>8s} {pos:>7s} | "
            + "  ".join(f"{pans[c]:>+14.1f}" for c in PROMPTS)
            + f" | {spread:>6.1f}  {corr_slot:>12s}{hit:>3s}  {slotclf:>8s}")

    mean_delta = sum(deltas) / len(deltas)
    log(f"  => correct-prompt slot hits {correct_hits}/3   mean Δmax {mean_delta:.1f}deg"
        f"   (healthy: hits 3/3, Δ>=20)")
    if h is not None:
        h.remove()
    del bundle, policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, help="train output dir (contains checkpoints/)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets-root", default=str(_REPO / "datasets"))
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--final-step", type=int, default=5000, help="stop after this checkpoint is evaluated")
    ap.add_argument("--max-idle-min", type=int, default=90, help="give up after this long with no new checkpoint")
    args = ap.parse_args()

    ckpt_root = Path(args.output_dir) / "checkpoints"
    datasets_root = Path(args.datasets_root)

    log(f"[slot-eval-watcher] watching {ckpt_root}  (final_step={args.final_step})")
    seen: set[int] = set()
    last_new = time.time()

    while True:
        steps = []
        if ckpt_root.is_dir():
            for d in sorted(ckpt_root.iterdir()):
                if d.name.isdigit() and (d / "pretrained_model").is_dir():
                    steps.append(int(d.name))
        for step in sorted(steps):
            if step in seen:
                continue
            seen.add(step)
            last_new = time.time()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            log(f"\n=== checkpoint step {step}  ({ts}) ===")
            try:
                eval_checkpoint(
                    str(ckpt_root / f"{step:06d}" / "pretrained_model"),
                    args.device, datasets_root,
                )
            except Exception as exc:  # keep watching even if one eval fails
                log(f"  eval error: {type(exc).__name__}: {exc}")
            if step >= args.final_step:
                log("[slot-eval-watcher] final step reached — exiting.")
                return 0
        if (time.time() - last_new) > args.max_idle_min * 60:
            log(f"[slot-eval-watcher] no new checkpoint for {args.max_idle_min} min — exiting.")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

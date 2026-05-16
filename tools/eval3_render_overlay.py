#!/usr/bin/env python3
"""Render mp4 videos of dataset episodes with the policy's predicted action overlaid.

Layout per video frame:
   ┌──────────────────────────┬─────────────────────────────────────────┐
   │                          │   shoulder_pan  ░░░░░░ GT  ███ Pred     │
   │                          │   shoulder_lift                          │
   │   Camera frame (480x640) │   elbow_flex                             │
   │                          │   wrist_flex                             │
   │                          │   wrist_roll                             │
   │                          │   gripper                                │
   │                          │                                          │
   │   Episode + prompt info  │   per-frame MAE: 3.42 deg                │
   └──────────────────────────┴─────────────────────────────────────────┘

Each bar shows the joint's value within its full action range (from the dataset analysis):
   gray  = current proprio state
   blue  = ground-truth recorded action
   red   = policy's predicted action

The closer the red bar is to the blue bar, the better the model is tracking the demo.

Inference is run with action chunking left ON (no policy.reset between frames),
so this faithfully mimics deploy-time behavior — 1 inference event per ~50 frames,
queue-popped actions in between.
"""
from __future__ import annotations
import argparse, sys, os, time
from pathlib import Path

import numpy as np
import torch
import imageio_ffmpeg
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eval3_lerobot_shim import apply as _shim  # noqa: E402

_shim()

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.feature_utils import build_dataset_frame  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import make_robot_action  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from lerobot.utils.constants import OBS_STR  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Display-range envelope per joint (from the deep-audit step) — used to lay out bars.
# (min_for_display, max_for_display)
JOINT_RANGES = {
    "shoulder_pan":  (-50.0,   50.0),
    "shoulder_lift": (-110.0,  100.0),
    "elbow_flex":    (-80.0,  110.0),
    "wrist_flex":    (-110.0,  50.0),
    "wrist_roll":    (-110.0, 120.0),  # spans Swift (-98) and LeCun (+112) ranges
    "gripper":       (   0.0,  80.0),
}


def _load_font(size: int):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_joint_panel(panel_w: int, panel_h: int, state, gt_action, pred_action,
                     episode_label: str, prompt: str, t_s: float,
                     frame_mae: float, grasp_frame: int | None,
                     release_frame: int | None, frame_i: int, total_frames: int) -> Image.Image:
    img = Image.new("RGB", (panel_w, panel_h), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    font_big = _load_font(22)
    font_mid = _load_font(18)
    font_small = _load_font(14)

    # ----- Header -----
    draw.text((12, 8),  f"{episode_label}    t = {t_s:5.2f}s    frame {frame_i}/{total_frames}",
              fill=(0, 0, 0), font=font_mid)
    draw.text((12, 30), f"prompt: {prompt}", fill=(60, 60, 60), font=font_small)

    # ----- Phase pill -----
    phase = "approach"
    if grasp_frame is not None and frame_i >= grasp_frame:
        phase = "transport"
        if grasp_frame is not None and frame_i < grasp_frame + 30:
            phase = "grasp"
    if release_frame is not None and frame_i >= release_frame - 15:
        phase = "release"
    phase_colors = {"approach": (200, 200, 255), "grasp": (255, 230, 180),
                    "transport": (200, 240, 200), "release": (255, 200, 200)}
    px, py, pw, ph = panel_w - 130, 8, 118, 26
    draw.rectangle([px, py, px + pw, py + ph], fill=phase_colors[phase], outline=(80, 80, 80))
    draw.text((px + 12, py + 4), f"PHASE: {phase}", fill=(0, 0, 0), font=font_small)

    # ----- Frame-level MAE pill -----
    color = (60, 180, 60) if frame_mae < 4 else ((220, 160, 0) if frame_mae < 8 else (210, 60, 60))
    draw.rectangle([12, 50, 240, 76], fill=(255, 255, 255), outline=color)
    draw.text((20, 54), f"frame MAE = {frame_mae:5.2f} deg", fill=color, font=font_mid)

    # ----- Per-joint bars -----
    top = 90
    row_h = (panel_h - top - 20) // len(JOINTS)
    label_w = 130
    bar_x0 = 12 + label_w
    bar_x1 = panel_w - 12
    bar_w = bar_x1 - bar_x0

    for j, jn in enumerate(JOINTS):
        y = top + j * row_h
        # Label
        draw.text((12, y + row_h // 2 - 10), jn, fill=(0, 0, 0), font=font_small)
        # Bar track (background)
        track_top = y + 4
        track_bot = y + row_h - 4
        draw.rectangle([bar_x0, track_top, bar_x1, track_bot], fill=(225, 225, 225), outline=(180, 180, 180))
        # Tick marks
        lo, hi = JOINT_RANGES[jn]
        # Draw zero tick if zero in range
        if lo <= 0 <= hi:
            zx = bar_x0 + int((0 - lo) / (hi - lo) * bar_w)
            draw.line([(zx, track_top), (zx, track_bot)], fill=(150, 150, 150), width=1)
        # Map a value (deg) to x pixel
        def to_x(v):
            v_clipped = max(lo, min(hi, float(v)))
            return bar_x0 + int((v_clipped - lo) / (hi - lo) * bar_w)
        # State (gray triangle)
        sx = to_x(state[j])
        draw.polygon([(sx - 6, track_top - 4), (sx + 6, track_top - 4), (sx, track_top + 2)],
                     fill=(120, 120, 120))
        # GT action (blue line + marker)
        gx = to_x(gt_action[j])
        draw.line([(gx, track_top), (gx, track_bot)], fill=(50, 100, 220), width=3)
        draw.ellipse([gx - 5, track_top - 5, gx + 5, track_top + 5], fill=(50, 100, 220))
        # Pred action (red line + marker)
        px2 = to_x(pred_action[j])
        draw.line([(px2, track_top), (px2, track_bot)], fill=(220, 40, 40), width=3)
        draw.ellipse([px2 - 5, track_bot - 5, px2 + 5, track_bot + 5], fill=(220, 40, 40))
        # Tail showing distance between GT and pred
        if abs(gt_action[j] - pred_action[j]) > 0.5:
            draw.line([(gx, track_top + 8), (px2, track_top + 8)], fill=(180, 100, 100), width=1)
        # Numeric values to right
        draw.text((bar_x1 + 4, y + 2),
                  f"GT {float(gt_action[j]):+6.1f}", fill=(50, 100, 220), font=font_small)
        # (avoid drawing pred outside panel — skip if cramped)

    # Legend at bottom
    yL = panel_h - 18
    draw.text((12, yL), "▶ state (gray)   ● GT action (blue)   ● Pred action (red)",
              fill=(80, 80, 80), font=font_small)
    return img


def render_episode_video(repo: str, ep_idx: int, label: str, prompt: str,
                         policy, preprocessor, postprocessor,
                         device: str, fps: int, out_path: Path) -> dict:
    print(f"  loading {repo} episode {ep_idx}...")
    ds = LeRobotDataset(repo, video_backend="pyav", episodes=[ep_idx])
    ds_features = ds.meta.features
    n = len(ds)
    full_actions = np.stack([ds[i]["action"].numpy() for i in range(n)])
    full_states = np.stack([ds[i]["observation.state"].numpy() for i in range(n)])

    # Detect grasp/release for the phase pill
    g = full_actions[:, JOINTS.index("gripper")]
    grasp_frame = None
    for i in range(n - 10):
        if g[i] > 25 and (g[i:i + 10] > 20).all():
            grasp_frame = i; break
    release_frame = None
    if grasp_frame is not None:
        for i in range(n - 1, grasp_frame + 10, -1):
            if g[i] > 25 and (g[max(i - 10, 0):i] > 20).all():
                release_frame = i; break

    # Reset the policy ONCE at episode start; let action queue carry between frames (= deploy behavior).
    policy.reset(); preprocessor.reset(); postprocessor.reset()
    predicted = np.zeros_like(full_actions)
    t0 = time.time()
    last_print = t0
    with torch.inference_mode():
        for i in range(n):
            row = ds[i]
            state = row["observation.state"].numpy()
            img_chw = row["observation.images.front"].numpy()
            img_hwc = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            obs = {f"{nm}.pos": float(state[j]) for j, nm in enumerate(JOINTS)}
            obs["front"] = img_hwc
            obs_frame = build_dataset_frame(ds_features, obs, prefix=OBS_STR)
            a_tensor = predict_action(
                observation=obs_frame, policy=policy, device=torch.device(device),
                preprocessor=preprocessor, postprocessor=postprocessor,
                use_amp=policy.config.use_amp, task=prompt, robot_type="so101_follower",
            )
            a_dict = make_robot_action(a_tensor, ds_features)
            predicted[i] = np.array([a_dict[f"{nm}.pos"] for nm in JOINTS])
            if time.time() - last_print > 5:
                print(f"    inferred {i+1}/{n} frames ({(i+1)/n*100:.1f}%)")
                last_print = time.time()
    elapsed = time.time() - t0
    print(f"  inference done: {n} frames in {elapsed:.1f}s ({n/elapsed:.1f} fps)")

    # Now render frames + encode
    CAM_W, CAM_H = 640, 480
    PANEL_W = 720
    OUT_W = CAM_W + PANEL_W
    OUT_H = CAM_H
    print(f"  writing video {OUT_W}x{OUT_H} @ {fps}fps -> {out_path}")
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                ffmpeg_log_level="error", macro_block_size=8)
    t0 = time.time()
    for i in range(n):
        # Camera frame
        row = ds[i]
        img_chw = row["observation.images.front"].numpy()
        cam_img = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        cam_pil = Image.fromarray(cam_img)
        # Joint panel
        frame_mae = float(np.abs(predicted[i] - full_actions[i]).mean())
        panel = draw_joint_panel(PANEL_W, OUT_H, full_states[i], full_actions[i], predicted[i],
                                  f"{label} ep{ep_idx}", prompt, i / fps, frame_mae,
                                  grasp_frame, release_frame, i, n)
        composite = Image.new("RGB", (OUT_W, OUT_H), (0, 0, 0))
        composite.paste(cam_pil, (0, 0))
        composite.paste(panel, (CAM_W, 0))
        writer.append_data(np.array(composite))
    writer.close()
    elapsed = time.time() - t0
    print(f"  rendered + encoded in {elapsed:.1f}s")

    return {
        "n_frames": n,
        "overall_mae": float(np.abs(predicted - full_actions).mean()),
        "grasp_frame": grasp_frame,
        "release_frame": release_frame,
        "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/train/eval3_swift_lecun_8k/checkpoints/008000/pretrained_model")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out-dir", default="outputs/eval3_deep_analysis/videos")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    rename_map = {"observation.images.front": "observation.images.camera1"}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading policy...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    policy_cfg.device = args.device
    ds_meta = LeRobotDatasetMetadata("RobotLearningVLA/taylor_swift_1")
    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.ckpt,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    policy.eval()
    print(f"Policy loaded ({sum(p.numel() for p in policy.parameters())/1e6:.0f}M params, device={args.device})\n")

    EPISODES = [
        ("RobotLearningVLA/taylor_swift_1", "Swift", "Place the coke on the Taylor Swift", 0),
        ("RobotLearningVLA/taylor_swift_1", "Swift", "Place the coke on the Taylor Swift", 5),
        ("RobotLearningVLA/yann_lecun_1",   "LeCun", "Place the coke on the Yann LeCun", 0),
        ("RobotLearningVLA/yann_lecun_1",   "LeCun", "Place the coke on the Yann LeCun", 5),
    ]
    summary = []
    for repo, label, prompt, ep in EPISODES:
        fname = f"{label.lower()}_ep{ep}.mp4"
        info = render_episode_video(repo, ep, label, prompt,
                                     policy, preprocessor, postprocessor,
                                     args.device, args.fps, out_dir / fname)
        summary.append((fname, info))
        print()

    print("\n=== Summary ===")
    for fn, info in summary:
        print(f"  {fn:30s}  {info['n_frames']:4d} frames   overall MAE = {info['overall_mae']:5.2f} deg")
    print(f"\nVideos saved under {out_dir.resolve()}/")


if __name__ == "__main__":
    main()

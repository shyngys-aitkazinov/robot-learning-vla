#!/usr/bin/env python3
"""Export playable videos for Eval 3 v6 truncated dataset_v2 episodes.

This materializes the in-memory v6 truncation rule into MP4 review clips. Each
clip contains only frames kept for training:

  first frame through first placement event + buffer_frames

Default rule:
  last frame where gripper >= 20 and shoulder_lift >= -30 before the final
  home-return tail, capped at max_frames_per_episode.

Outputs:
  outputs/eval3_truncated_videos/
    index.html
    manifest.json
    <repo_slug>/ep_000_truncated.mp4
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


DEFAULT_OVER_CAP_EPISODES = (
    "dataset_v2_barack_obama_middle_1:0,5,7;"
    "dataset_v2_taylor_swift_left_1:1,3;"
    "dataset_v2_yann_lecun_left_1:1"
)

DEFAULT_NEW_REPOS = [
    "RobotLearningVLA/dataset_v2_taylor_swift_left_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_middle_1",
    "RobotLearningVLA/dataset_v2_taylor_swift_right_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_left_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_middle_1",
    "RobotLearningVLA/dataset_v2_yann_lecun_right_1",
    "RobotLearningVLA/dataset_v2_barack_obama_left_1",
    "RobotLearningVLA/dataset_v2_barack_obama_middle_1",
    "RobotLearningVLA/dataset_v2_barack_obama_right_1",
]

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
LIFT_IDX = JOINTS.index("shoulder_lift")
WRIST_ROLL_IDX = JOINTS.index("wrist_roll")
GRIPPER_IDX = JOINTS.index("gripper")


@dataclass
class ClipInfo:
    repo: str
    repo_slug: str
    episode: int
    raw_frames: int
    kept_frames: int
    match_frame: int | None
    fps: int
    duration_s: float
    match_shoulder_lift: float | None
    match_gripper: float | None
    match_wrist_roll: float | None
    end_shoulder_lift: float
    end_gripper: float
    end_wrist_roll: float
    original_end_shoulder_lift: float
    original_end_gripper: float
    original_end_wrist_roll: float
    video: str


def _repo_slug(repo: str) -> str:
    return repo.rsplit("/", 1)[-1]


def _load_font(size: int):
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _action_at(actions: list[Any], global_idx: int) -> list[float]:
    value = actions[global_idx]
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().cpu().flatten().tolist()]
    return [float(x) for x in np.asarray(value).reshape(-1).tolist()]


def _find_placement_match(
    actions: list[Any],
    f0: int,
    f1: int,
    grip_threshold: float,
    lift_threshold: float,
    placement_mode: str,
) -> int | None:
    matches = []
    for off in range(f1 - f0):
        action = _action_at(actions, f0 + off)
        if action[GRIPPER_IDX] >= grip_threshold and action[LIFT_IDX] >= lift_threshold:
            matches.append(off)
            if placement_mode == "first":
                break
    if not matches:
        return None
    return matches[0] if placement_mode == "first" else matches[-1]


def _parse_repo_episode_map(raw: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        repo_key, eps_raw = part.split(":", 1)
        out[repo_key.strip()] = {int(x) for x in eps_raw.split(",") if x.strip()}
    return out


def _tensor_image_to_pil(img: Any) -> Image.Image:
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.dtype.kind == "f":
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    else:
        arr = np.asarray(img)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def _draw_overlay(
    frame: Image.Image,
    repo_slug: str,
    episode: int,
    frame_i: int,
    kept_frames: int,
    raw_frames: int,
    match_frame: int | None,
    action: list[float],
    is_match: bool,
    is_trunc_end: bool,
) -> np.ndarray:
    frame = frame.convert("RGB")
    w, h = frame.size
    header_h = 80
    out = Image.new("RGB", (w, h + header_h), (245, 245, 245))
    out.paste(frame, (0, header_h))
    draw = ImageDraw.Draw(out)
    font = _load_font(16)
    small = _load_font(13)

    draw.rectangle([0, 0, w, header_h], fill=(20, 20, 20))
    title = (
        f"{repo_slug}  ep {episode:03d}  kept frame {frame_i + 1}/{kept_frames} "
        f"(raw {raw_frames})"
    )
    draw.text((10, 8), title, fill=(255, 255, 255), font=font)
    subtitle = (
        f"Rule A: first gripper>=20 and shoulder_lift>=-30, +60 frames. "
        f"match={match_frame}, trunc_end={kept_frames - 1}"
    )
    draw.text((10, 34), subtitle, fill=(220, 220, 220), font=small)
    joints = (
        f"lift={action[LIFT_IDX]:+.1f}  gripper={action[GRIPPER_IDX]:+.1f}  "
        f"wrist_roll={action[WRIST_ROLL_IDX]:+.1f}"
    )
    draw.text((10, 56), joints, fill=(220, 220, 220), font=small)

    tags = []
    if is_match:
        tags.append("FIRST MATCH")
    if is_trunc_end:
        tags.append("TRUNCATED END")
    if tags:
        text = " / ".join(tags)
        bbox = draw.textbbox((0, 0), text, font=font)
        pad = 8
        x2 = w - 10
        x1 = x2 - (bbox[2] - bbox[0]) - 2 * pad
        y1, y2 = 8, 36
        draw.rectangle([x1, y1, x2, y2], fill=(30, 130, 55))
        draw.text((x1 + pad, y1 + 5), text, fill=(255, 255, 255), font=small)

    return np.asarray(out)


def _export_clip(
    ds: LeRobotDataset,
    actions: list[Any],
    repo: str,
    episode: int,
    f0: int,
    f1: int,
    match_frame: int | None,
    kept_frames: int,
    out_path: Path,
    fps: int,
    overwrite: bool,
) -> ClipInfo:
    repo_slug = _repo_slug(repo)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"    exists: {out_path.name}")
    else:
        print(f"    writing {out_path.name} ({kept_frames} frames)")
        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=8,
            ffmpeg_log_level="error",
            macro_block_size=8,
        )
        try:
            for frame_i in range(kept_frames):
                global_idx = f0 + frame_i
                row = ds[global_idx]
                frame = _tensor_image_to_pil(row["observation.images.front"])
                action = _action_at(actions, global_idx)
                writer.append_data(
                    _draw_overlay(
                        frame=frame,
                        repo_slug=repo_slug,
                        episode=episode,
                        frame_i=frame_i,
                        kept_frames=kept_frames,
                        raw_frames=f1 - f0,
                        match_frame=match_frame,
                        action=action,
                        is_match=match_frame is not None and frame_i == match_frame,
                        is_trunc_end=frame_i == kept_frames - 1,
                    )
                )
        finally:
            writer.close()

    match_action = _action_at(actions, f0 + match_frame) if match_frame is not None else None
    end_action = _action_at(actions, f0 + kept_frames - 1)
    original_end_action = _action_at(actions, f1 - 1)
    return ClipInfo(
        repo=repo,
        repo_slug=repo_slug,
        episode=episode,
        raw_frames=f1 - f0,
        kept_frames=kept_frames,
        match_frame=match_frame,
        fps=fps,
        duration_s=kept_frames / fps,
        match_shoulder_lift=match_action[LIFT_IDX] if match_action is not None else None,
        match_gripper=match_action[GRIPPER_IDX] if match_action is not None else None,
        match_wrist_roll=match_action[WRIST_ROLL_IDX] if match_action is not None else None,
        end_shoulder_lift=end_action[LIFT_IDX],
        end_gripper=end_action[GRIPPER_IDX],
        end_wrist_roll=end_action[WRIST_ROLL_IDX],
        original_end_shoulder_lift=original_end_action[LIFT_IDX],
        original_end_gripper=original_end_action[GRIPPER_IDX],
        original_end_wrist_roll=original_end_action[WRIST_ROLL_IDX],
        video=str(out_path.relative_to(out_path.parents[1])),
    )


def _write_index(out_dir: Path, clips: list[ClipInfo]) -> None:
    by_repo: dict[str, list[ClipInfo]] = {}
    for clip in clips:
        by_repo.setdefault(clip.repo_slug, []).append(clip)

    parts = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Eval 3 v6 Truncated Episode Videos</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;background:#f7f7f7;color:#111}",
        "h1{font-size:24px;margin:0 0 8px}",
        "h2{font-size:18px;margin:28px 0 10px}",
        ".meta{color:#555;margin-bottom:16px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px}",
        "video{width:100%;background:#000;border-radius:4px}",
        "table{border-collapse:collapse;margin:12px 0 20px;background:#fff}",
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:right;font-size:13px}",
        "th:first-child,td:first-child{text-align:left}",
        ".ok{color:#087a2f;font-weight:600}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Eval 3 v6 Truncated Episode Videos</h1>",
        '<div class="meta">',
        f'<span class="ok">Generated {len(clips)} clips.</span> ',
        "Each clip contains the training-kept prefix using the selected placement rule.",
        "</div>",
        "<table>",
        "<tr><th>dataset</th><th>episodes</th><th>kept min</th><th>kept max</th><th>end lift min</th><th>end gripper min</th></tr>",
    ]
    for repo_slug, repo_clips in by_repo.items():
        parts.append(
            "<tr>"
            f"<td>{html.escape(repo_slug)}</td>"
            f"<td>{len(repo_clips)}</td>"
            f"<td>{min(c.kept_frames for c in repo_clips)}</td>"
            f"<td>{max(c.kept_frames for c in repo_clips)}</td>"
            f"<td>{min(c.end_shoulder_lift for c in repo_clips):+.1f}</td>"
            f"<td>{min(c.end_gripper for c in repo_clips):+.1f}</td>"
            "</tr>"
        )
    parts.append("</table>")

    for repo_slug, repo_clips in by_repo.items():
        parts.append(f"<h2>{html.escape(repo_slug)}</h2>")
        parts.append('<div class="grid">')
        for clip in repo_clips:
            video_rel = html.escape(clip.video)
            parts.append('<div class="card">')
            parts.append(
                f"<strong>episode {clip.episode}</strong><br>"
                f"raw={clip.raw_frames}, kept={clip.kept_frames}, "
                f"match={clip.match_frame}, duration={clip.duration_s:.1f}s<br>"
                f"end lift={clip.end_shoulder_lift:+.1f}, "
                f"gripper={clip.end_gripper:+.1f}, roll={clip.end_wrist_roll:+.1f}"
            )
            parts.append(f'<video controls preload="metadata" src="{video_rel}"></video>')
            parts.append(f'<div><a href="{video_rel}">open mp4</a></div>')
            parts.append("</div>")
        parts.append("</div>")

    parts.extend(["</body>", "</html>"])
    (out_dir / "index.html").write_text("\n".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos",
        default=",".join(DEFAULT_NEW_REPOS),
        help="Comma-separated dataset_v2 repo ids.",
    )
    parser.add_argument("--out-dir", default="outputs/eval3_truncated_videos")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grip-threshold", type=float, default=20.0)
    parser.add_argument("--lift-threshold", type=float, default=-30.0)
    parser.add_argument("--placement-mode", choices=["first", "last"], default="last")
    parser.add_argument("--allow-over-cap-episodes", default=DEFAULT_OVER_CAP_EPISODES)
    parser.add_argument("--over-cap-extra-frames", type=int, default=90)
    parser.add_argument("--buffer-frames", type=int, default=60)
    parser.add_argument("--max-frames-per-episode", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repos = [repo.strip() for repo in args.repos.split(",") if repo.strip()]
    over_cap_map = _parse_repo_episode_map(args.allow_over_cap_episodes)
    all_clips: list[ClipInfo] = []

    for repo in repos:
        repo_slug = _repo_slug(repo)
        over_cap_eps = over_cap_map.get(repo_slug, set()) | over_cap_map.get(repo, set())
        print(f"loading {repo}...")
        ds = LeRobotDataset(repo, video_backend=args.video_backend)
        actions = ds.hf_dataset["action"]
        ep_df = ds.meta.episodes
        boundaries = [
            (int(f0), int(f1))
            for f0, f1 in zip(ep_df["dataset_from_index"], ep_df["dataset_to_index"])
        ]
        repo_clips: list[ClipInfo] = []
        for ep_idx, (f0, f1) in enumerate(boundaries):
            match_frame = _find_placement_match(
                actions,
                f0,
                f1,
                grip_threshold=args.grip_threshold,
                lift_threshold=args.lift_threshold,
                placement_mode=args.placement_mode,
            )
            if match_frame is None:
                kept_frames = min(args.max_frames_per_episode, f1 - f0)
            elif args.placement_mode == "first":
                kept_frames = min(match_frame + args.buffer_frames, f1 - f0)
                extra = args.over_cap_extra_frames if ep_idx in over_cap_eps else 0
                kept_frames = min(kept_frames, args.max_frames_per_episode + extra)
            else:
                kept_frames = min(match_frame + 1, f1 - f0)
                extra = args.over_cap_extra_frames if ep_idx in over_cap_eps else 0
                kept_frames = min(kept_frames, args.max_frames_per_episode + extra)
            clip = _export_clip(
                ds=ds,
                actions=actions,
                repo=repo,
                episode=ep_idx,
                f0=f0,
                f1=f1,
                match_frame=match_frame,
                kept_frames=kept_frames,
                out_path=out_dir / repo_slug / f"ep_{ep_idx:03d}_truncated.mp4",
                fps=args.fps,
                overwrite=args.overwrite,
            )
            repo_clips.append(clip)
            all_clips.append(clip)
        print(
            f"  wrote {len(repo_clips)} clips for {repo_slug}: "
            f"kept {min(c.kept_frames for c in repo_clips)}-"
            f"{max(c.kept_frames for c in repo_clips)} frames"
        )

    manifest = {
        "rule": {
            "grip_threshold": args.grip_threshold,
            "lift_threshold": args.lift_threshold,
            "placement_mode": args.placement_mode,
            "allow_over_cap_episodes": args.allow_over_cap_episodes,
            "over_cap_extra_frames": args.over_cap_extra_frames,
            "buffer_frames": args.buffer_frames,
            "fps": args.fps,
        },
        "clips": [asdict(c) for c in all_clips],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_index(out_dir, all_clips)

    print("")
    print(f"wrote {len(all_clips)} clips")
    print(f"wrote {out_dir / 'manifest.json'}")
    print(f"wrote {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

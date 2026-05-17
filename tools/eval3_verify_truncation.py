#!/usr/bin/env python3
"""Verify Eval 3 v6 placement truncation and render visual audit sheets.

The v6 training wrapper does not materialize new datasets on disk. It wraps
each LeRobotDataset at load time and keeps frames up to the first placement
event plus a buffer. This tool independently recomputes those cut points,
checks them against the same rule, and renders contact sheets so a human can
inspect the retained endpoint.

Default rule:
  last frame where gripper >= 20 and shoulder_lift >= -30 before the final
  home-return tail, capped at max_frames_per_episode.

Outputs:
  - truncation_report.csv: one row per episode
  - truncation_summary.json: aggregate stats and failures
  - contact_sheets/*.jpg: start / match / trunc_end / original_end per episode
  - index.md: links and quick summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402

_shim_apply()

from eval3_dataset_prep import Eval3PrepDataset  # noqa: E402
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
class EpisodeCheck:
    repo: str
    episode: int
    n_frames: int
    match_frame: int | None
    kept_frames: int
    kept_fraction: float
    match_t_s: float | None
    kept_t_s: float
    match_shoulder_lift: float | None
    match_gripper: float | None
    match_wrist_roll: float | None
    end_shoulder_lift: float
    end_gripper: float
    end_wrist_roll: float
    original_end_shoulder_lift: float
    original_end_gripper: float
    original_end_wrist_roll: float
    wrapper_kept_frames: int | None
    ok_match_found: bool
    ok_kept_matches_wrapper: bool
    ok_kept_range: bool
    ok_end_pose: bool

    @property
    def ok(self) -> bool:
        return (
            self.ok_match_found
            and self.ok_kept_matches_wrapper
            and self.ok_kept_range
            and self.ok_end_pose
        )


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


def _frame_tile(
    ds: LeRobotDataset,
    global_idx: int,
    title: str,
    subtitle: str,
    tile_w: int,
    tile_h: int,
) -> Image.Image:
    row = ds[global_idx]
    img = _tensor_image_to_pil(row["observation.images.front"])
    label_h = 54
    image_h = tile_h - label_h
    img.thumbnail((tile_w, image_h), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (tile_w, tile_h), (245, 245, 245))
    x = (tile_w - img.width) // 2
    tile.paste(img, (x, label_h))

    draw = ImageDraw.Draw(tile)
    font_title = _load_font(14)
    font_small = _load_font(11)
    draw.rectangle([0, 0, tile_w, label_h], fill=(255, 255, 255))
    draw.text((8, 6), title, fill=(0, 0, 0), font=font_title)
    draw.text((8, 28), subtitle, fill=(45, 45, 45), font=font_small)
    return tile


def _render_contact_sheet(
    ds: LeRobotDataset,
    repo: str,
    ep_rows: list[EpisodeCheck],
    boundaries: list[tuple[int, int]],
    out_path: Path,
) -> None:
    tile_w, tile_h = 260, 250
    label_w = 310
    header_h = 38
    row_h = tile_h
    cols = ["start", "match", "trunc_end", "orig_end"]
    sheet_w = label_w + tile_w * len(cols)
    sheet_h = header_h + row_h * len(ep_rows)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    font_header = _load_font(17)
    font_repo = _load_font(14)
    font_label = _load_font(13)
    font_small = _load_font(11)

    draw.rectangle([0, 0, sheet_w, header_h], fill=(30, 30, 30))
    draw.text((10, 10), _repo_slug(repo), fill=(255, 255, 255), font=font_repo)
    for c, name in enumerate(cols):
        draw.text((label_w + c * tile_w + 8, 10), name, fill=(255, 255, 255), font=font_header)

    for row_i, check in enumerate(ep_rows):
        y = header_h + row_i * row_h
        f0, f1 = boundaries[check.episode]
        match_global = f0 + check.match_frame if check.match_frame is not None else f0
        trunc_end_global = f0 + check.kept_frames - 1
        orig_end_global = f1 - 1
        action_end = (
            f"end lift={check.end_shoulder_lift:+.1f} "
            f"grip={check.end_gripper:+.1f} "
            f"roll={check.end_wrist_roll:+.1f}"
        )
        status = "OK" if check.ok else "FAIL"
        status_color = (30, 120, 50) if check.ok else (180, 40, 40)
        draw.rectangle([0, y, label_w, y + row_h], fill=(250, 250, 250))
        draw.text((10, y + 10), f"episode {check.episode}  {status}", fill=status_color, font=font_label)
        draw.text((10, y + 34), f"raw={check.n_frames} kept={check.kept_frames}", fill=(0, 0, 0), font=font_small)
        draw.text((10, y + 54), f"match={check.match_frame}", fill=(0, 0, 0), font=font_small)
        draw.text((10, y + 74), action_end, fill=(0, 0, 0), font=font_small)
        draw.text(
            (10, y + 94),
            f"orig_end lift={check.original_end_shoulder_lift:+.1f} grip={check.original_end_gripper:+.1f}",
            fill=(90, 90, 90),
            font=font_small,
        )

        frames = [
            (f0, "frame 0", f"t=0.0s"),
            (
                match_global,
                f"frame {check.match_frame}",
                (
                    f"lift={check.match_shoulder_lift:+.1f} grip={check.match_gripper:+.1f}"
                    if check.match_frame is not None
                    else "no match"
                ),
            ),
            (
                trunc_end_global,
                f"frame {check.kept_frames - 1}",
                f"t={check.kept_t_s:.1f}s {action_end}",
            ),
            (
                orig_end_global,
                f"frame {check.n_frames - 1}",
                (
                    f"lift={check.original_end_shoulder_lift:+.1f} "
                    f"grip={check.original_end_gripper:+.1f}"
                ),
            ),
        ]
        for c, (idx, title, subtitle) in enumerate(frames):
            tile = _frame_tile(ds, idx, title, subtitle, tile_w, tile_h)
            sheet.paste(tile, (label_w + c * tile_w, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": math.nan, "mean": math.nan, "max": math.nan}
    return {
        "min": float(min(values)),
        "mean": float(statistics.mean(values)),
        "max": float(max(values)),
    }


def _parse_repo_episode_map(raw: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        repo_key, eps_raw = part.split(":", 1)
        out[repo_key.strip()] = {int(x) for x in eps_raw.split(",") if x.strip()}
    return out


def _check_repo(args: argparse.Namespace, repo: str, out_dir: Path) -> tuple[list[EpisodeCheck], dict]:
    ds = LeRobotDataset(repo, video_backend=args.video_backend)
    repo_slug = _repo_slug(repo)
    over_cap_map = _parse_repo_episode_map(args.allow_over_cap_episodes)
    over_cap_eps = over_cap_map.get(repo_slug, set()) | over_cap_map.get(repo, set())
    actions = ds.hf_dataset["action"]
    ep_df = ds.meta.episodes
    boundaries = [
        (int(f0), int(f1))
        for f0, f1 in zip(ep_df["dataset_from_index"], ep_df["dataset_to_index"])
    ]

    wrapper_kept_by_ep: dict[int, int] = {}
    wrapper = Eval3PrepDataset(
        ds,
        max_frames_per_episode=args.max_frames_per_episode,
        truncate_at_placement=True,
        truncate_placement_mode=args.placement_mode,
        truncate_allow_over_cap_episodes=over_cap_eps,
        truncate_over_cap_extra_frames=args.over_cap_extra_frames,
        truncate_grip_threshold=args.grip_threshold,
        truncate_lift_threshold=args.lift_threshold,
        truncate_buffer_frames=args.buffer_frames,
    )
    for log_row in wrapper._truncation_log:
        wrapper_kept_by_ep[int(log_row["ep_idx"])] = int(log_row["kept_frames"])

    checks: list[EpisodeCheck] = []
    for ep_idx, (f0, f1) in enumerate(boundaries):
        n_frames = f1 - f0
        match = _find_placement_match(
            actions,
            f0,
            f1,
            args.grip_threshold,
            args.lift_threshold,
            args.placement_mode,
        )
        if match is None:
            kept = min(args.max_frames_per_episode, n_frames) if args.max_frames_per_episode else n_frames
            match_action = None
        elif args.placement_mode == "first":
            kept = min(match + args.buffer_frames, n_frames)
            if args.max_frames_per_episode:
                extra = args.over_cap_extra_frames if ep_idx in over_cap_eps else 0
                kept = min(kept, args.max_frames_per_episode + extra)
            match_action = _action_at(actions, f0 + match)
        else:
            kept = min(match + 1, n_frames)
            if args.max_frames_per_episode:
                extra = args.over_cap_extra_frames if ep_idx in over_cap_eps else 0
                kept = min(kept, args.max_frames_per_episode + extra)
            match_action = _action_at(actions, f0 + match)
        end_action = _action_at(actions, f0 + kept - 1)
        original_end_action = _action_at(actions, f1 - 1)
        wrapper_kept = wrapper_kept_by_ep.get(ep_idx)

        checks.append(
            EpisodeCheck(
                repo=repo,
                episode=ep_idx,
                n_frames=n_frames,
                match_frame=match,
                kept_frames=kept,
                kept_fraction=kept / n_frames,
                match_t_s=(match / ds.meta.fps) if match is not None else None,
                kept_t_s=kept / ds.meta.fps,
                match_shoulder_lift=match_action[LIFT_IDX] if match_action is not None else None,
                match_gripper=match_action[GRIPPER_IDX] if match_action is not None else None,
                match_wrist_roll=match_action[WRIST_ROLL_IDX] if match_action is not None else None,
                end_shoulder_lift=end_action[LIFT_IDX],
                end_gripper=end_action[GRIPPER_IDX],
                end_wrist_roll=end_action[WRIST_ROLL_IDX],
                original_end_shoulder_lift=original_end_action[LIFT_IDX],
                original_end_gripper=original_end_action[GRIPPER_IDX],
                original_end_wrist_roll=original_end_action[WRIST_ROLL_IDX],
                wrapper_kept_frames=wrapper_kept,
                ok_match_found=match is not None,
                ok_kept_matches_wrapper=wrapper_kept == kept,
                ok_kept_range=args.min_kept_frames <= kept <= args.max_kept_frames,
                ok_end_pose=(
                    end_action[LIFT_IDX] >= args.end_lift_min
                    and end_action[GRIPPER_IDX] >= args.end_gripper_min
                ),
            )
        )

    if args.render_contact_sheets:
        _render_contact_sheet(
            ds,
            repo,
            checks,
            boundaries,
            out_dir / "contact_sheets" / f"{_repo_slug(repo)}.jpg",
        )

    repo_summary = {
        "repo": repo,
        "episodes": len(checks),
        "ok": all(c.ok for c in checks),
        "failures": [
            asdict(c) | {"ok": c.ok}
            for c in checks
            if not c.ok
        ],
        "n_frames": _stats([c.n_frames for c in checks]),
        "match_frame": _stats([c.match_frame for c in checks if c.match_frame is not None]),
        "kept_frames": _stats([c.kept_frames for c in checks]),
        "kept_fraction": _stats([c.kept_fraction for c in checks]),
        "end_shoulder_lift": _stats([c.end_shoulder_lift for c in checks]),
        "end_gripper": _stats([c.end_gripper for c in checks]),
        "original_end_shoulder_lift": _stats([c.original_end_shoulder_lift for c in checks]),
        "original_end_gripper": _stats([c.original_end_gripper for c in checks]),
    }
    return checks, repo_summary


def _write_csv(rows: list[EpisodeCheck], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) + ["ok"] if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) | {"ok": row.ok})


def _write_index(summary: dict, out_dir: Path) -> None:
    lines = [
        "# Eval 3 v6 Truncation Verification",
        "",
        f"- Overall: {'PASS' if summary['ok'] else 'FAIL'}",
        f"- Episodes checked: {summary['episodes']}",
        f"- Repos checked: {summary['repos']}",
        f"- Total kept frames: {summary['total_kept_frames']}",
        f"- Kept frame range: {summary['overall_stats']['kept_frames']['min']:.0f}"
        f"-{summary['overall_stats']['kept_frames']['max']:.0f}",
        f"- Truncated end shoulder_lift range: "
        f"{summary['overall_stats']['end_shoulder_lift']['min']:+.1f}"
        f" to {summary['overall_stats']['end_shoulder_lift']['max']:+.1f}",
        f"- Truncated end gripper range: "
        f"{summary['overall_stats']['end_gripper']['min']:+.1f}"
        f" to {summary['overall_stats']['end_gripper']['max']:+.1f}",
        f"- Original end shoulder_lift mean: "
        f"{summary['overall_stats']['original_end_shoulder_lift']['mean']:+.1f}",
        f"- Original end gripper mean: "
        f"{summary['overall_stats']['original_end_gripper']['mean']:+.1f}",
        "",
        "## Files",
        "",
        "- `truncation_report.csv`: one row per episode.",
        "- `truncation_summary.json`: aggregate stats and failure details.",
        "- `contact_sheets/*.jpg`: visual start / match / trunc_end / original_end audit.",
        "",
        "## Contact Sheets",
        "",
    ]
    for repo_summary in summary["repo_summaries"]:
        slug = _repo_slug(repo_summary["repo"])
        lines.append(f"- [{slug}](contact_sheets/{slug}.jpg)")
    lines.append("")
    (out_dir / "index.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repos",
        default=",".join(DEFAULT_NEW_REPOS),
        help="Comma-separated dataset_v2 repo ids to verify.",
    )
    parser.add_argument("--out-dir", default="outputs/eval3_truncation_verify")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--grip-threshold", type=float, default=20.0)
    parser.add_argument("--lift-threshold", type=float, default=-30.0)
    parser.add_argument("--buffer-frames", type=int, default=60)
    parser.add_argument("--placement-mode", choices=["first", "last"], default="last")
    parser.add_argument("--allow-over-cap-episodes", default=DEFAULT_OVER_CAP_EPISODES)
    parser.add_argument("--over-cap-extra-frames", type=int, default=90)
    parser.add_argument("--max-frames-per-episode", type=int, default=600)
    parser.add_argument("--min-kept-frames", type=int, default=400)
    parser.add_argument("--max-kept-frames", type=int, default=690)
    parser.add_argument("--end-lift-min", type=float, default=-30.0)
    parser.add_argument("--end-gripper-min", type=float, default=10.0)
    parser.add_argument(
        "--no-contact-sheets",
        action="store_true",
        help="Only write CSV/JSON; skip video frame decoding.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.render_contact_sheets = not args.no_contact_sheets
    repos = [repo.strip() for repo in args.repos.split(",") if repo.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_checks: list[EpisodeCheck] = []
    repo_summaries = []
    for repo in repos:
        print(f"checking {repo}...")
        checks, repo_summary = _check_repo(args, repo, out_dir)
        all_checks.extend(checks)
        repo_summaries.append(repo_summary)
        status = "OK" if repo_summary["ok"] else "FAIL"
        print(
            f"  {status}: {repo_summary['episodes']} eps, kept "
            f"{repo_summary['kept_frames']['min']:.0f}-"
            f"{repo_summary['kept_frames']['max']:.0f}, end lift "
            f"{repo_summary['end_shoulder_lift']['min']:+.1f}-"
            f"{repo_summary['end_shoulder_lift']['max']:+.1f}, end grip "
            f"{repo_summary['end_gripper']['min']:+.1f}-"
            f"{repo_summary['end_gripper']['max']:+.1f}"
        )

    failures = [asdict(c) | {"ok": c.ok} for c in all_checks if not c.ok]
    total_kept_frames = sum(c.kept_frames for c in all_checks)
    summary = {
        "ok": not failures,
        "repos": len(repos),
        "episodes": len(all_checks),
        "total_kept_frames": total_kept_frames,
        "rule": {
            "grip_threshold": args.grip_threshold,
            "lift_threshold": args.lift_threshold,
            "placement_mode": args.placement_mode,
            "buffer_frames": args.buffer_frames,
            "min_kept_frames": args.min_kept_frames,
            "max_kept_frames": args.max_kept_frames,
            "end_lift_min": args.end_lift_min,
            "end_gripper_min": args.end_gripper_min,
        },
        "overall_stats": {
            "match_frame": _stats([c.match_frame for c in all_checks if c.match_frame is not None]),
            "kept_frames": _stats([c.kept_frames for c in all_checks]),
            "kept_fraction": _stats([c.kept_fraction for c in all_checks]),
            "end_shoulder_lift": _stats([c.end_shoulder_lift for c in all_checks]),
            "end_gripper": _stats([c.end_gripper for c in all_checks]),
            "original_end_shoulder_lift": _stats([c.original_end_shoulder_lift for c in all_checks]),
            "original_end_gripper": _stats([c.original_end_gripper for c in all_checks]),
        },
        "failures": failures,
        "repo_summaries": repo_summaries,
    }

    _write_csv(all_checks, out_dir / "truncation_report.csv")
    (out_dir / "truncation_summary.json").write_text(json.dumps(summary, indent=2))
    _write_index(summary, out_dir)

    print("")
    print(f"wrote {out_dir / 'truncation_report.csv'}")
    print(f"wrote {out_dir / 'truncation_summary.json'}")
    if args.render_contact_sheets:
        print(f"wrote contact sheets under {out_dir / 'contact_sheets'}")
    print(f"wrote {out_dir / 'index.md'}")
    print(f"overall: {'PASS' if summary['ok'] else 'FAIL'}")
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

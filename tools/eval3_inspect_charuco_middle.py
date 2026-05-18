#!/usr/bin/env python3
"""Visual inspection of dataset_v3_charuco_middle_1: which episodes are actually
misplaced and where does the can end up?

The audit flagged 5/10 episodes as classified `right` instead of `middle`. This
tool produces:
  * a shoulder_pan time-series PNG with one trace per episode, coloured by the
    audit's classification (middle = blue, right = red), with the placement-pose
    moment marked;
  * one camera-frame PNG per episode taken at that placement moment — so you
    can literally see which physical board the can is sitting on.

Outputs are under ``outputs/eval3_audit_dataset_labels/charuco_middle_inspection/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import cv2


DATASET = "datasets/dataset_v3_charuco_middle_1"
AUDIT_JSON = "outputs/eval3_audit_dataset_labels/audit_report.json"
OUT_DIR = Path("outputs/eval3_audit_dataset_labels/charuco_middle_inspection")
IDX_SHOULDER_PAN = 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load audit results for this dataset --------------------------
    audit = json.loads(Path(AUDIT_JSON).read_text())
    me = next(r for r in audit["reports"]
              if r["name"] == "dataset_v3_charuco_middle_1")
    m1 = next(m for m in me["metrics"] if m["metric_id"] == "M1_placement_position")
    per_ep = {pe["ep"]: pe for pe in m1["per_episode"]}
    centroids = {
        "left":   audit["calibration"]["centroid_left"],
        "middle": audit["calibration"]["centroid_middle"],
        "right":  audit["calibration"]["centroid_right"],
    }

    # --- Load action time series + episode boundaries -----------------
    root = Path(DATASET)
    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    actions = np.concatenate([
        np.array([np.asarray(a, dtype=np.float64)
                  for a in pq.read_table(f, columns=["action"]).to_pandas()["action"]])
        for f in data_files
    ])
    ep_files = sorted(root.glob("meta/episodes/chunk-*/file-*.parquet"))
    ep_df = (pq.read_table(ep_files[0], columns=[
        "episode_index", "length", "dataset_from_index", "dataset_to_index",
        "videos/observation.images.front/chunk_index",
        "videos/observation.images.front/file_index",
        "videos/observation.images.front/from_timestamp",
        "videos/observation.images.front/to_timestamp",
    ]).to_pandas()
        if len(ep_files) == 1
        else __import__("pandas").concat(
            [pq.read_table(f).to_pandas() for f in ep_files],
            ignore_index=True))

    n_ep = len(ep_df)
    print(f"Loaded {n_ep} episodes, {len(actions)} total frames")

    # --- Per-episode metadata + placement-frame extraction ------------
    fps = 30.0
    table = []
    for _, row in ep_df.iterrows():
        ep = int(row["episode_index"])
        f0 = int(row["dataset_from_index"])
        f1 = int(row["dataset_to_index"])
        pan = actions[f0:f1, IDX_SHOULDER_PAN]
        # Signed peak (extremum with largest |value|).
        pan_min, pan_max = float(pan.min()), float(pan.max())
        signed_peak = pan_max if abs(pan_max) >= abs(pan_min) else pan_min
        # Frame WITHIN the episode where the signed peak happens.
        peak_local_idx = int(np.argmax(np.abs(pan - 0)))  # actually want argmax(|pan|)
        peak_local_idx = int(np.argmax(np.abs(pan)))
        # Convert to a timestamp inside the chunked MP4.
        vchunk = int(row["videos/observation.images.front/chunk_index"])
        vfile = int(row["videos/observation.images.front/file_index"])
        from_ts = float(row["videos/observation.images.front/from_timestamp"])
        peak_ts_in_file = from_ts + peak_local_idx / fps
        audit_pe = per_ep.get(ep, {})
        table.append({
            "ep": ep,
            "length": int(row["length"]),
            "signed_peak": signed_peak,
            "peak_local_idx": peak_local_idx,
            "video_chunk": vchunk,
            "video_file": vfile,
            "peak_ts_in_file": peak_ts_in_file,
            "classified_as": audit_pe.get("classified_as", "?"),
            "audit_pass": audit_pe.get("pass", None),
        })

    # Order: passes first, then fails
    table.sort(key=lambda r: (not bool(r["audit_pass"]), r["ep"]))
    print(f"\n{'ep':>3s}  {'len':>4s}  {'signed_peak':>11s}  {'classif':>8s}  "
          f"{'pass':>5s}  {'placement_frame':>14s}")
    for r in table:
        print(f" {r['ep']:>3d}  {r['length']:>4d}  {r['signed_peak']:>+10.2f}°  "
              f"{r['classified_as']:>8s}  {str(r['audit_pass']):>5s}  "
              f"   ep_frame={r['peak_local_idx']}")

    # --- 1) Shoulder-pan time-series PNG ------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    for r in table:
        f0 = int(ep_df[ep_df["episode_index"] == r["ep"]]["dataset_from_index"].iloc[0])
        f1 = int(ep_df[ep_df["episode_index"] == r["ep"]]["dataset_to_index"].iloc[0])
        pan = actions[f0:f1, IDX_SHOULDER_PAN]
        t = np.arange(len(pan)) / fps
        color = {"middle": "tab:blue", "right": "tab:red",
                 "left": "tab:green"}.get(r["classified_as"], "gray")
        ax.plot(t, pan, color=color, alpha=0.7,
                label=f"ep{r['ep']} ({r['classified_as']})")
        # Mark the signed-peak moment.
        ax.scatter([r["peak_local_idx"] / fps], [r["signed_peak"]],
                   color=color, marker="o", s=60, edgecolors="black", zorder=5)
    # Centroid + boundary reference lines.
    for label, val in centroids.items():
        ax.axhline(val, color={"left": "tab:green", "middle": "tab:blue",
                               "right": "tab:red"}[label],
                   linestyle="--", alpha=0.4)
        ax.text(ax.get_xlim()[1] * 0.99, val, f" centroid {label.upper()} ({val:+.1f}°)",
                color={"left": "tab:green", "middle": "tab:blue",
                       "right": "tab:red"}[label],
                fontsize=9, ha="right", va="bottom")
    ax.axhline(audit["calibration"]["boundary_lm"], color="black",
               linestyle=":", alpha=0.3)
    ax.axhline(audit["calibration"]["boundary_mr"], color="black",
               linestyle=":", alpha=0.3)
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("shoulder_pan (deg)")
    ax.set_title(
        "dataset_v3_charuco_middle_1 — shoulder_pan over time, per episode\n"
        "blue=classified middle ✓, red=classified right ✗ (signed-peak marker = placement moment)"
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_ts = OUT_DIR / "shoulder_pan_timeseries.png"
    fig.savefig(out_ts, dpi=130)
    plt.close(fig)
    print(f"\nwrote {out_ts}")

    # --- 2) Placement-frame thumbnails grid ---------------------------
    # For each episode, extract the camera frame at the signed-peak moment
    # and tile them in a 2-row x 5-col grid (10 episodes).
    thumbs = []
    for r in table:
        src_mp4 = (root / "videos" / "observation.images.front" /
                   f"chunk-{r['video_chunk']:03d}" /
                   f"file-{r['video_file']:03d}.mp4")
        cap = cv2.VideoCapture(str(src_mp4))
        cap.set(cv2.CAP_PROP_POS_MSEC, r["peak_ts_in_file"] * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  WARN: could not extract frame for ep{r['ep']} "
                  f"at {r['peak_ts_in_file']:.2f}s in {src_mp4.name}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # HUD overlay
        emoji = "OK" if r["audit_pass"] else "MISLABEL"
        hud_color = (40, 200, 40) if r["audit_pass"] else (40, 40, 220)
        cv2.rectangle(frame, (0, 0), (640, 70), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"ep{r['ep']:02d}  signed_peak={r['signed_peak']:+.1f} deg",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"classified as: {r['classified_as'].upper()}  -> {emoji}",
                    (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
        thumbs.append(frame)
        # Also save per-episode frame.
        out_ep = OUT_DIR / f"ep{r['ep']:02d}_placement_{r['classified_as']}.png"
        cv2.imwrite(str(out_ep), frame)
        print(f"  wrote {out_ep}")

    # Grid: 2 rows x 5 cols
    rows = []
    for i in range(0, len(thumbs), 5):
        rows.append(np.concatenate(thumbs[i:i + 5], axis=1))
    grid = np.concatenate(rows, axis=0)
    out_grid = OUT_DIR / "placement_frames_grid.png"
    cv2.imwrite(str(out_grid), grid)
    print(f"\nwrote {out_grid}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()

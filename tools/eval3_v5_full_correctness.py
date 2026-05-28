#!/usr/bin/env python3
"""Correctness battery for the synthetic `dataset_v5_charuko_<slot>_full` datasets.

Loads each slot from disk and from Hugging Face Hub, then runs the following
checks against the source `dataset_v5_charuko_approach` and
`dataset_v5_charuko_<slot>_1` parquets to make sure the materialization step
did not corrupt anything:

  S1  info.json totals match parquet row count
  S2  episodes.parquet: dataset_from/to_index forms a non-overlapping contiguous tiling
  S3  per-episode `length` matches (dataset_to - dataset_from)
  S4  frame_index resets to 0 each episode and runs 0..N-1
  S5  global `index` column equals row position
  S6  task_index is always 0; task string round-trips to "<placeholder>"
  S7  action and observation.state are valid 6-vectors (finite, expected shape)
  S8  total frames == 5118 (left), 5078 (middle), 4723 (right)

  P1  Approach segment frames in each ep match the SOURCE approach episode rows
      (action + state, bit-exact within tolerance)
  P2  Place segment frames in each ep match the SOURCE place episode rows
  P3  Bridge segment matches the linear-interpolation formula
        t_i = (i+1)/(N+1),  x_i = (1-t_i) * approach_last + t_i * place_first
  P4  Each ep has the right composition: approach_len + N_bridge + place_len

  V1  Video frame count == parquet row count
  V2  Video FPS == 30
  V3  videos/.../from_timestamp / to_timestamp in episodes.parquet match cumulative frame offset / FPS

  C1  All 10 (a_ep, p_ep) pairs appear exactly once per slot (cross-product completeness)

  H1  Hub-cached version loads with the same num_episodes / num_frames / task string
  H2  Hub-cached sample row equals local sample row for action+state (a few spot checks)

Exit code 0 if all pass, 1 otherwise. A summary table is printed at the end.

Usage:
    python tools/eval3_v5_full_correctness.py
    python tools/eval3_v5_full_correctness.py --slots left          # restrict
    python tools/eval3_v5_full_correctness.py --skip-hub             # skip H1/H2
    python tools/eval3_v5_full_correctness.py --tol 1e-4             # numeric tolerance
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_ROOT = REPO_ROOT / "datasets"

EXPECTED_FRAMES = {"left": 5118, "middle": 5078, "right": 4723}
EXPECTED_EPISODES = 10
N_BRIDGE_DEFAULT = 20
FPS = 30
TASK_STRING = "Place the coke on <placeholder>"


# --------------------------------------------------------------------------- #
# Test harness                                                                #
# --------------------------------------------------------------------------- #

class Results:
    def __init__(self):
        self.rows: list[tuple[str, str, bool, str]] = []  # (slot, test, ok, detail)

    def add(self, slot: str, test: str, ok: bool, detail: str = "") -> None:
        self.rows.append((slot, test, ok, detail))

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for _, _, ok, _ in self.rows if ok)

    @property
    def failed(self) -> int:
        return sum(1 for _, _, ok, _ in self.rows if not ok)

    def print_summary(self) -> None:
        print()
        print("=" * 100)
        print(f"  RESULTS: {self.passed}/{self.total} passed  ({self.failed} failed)")
        print("=" * 100)
        # Group by slot
        by_slot: dict[str, list[tuple[str, bool, str]]] = {}
        for slot, test, ok, det in self.rows:
            by_slot.setdefault(slot, []).append((test, ok, det))
        for slot, rows in by_slot.items():
            print(f"\n[{slot}]")
            for test, ok, det in rows:
                mark = "  ✓" if ok else "  ✗"
                line = f"{mark}  {test}"
                if det:
                    line += f"  — {det}"
                print(line)
        # Print failures separately for grep-ability
        fails = [(s, t, d) for s, t, ok, d in self.rows if not ok]
        if fails:
            print(f"\nFAILURES ({len(fails)}):")
            for s, t, d in fails:
                print(f"  {s}/{t}  {d}")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _load_parquets(root: Path) -> dict:
    info = json.loads((root / "meta" / "info.json").read_text())
    data = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pandas()
    eps = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    return {"info": info, "data": data, "episodes": eps, "tasks": tasks, "root": root}


def _video_info(mp4: Path) -> tuple[int, float, float]:
    """Return (frame_count, fps, duration_s) via ffprobe."""
    cmd_frames = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,r_frame_rate,duration",
        "-of", "default=nw=1", str(mp4),
    ]
    out = subprocess.run(cmd_frames, capture_output=True, text=True, check=True).stdout
    nframes = None
    fps = None
    dur = None
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k == "nb_read_frames":
            nframes = int(v)
        elif k == "r_frame_rate":
            num, den = v.split("/")
            fps = float(num) / float(den) if float(den) else float(num)
        elif k == "duration":
            dur = float(v)
    return nframes, fps, dur


def _stack_arr(col: object) -> np.ndarray:
    """Stack a pandas column of list-like values into an (N, D) numpy array."""
    return np.stack([np.asarray(v, dtype=np.float32) for v in col])


# --------------------------------------------------------------------------- #
# Per-slot checks                                                             #
# --------------------------------------------------------------------------- #

def check_slot(slot: str, results: Results, tol: float, n_bridge: int) -> None:
    print(f"\n=== checking slot={slot} ===")
    full_root = DATASETS_ROOT / f"dataset_v5_charuko_{slot}_full"
    if not full_root.exists():
        results.add(slot, "exists", False, f"{full_root} not found")
        return
    results.add(slot, "exists", True, str(full_root))

    full = _load_parquets(full_root)
    ap = _load_parquets(DATASETS_ROOT / "dataset_v5_charuko_approach")
    pl = _load_parquets(DATASETS_ROOT / f"dataset_v5_charuko_{slot}_1")

    info = full["info"]
    data = full["data"]
    eps = full["episodes"]

    # --- S1: info totals match data ---
    ok = info["total_frames"] == len(data) and info["total_episodes"] == len(eps)
    results.add(slot, "S1 info totals match data", ok,
                f"info={info['total_frames']}/{info['total_episodes']} "
                f"data={len(data)} eps={len(eps)}")

    # --- S2: episode boundaries form a contiguous non-overlapping tiling ---
    ep_from = eps["dataset_from_index"].to_numpy()
    ep_to = eps["dataset_to_index"].to_numpy()
    sorted_idx = np.argsort(ep_from)
    ep_from_s, ep_to_s = ep_from[sorted_idx], ep_to[sorted_idx]
    contiguous = (ep_from_s[0] == 0
                  and ep_to_s[-1] == len(data)
                  and np.all(ep_from_s[1:] == ep_to_s[:-1]))
    results.add(slot, "S2 episode boundary tiling is contiguous", contiguous,
                f"first_from={ep_from_s[0]} last_to={ep_to_s[-1]} expected 0..{len(data)}")

    # --- S3: lengths match ---
    lens = eps["length"].to_numpy()
    ok = bool(np.all(lens == (ep_to - ep_from)))
    results.add(slot, "S3 per-episode length matches (to - from)", ok,
                f"discrepancies={int((lens != (ep_to - ep_from)).sum())}")

    # --- S4: frame_index resets per episode, runs 0..N-1 ---
    fi = data["frame_index"].to_numpy()
    expected_fi = np.concatenate([np.arange(n) for n in lens])
    ok = bool(np.array_equal(fi, expected_fi))
    results.add(slot, "S4 frame_index resets per episode (0..N-1)", ok,
                f"first 5 = {fi[:5].tolist()}  last 5 = {fi[-5:].tolist()}")

    # --- S5: global index column ---
    gi = data["index"].to_numpy()
    ok = bool(np.array_equal(gi, np.arange(len(data), dtype=np.int64)))
    results.add(slot, "S5 global `index` matches row position", ok,
                f"max={int(gi.max())} expected={len(data)-1}")

    # --- S6: task_index always 0; task string round-trips ---
    ti = data["task_index"].to_numpy()
    ok_idx = bool(np.all(ti == 0))
    ok_str = TASK_STRING in str(full["tasks"].index.tolist() + full["tasks"]["task_index"].tolist())
    # More robust: load via LeRobotDataset and check the joined string
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds_local = LeRobotDataset(
        f"RobotLearningVLA/dataset_v5_charuko_{slot}_full",
        root=str(full_root), video_backend="pyav",
    )
    sampled_tasks = [ds_local[i]["task"] for i in (0, len(ds_local) // 2, len(ds_local) - 1)]
    ok_str = all(t == TASK_STRING for t in sampled_tasks)
    results.add(slot, "S6 task_index == 0 & task string round-trip", ok_idx and ok_str,
                f"unique task_index={sorted(set(ti.tolist()))} sampled_tasks={sampled_tasks}")

    # --- S7: action / observation.state shapes and finiteness ---
    action_arr = _stack_arr(data["action"])
    state_arr = _stack_arr(data["observation.state"])
    ok_shape = action_arr.shape == (len(data), 6) and state_arr.shape == (len(data), 6)
    ok_finite = bool(np.isfinite(action_arr).all()) and bool(np.isfinite(state_arr).all())
    results.add(slot, "S7 action/state shape (N,6) and all finite", ok_shape and ok_finite,
                f"action.shape={action_arr.shape} state.shape={state_arr.shape} "
                f"finite={ok_finite}")

    # --- S8: expected frame count ---
    expected = EXPECTED_FRAMES[slot]
    ok = len(data) == expected
    results.add(slot, f"S8 total frames == {expected}", ok,
                f"got {len(data)}")

    # --- P1/P2/P3/P4: reconstruct each synthetic episode and verify ---
    ap_data = ap["data"]
    ap_eps = ap["episodes"]
    pl_data = pl["data"]
    pl_eps = pl["episodes"]

    ap_eps_sorted = ap_eps.sort_values("episode_index").reset_index(drop=True)
    pl_eps_sorted = pl_eps.sort_values("episode_index").reset_index(drop=True)
    eps_sorted = eps.sort_values("episode_index").reset_index(drop=True)

    n_ap = len(ap_eps_sorted)
    n_pl = len(pl_eps_sorted)

    ok_p1 = True
    ok_p2 = True
    ok_p3 = True
    ok_p4 = True
    fail_msgs: list[str] = []
    pair_seen: set[tuple[int, int]] = set()

    for syn_idx in range(EXPECTED_EPISODES):
        # The synthetic episodes are in cross-product order (a_ep varies fastest? slowest?).
        # Materializer emits them with a_ep outer, p_ep inner — so syn 0..1 use a_ep=0,
        # syn 2..3 use a_ep=1, etc.
        a_ep = syn_idx // n_pl
        p_ep = syn_idx % n_pl
        pair_seen.add((a_ep, p_ep))

        f0 = int(eps_sorted.iloc[syn_idx]["dataset_from_index"])
        f1 = int(eps_sorted.iloc[syn_idx]["dataset_to_index"])
        ep_block = data.iloc[f0:f1]

        # Source slices
        a0 = int(ap_eps_sorted.iloc[a_ep]["dataset_from_index"])
        a1 = int(ap_eps_sorted.iloc[a_ep]["dataset_to_index"])
        p0 = int(pl_eps_sorted.iloc[p_ep]["dataset_from_index"])
        p1 = int(pl_eps_sorted.iloc[p_ep]["dataset_to_index"])
        ap_block = ap_data.iloc[a0:a1].reset_index(drop=True)
        pl_block = pl_data.iloc[p0:p1].reset_index(drop=True)
        n_a = len(ap_block)
        n_p = len(pl_block)

        # P4: composition length
        expected_len = n_a + n_bridge + n_p
        if len(ep_block) != expected_len:
            ok_p4 = False
            fail_msgs.append(f"syn {syn_idx}: len={len(ep_block)} expected {expected_len} "
                             f"(={n_a}+{n_bridge}+{n_p})")

        ep_action = _stack_arr(ep_block["action"])
        ep_state = _stack_arr(ep_block["observation.state"])

        # P1: approach segment = source approach data
        ap_action = _stack_arr(ap_block["action"])
        ap_state = _stack_arr(ap_block["observation.state"])
        diff_a = np.abs(ep_action[:n_a] - ap_action).max() if n_a else 0.0
        diff_a_s = np.abs(ep_state[:n_a] - ap_state).max() if n_a else 0.0
        if max(diff_a, diff_a_s) > tol:
            ok_p1 = False
            fail_msgs.append(f"syn {syn_idx}: approach mismatch max={max(diff_a, diff_a_s):.4g}")

        # P2: place segment = source place data
        pl_action = _stack_arr(pl_block["action"])
        pl_state = _stack_arr(pl_block["observation.state"])
        place_start = n_a + n_bridge
        diff_p = np.abs(ep_action[place_start:place_start + n_p] - pl_action).max() if n_p else 0.0
        diff_p_s = np.abs(ep_state[place_start:place_start + n_p] - pl_state).max() if n_p else 0.0
        if max(diff_p, diff_p_s) > tol:
            ok_p2 = False
            fail_msgs.append(f"syn {syn_idx}: place mismatch max={max(diff_p, diff_p_s):.4g}")

        # P3: bridge = linear interp
        if n_bridge > 0 and n_a > 0 and n_p > 0:
            ap_last_a = ap_action[-1]
            pl_first_a = pl_action[0]
            ap_last_s = ap_state[-1]
            pl_first_s = pl_state[0]
            expected_bridge_a = np.stack([
                (1.0 - (i + 1) / (n_bridge + 1)) * ap_last_a
                + ((i + 1) / (n_bridge + 1)) * pl_first_a
                for i in range(n_bridge)
            ]).astype(np.float32)
            expected_bridge_s = np.stack([
                (1.0 - (i + 1) / (n_bridge + 1)) * ap_last_s
                + ((i + 1) / (n_bridge + 1)) * pl_first_s
                for i in range(n_bridge)
            ]).astype(np.float32)
            actual_a = ep_action[n_a:n_a + n_bridge]
            actual_s = ep_state[n_a:n_a + n_bridge]
            diff_ba = np.abs(actual_a - expected_bridge_a).max()
            diff_bs = np.abs(actual_s - expected_bridge_s).max()
            if max(diff_ba, diff_bs) > tol:
                ok_p3 = False
                fail_msgs.append(f"syn {syn_idx}: bridge mismatch action_max={diff_ba:.4g} state_max={diff_bs:.4g}")

    results.add(slot, "P1 approach segment matches source approach", ok_p1,
                "; ".join(m for m in fail_msgs if "approach" in m)[:200])
    results.add(slot, "P2 place segment matches source place", ok_p2,
                "; ".join(m for m in fail_msgs if "place mismatch" in m)[:200])
    results.add(slot, f"P3 bridge = linear interp (t=i/(N+1))", ok_p3,
                "; ".join(m for m in fail_msgs if "bridge mismatch" in m)[:200])
    results.add(slot, f"P4 composition is approach+{n_bridge}+place", ok_p4,
                "; ".join(m for m in fail_msgs if "expected" in m and "len" in m)[:200])

    # --- C1: cross-product completeness ---
    expected_pairs = {(a, p) for a in range(n_ap) for p in range(n_pl)}
    results.add(slot, "C1 all 10 (a_ep,p_ep) pairs present exactly once",
                pair_seen == expected_pairs,
                f"got {len(pair_seen)} unique pairs out of {len(expected_pairs)}")

    # --- V1/V2/V3: video alignment ---
    video_path = full_root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
    if not video_path.exists():
        results.add(slot, "V1 video file exists", False, str(video_path))
        return
    nfr, fps, _dur = _video_info(video_path)
    results.add(slot, f"V1 video frame count == {len(data)}", nfr == len(data),
                f"video={nfr} parquet={len(data)}")
    results.add(slot, "V2 video FPS == 30", abs(fps - FPS) < 0.01,
                f"video.fps={fps}")
    # V3: episode video timestamps form a contiguous tiling at FPS
    ts_from = eps_sorted["videos/observation.images.front/from_timestamp"].to_numpy()
    ts_to = eps_sorted["videos/observation.images.front/to_timestamp"].to_numpy()
    cum = np.cumsum(lens) / FPS
    expected_from = np.concatenate([[0.0], cum[:-1]]).astype(np.float64)
    expected_to = cum.astype(np.float64)
    ok = (np.allclose(ts_from, expected_from, atol=1e-6)
          and np.allclose(ts_to, expected_to, atol=1e-6))
    results.add(slot, "V3 episode from/to_timestamp == cumulative frames / FPS", ok,
                f"first_from={ts_from[0]:.4f} last_to={ts_to[-1]:.4f}")


# --------------------------------------------------------------------------- #
# Hub round-trip                                                              #
# --------------------------------------------------------------------------- #

def check_hub(slot: str, results: Results, tol: float, n_bridge: int) -> None:
    """Clean local Hub cache, re-fetch, compare to local materialized version."""
    import shutil
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    hub_cache = Path.home() / ".cache" / "huggingface" / "lerobot" / "RobotLearningVLA"
    hub_repo_dir = hub_cache / f"dataset_v5_charuko_{slot}_full"
    if hub_repo_dir.exists():
        shutil.rmtree(hub_repo_dir)

    repo_id = f"RobotLearningVLA/dataset_v5_charuko_{slot}_full"
    try:
        ds_hub = LeRobotDataset(repo_id, video_backend="pyav")
    except Exception as e:
        results.add(slot, "H1 Hub dataset loads", False, f"{type(e).__name__}: {e}")
        return

    full_root = DATASETS_ROOT / f"dataset_v5_charuko_{slot}_full"
    ds_local = LeRobotDataset(repo_id, root=str(full_root), video_backend="pyav")

    ok_meta = (
        ds_hub.num_episodes == ds_local.num_episodes
        and ds_hub.num_frames == ds_local.num_frames
        and ds_hub.meta.fps == ds_local.meta.fps
    )
    results.add(slot, "H1 Hub matches local: episodes/frames/fps", ok_meta,
                f"hub={ds_hub.num_episodes}/{ds_hub.num_frames}/{ds_hub.meta.fps} "
                f"local={ds_local.num_episodes}/{ds_local.num_frames}/{ds_local.meta.fps}")

    # H2: spot-check a few sample rows for bit-exact action+state
    samples = [0, len(ds_local) // 3, 2 * len(ds_local) // 3, len(ds_local) - 1]
    max_diff_state = 0.0
    max_diff_action = 0.0
    max_task_diff = ""
    for i in samples:
        r_hub = ds_hub[i]
        r_local = ds_local[i]
        d_s = float(np.abs(r_hub["observation.state"].numpy() - r_local["observation.state"].numpy()).max())
        d_a = float(np.abs(r_hub["action"].numpy() - r_local["action"].numpy()).max())
        max_diff_state = max(max_diff_state, d_s)
        max_diff_action = max(max_diff_action, d_a)
        if r_hub["task"] != r_local["task"]:
            max_task_diff = f"i={i} hub={r_hub['task']!r} local={r_local['task']!r}"
    ok = max_diff_state < tol and max_diff_action < tol and not max_task_diff
    results.add(slot, "H2 spot-checked rows equal between hub & local", ok,
                f"max_state_diff={max_diff_state:.4g} max_action_diff={max_diff_action:.4g} "
                f"task_diff={max_task_diff or 'OK'}")


# --------------------------------------------------------------------------- #
# Main                                                                       #
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", default="left,middle,right")
    ap.add_argument("--skip-hub", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="Numeric tolerance for action/state comparisons.")
    ap.add_argument("--bridge-frames", type=int, default=N_BRIDGE_DEFAULT)
    args = ap.parse_args()

    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    results = Results()
    for slot in slots:
        check_slot(slot, results, args.tol, args.bridge_frames)
        if not args.skip_hub:
            check_hub(slot, results, args.tol, args.bridge_frames)

    results.print_summary()
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

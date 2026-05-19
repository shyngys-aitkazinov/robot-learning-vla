#!/usr/bin/env python3
"""Diagnose celebrity-differentiation invariants in Eval3 SmolVLA training data.

Two modes, selected via ``--mode``:

* ``v3_slots`` (default, historical) — proves the v9 collapse bug.
  Pulls one ~6 MB parquet per ``dataset_v3_synth_<celeb>_<position>_2`` repo,
  asserts that for a fixed slot the action arrays are byte-identical across
  the three celebrity datasets. When that's true, BC can solve the training
  task by ignoring the language token entirely (action = f(slot)). See
  ``docs/eval3/charuco_pipeline.md:281`` for the root cause walk-through.

* ``v4_balanced`` — Fix A preflight, cheap (no GPU, ~30 s for 3 repos).
  Pulls ``meta/episodes/chunk-000/file-000.parquet`` + the data parquet
  for each ``dataset_v4_synth_<celeb>_balanced_1`` repo and asserts that
  each per-celeb dataset spans ALL THREE source slots (left / middle /
  right). The invariant: within a single (task = "Place the coke on <Celeb>")
  dataset, episode-0-frame-0 actions must cluster into 3 distinct slot
  signatures. If a repo collapses into 1 or 2 clusters, the generator
  regressed and the dataset is back to v3-style "action = f(slot)"
  shortcuts — train would waste GPU time.

Both modes only download parquet (no video), are safe to run on the laptop,
and write a JSON artifact under ``outputs/eval3_diag/`` for later inspection.

Usage::

    # v3 slot duplication check (historical, what the v9 bug needed):
    .venv/bin/python tools/eval3_diagnose_celeb_confusion.py \\
        --mode v3_slots --slots left middle right \\
        --out outputs/eval3_diag/celeb_confusion.json

    # v4 balanced preflight (run AFTER ``run_eval3_synth_dataset_gen.sh``
    # with EVAL3_SYNTH_BALANCED=1, BEFORE launching the v10 trainer):
    .venv/bin/python tools/eval3_diagnose_celeb_confusion.py \\
        --mode v4_balanced \\
        --out outputs/eval3_diag/v4_balanced_preflight.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ORG = "RobotLearningVLA"
CELEBS = ("taylor_swift", "yann_lecun", "barack_obama")
SLOTS = ("left", "middle", "right")
DATA_PATH = "data/chunk-000/file-000.parquet"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_PATH = "meta/episodes/chunk-000/file-000.parquet"

# v3-charuco source first-frame shoulder_lift signatures (from
# outputs/eval3_diag/celeb_confusion.json frame_0 samples — captured BEFORE
# any synth-side noise / smoothing). Each slot has a distinct value, and
# all three v3_synth_*_<slot>_2 repos share the same source per slot,
# so these doubles as the v4-balanced cluster centres.
V4_SLOT_FIRST_SHOULDER_LIFT = {
    "left":   -103.2527,
    "middle": -103.3407,
    "right":  -103.1648,
}
# Tolerance for assigning an observed first-frame shoulder_lift to a slot.
# Source values are ~0.09 deg apart at the closest pair (left vs middle),
# so anything tighter than that won't disambiguate. 0.05 deg is comfortably
# inside the source-side noise floor.
V4_SLOT_TOLERANCE_DEG = 0.05


def _download(repo_id: str, rel_path: str) -> Path:
    return Path(hf_hub_download(repo_id, rel_path, repo_type="dataset"))


def _load_actions_states(repo_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = _download(repo_id, DATA_PATH)
    table = pq.read_table(str(path), columns=["action", "observation.state"])
    df = table.to_pandas()
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    return actions, states


def _load_task_strings(repo_id: str) -> list[str]:
    path = _download(repo_id, TASKS_PATH)
    table = pq.read_table(str(path))
    df = table.to_pandas().reset_index()
    if "task" in df.columns:
        return [str(t) for t in df["task"].tolist()]
    if df.index.name == "task":
        return [str(t) for t in df.index.tolist()]
    return [str(t) for t in df.iloc[:, 0].tolist()]


def _compare_triplet(arrays: dict[str, np.ndarray]) -> dict:
    """Pairwise element-wise comparison across the 3 celeb arrays for one slot."""
    keys = list(arrays.keys())
    assert len(keys) == 3, keys
    n = min(a.shape[0] for a in arrays.values())
    truncated = {k: a[:n] for k, a in arrays.items()}
    out: dict = {"frames_compared": int(n), "shape": tuple(truncated[keys[0]].shape)}
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = truncated[keys[i]], truncated[keys[j]]
            equal = bool(np.array_equal(a, b))
            allclose = bool(np.allclose(a, b, atol=1e-6, rtol=0))
            max_abs = float(np.max(np.abs(a - b)))
            mean_abs = float(np.mean(np.abs(a - b)))
            l2 = float(np.linalg.norm(a - b))
            out[f"{keys[i]}_vs_{keys[j]}"] = {
                "bitwise_equal": equal,
                "allclose_1e-6": allclose,
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "l2_norm_diff": l2,
            }
    return out


def diagnose_slot(slot: str) -> dict:
    print(f"\n=== slot={slot} ===", flush=True)
    repos = {c: f"{ORG}/dataset_v3_synth_{c}_{slot}_2" for c in CELEBS}
    actions: dict[str, np.ndarray] = {}
    states: dict[str, np.ndarray] = {}
    tasks: dict[str, list[str]] = {}
    for celeb, repo in repos.items():
        print(f"  loading {repo}", flush=True)
        a, s = _load_actions_states(repo)
        actions[celeb] = a
        states[celeb] = s
        tasks[celeb] = _load_task_strings(repo)
        print(f"    -> action {a.shape}  state {s.shape}  unique_tasks={sorted(set(tasks[celeb]))}", flush=True)

    action_cmp = _compare_triplet(actions)
    state_cmp = _compare_triplet(states)

    # Sample triplet of action vectors for human inspection.
    sample = {}
    indices_to_sample = [0, min(100, action_cmp["frames_compared"] - 1), min(200, action_cmp["frames_compared"] - 1)]
    for idx in indices_to_sample:
        sample[f"frame_{idx}"] = {c: actions[c][idx].round(4).tolist() for c in CELEBS}

    return {
        "slot": slot,
        "repos": repos,
        "action_comparison": action_cmp,
        "state_comparison": state_cmp,
        "task_strings_unique": {c: sorted(set(tasks[c])) for c in CELEBS},
        "action_sample_per_celeb": sample,
    }


# ---------------------------------------------------------------------------
# v4 balanced preflight (Fix A validation)
# ---------------------------------------------------------------------------

def _load_episode_metadata(repo_id: str) -> list[dict]:
    """Return per-episode metadata for one v3.0 LeRobotDataset repo.

    Each entry has ``episode_index``, ``length``, and the global
    (``dataset_from_index``, ``dataset_to_index``) into the data parquet.
    """
    path = _download(repo_id, EPISODES_PATH)
    cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    df = pq.read_table(str(path), columns=cols).to_pandas()
    df = df.sort_values("episode_index").reset_index(drop=True)
    out: list[dict] = []
    for _, r in df.iterrows():
        out.append({
            "episode_index": int(r["episode_index"]),
            "length": int(r["length"]),
            "from": int(r["dataset_from_index"]),
            "to": int(r["dataset_to_index"]),
        })
    return out


def _classify_slot(shoulder_lift_value: float) -> str | None:
    """Return 'left' | 'middle' | 'right' | None for an observed first-frame shoulder_lift."""
    best_slot, best_d = None, float("inf")
    for slot, ref in V4_SLOT_FIRST_SHOULDER_LIFT.items():
        d = abs(shoulder_lift_value - ref)
        if d < best_d:
            best_slot, best_d = slot, d
    if best_d <= V4_SLOT_TOLERANCE_DEG:
        return best_slot
    return None


def diagnose_v4_balanced_one_celeb(celeb: str) -> dict:
    """Validate that ``dataset_v4_synth_<celeb>_balanced_1`` spans all three slots."""
    repo = f"{ORG}/dataset_v4_synth_{celeb}_balanced_1"
    print(f"\n=== {repo} ===", flush=True)
    episodes = _load_episode_metadata(repo)
    print(f"  {len(episodes)} episodes", flush=True)
    actions, _states = _load_actions_states(repo)
    tasks = _load_task_strings(repo)
    print(f"  action shape {actions.shape}  unique_tasks={sorted(set(tasks))}", flush=True)

    # Classify each episode by its first-frame shoulder_lift (action axis 1).
    per_episode: list[dict] = []
    slot_counter: Counter = Counter()
    unclassified = 0
    for ep in episodes:
        first_idx = ep["from"]
        if first_idx >= actions.shape[0]:
            print(f"  WARN ep{ep['episode_index']} from_idx={first_idx} >= "
                  f"action_len={actions.shape[0]}; skipping", flush=True)
            continue
        sh_lift = float(actions[first_idx, 1])
        slot = _classify_slot(sh_lift)
        per_episode.append({
            "episode_index": ep["episode_index"],
            "first_frame_shoulder_lift": sh_lift,
            "slot": slot,
        })
        if slot is None:
            unclassified += 1
        else:
            slot_counter[slot] += 1

    total_classified = sum(slot_counter.values())
    fractions = {s: slot_counter.get(s, 0) / max(total_classified, 1) for s in SLOTS}
    # Balance criterion: each slot should be within +/- 20% of an even
    # 1/3 split. (v4 balanced gen alternates slot per config, so the
    # natural distribution is exactly equal; +/-20% allows for small
    # truncations.)
    target = 1.0 / 3.0
    max_dev = max(abs(fractions[s] - target) for s in SLOTS) if total_classified else 1.0
    balanced_enough = max_dev <= 0.20

    slots_present = sorted({s for s in slot_counter if slot_counter[s] > 0})
    return {
        "repo": repo,
        "celeb": celeb,
        "n_episodes": len(episodes),
        "n_classified": total_classified,
        "n_unclassified": unclassified,
        "task_strings_unique": sorted(set(tasks)),
        "slot_counts": dict(slot_counter),
        "slot_fractions": fractions,
        "slots_present": slots_present,
        "max_fraction_dev_from_third": max_dev,
        "balanced_enough_pm20pct": balanced_enough,
        "per_episode": per_episode,
    }


def _v4_verdict(per_celeb: dict[str, dict]) -> tuple[int, str]:
    """Aggregate v4-balanced verdict + exit code across the 3 celebs.

    Returns (exit_code, summary_line). Codes:
      0 = PASS  — every celeb covers all 3 slots and is +/-20% balanced
      1 = WEAK  — every celeb covers all 3 slots but distribution is skewed
      2 = FAIL  — at least one celeb is missing a slot
    """
    missing_repos = [r["repo"] for r in per_celeb.values() if len(r["slots_present"]) < 3]
    if missing_repos:
        return 2, ("FAIL — at least one repo is missing a slot: "
                   + ", ".join(missing_repos))
    skewed = [r["repo"] for r in per_celeb.values() if not r["balanced_enough_pm20pct"]]
    if skewed:
        return 1, ("WEAK — all 3 slots present in every repo, but distribution "
                   "is skewed (>20% off uniform) in: " + ", ".join(skewed))
    return 0, "PASS — all 3 v4 balanced repos span left/middle/right within +/-20% uniform."


# ---------------------------------------------------------------------------
# Mode dispatch + CLI
# ---------------------------------------------------------------------------

def _run_v3_slots(slots: list[str], out_path: Path) -> int:
    results = {}
    for slot in slots:
        results[slot] = diagnose_slot(slot)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}", flush=True)

    print("\n=== Verdict (v3_slots) ===")
    any_duplicated = False
    for slot, slot_res in results.items():
        acmp = slot_res["action_comparison"]
        pairs = [k for k in acmp if "_vs_" in k]
        all_equal = all(acmp[p]["bitwise_equal"] for p in pairs)
        max_l2 = max(acmp[p]["l2_norm_diff"] for p in pairs)
        marker = "DUPLICATED (bitwise equal)" if all_equal else f"distinct (max L2 {max_l2:.4f})"
        print(f"  slot={slot:<6}  actions across celebs: {marker}")
        if all_equal:
            any_duplicated = True
            unique_tasks = slot_res["task_strings_unique"]
            print(f"    only varying signal is the task string:")
            for celeb, ts in unique_tasks.items():
                print(f"      {celeb}: {ts}")

    if any_duplicated:
        print("\nCONFIRMED: at least one slot has byte-identical action labels across "
              "Taylor Swift / Yann LeCun / Barack Obama. The policy can solve the "
              "training task by ignoring language entirely (action = f(slot)).")
        return 0
    print("\nNot duplicated — action arrays differ across celebrities.")
    return 1


def _run_v4_balanced(celebs: list[str], out_path: Path) -> int:
    per_celeb: dict[str, dict] = {}
    for celeb in celebs:
        per_celeb[celeb] = diagnose_v4_balanced_one_celeb(celeb)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(per_celeb, indent=2))
    print(f"\nwrote {out_path}", flush=True)

    print("\n=== Verdict (v4_balanced preflight) ===")
    print(f"  {'celeb':<14}  {'n_ep':>5}  {'left':>6}  {'middle':>7}  {'right':>6}  "
          f"{'max_dev':>8}  status")
    print(f"  {'-' * 14}  {'-' * 5}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 8}  {'-' * 18}")
    for celeb, r in per_celeb.items():
        sc = r["slot_counts"]
        status_parts = []
        if len(r["slots_present"]) < 3:
            missing = sorted(set(SLOTS) - set(r["slots_present"]))
            status_parts.append("MISSING(" + "/".join(missing) + ")")
        elif not r["balanced_enough_pm20pct"]:
            status_parts.append("SKEWED")
        else:
            status_parts.append("balanced")
        if r["n_unclassified"]:
            status_parts.append(f"+{r['n_unclassified']} unclassified")
        status = "; ".join(status_parts)
        print(f"  {celeb:<14}  {r['n_episodes']:>5}  {sc.get('left', 0):>6}  "
              f"{sc.get('middle', 0):>7}  {sc.get('right', 0):>6}  "
              f"{r['max_fraction_dev_from_third']:>8.3f}  {status}")

    code, line = _v4_verdict(per_celeb)
    print(f"\n{line}")
    if code == 0:
        print("v4 balanced corpus is ready to train. Launch with "
              "`EVAL3_V10_RECIPE=v4_balanced_new66 ./scripts/run_eval3_smolvla_v10_train.sh`.")
    elif code == 1:
        print("v4 balanced corpus has all 3 slots per repo but is uneven. "
              "Training is OK to start; expect mildly weaker language conditioning. "
              "If v10 underperforms, regenerate with more configs/dataset.")
    else:
        print("v4 balanced corpus is broken. Regenerate via "
              "`EVAL3_SYNTH_BALANCED=1 EVAL3_SYNTH_OUTPUT_VERSION=v4 EVAL3_SYNTH_WORKERS=3 "
              "EVAL3_SYNTH_PUSH_TO_HUB=1 ./scripts/run_eval3_synth_dataset_gen.sh` "
              "and re-run this preflight. Do NOT launch training until this PASSes.")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("v3_slots", "v4_balanced"), default="v3_slots",
                    help="v3_slots (default) = check action duplication across the "
                         "v3 per-slot synth repos (proves the v9 collapse). "
                         "v4_balanced = preflight the new v4 balanced corpus "
                         "before launching the v10 trainer.")
    ap.add_argument("--slots", nargs="+", default=list(SLOTS), choices=list(SLOTS),
                    help="(v3_slots mode) Slots to compare across celebs.")
    ap.add_argument("--celebs", default=",".join(CELEBS),
                    help="(v4_balanced mode) Comma-separated celebs to preflight. "
                         "Default: all three.")
    ap.add_argument("--out", default=None, type=Path,
                    help="JSON artifact path. Default depends on --mode: "
                         "outputs/eval3_diag/celeb_confusion.json for v3_slots, "
                         "outputs/eval3_diag/v4_balanced_preflight.json for v4_balanced.")
    args = ap.parse_args()

    if args.out is None:
        args.out = Path("outputs/eval3_diag") / (
            "celeb_confusion.json" if args.mode == "v3_slots"
            else "v4_balanced_preflight.json"
        )

    if args.mode == "v3_slots":
        return _run_v3_slots(args.slots, args.out)

    celebs = [c.strip() for c in args.celebs.split(",") if c.strip()]
    for c in celebs:
        if c not in CELEBS:
            sys.exit(f"unknown celeb={c!r}; want any of {list(CELEBS)}")
    return _run_v4_balanced(celebs, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

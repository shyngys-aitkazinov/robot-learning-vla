#!/usr/bin/env python3
"""Unit tests for tools/eval3_val_watcher.py.

Exercises the structural pieces that don't need a heavy SmolVLA load:

  W1   slot_from_repo regex coverage (left/middle/right; _full suffix)
  W2   identity_from_repo coverage (swift/lecun/obama variants)
  W3   detect_rename_map reads train_config.json for v17 and falls
       back to single-cam default for v16/legacy
  W4   is_v17_ckpt detects frame0 keys
  W5   sample_frames is deterministic, honours episodes_per_repo /
       frames_per_episode, and never returns out-of-bounds indices
  W6   _image_to_uint8 handles tensor / numpy / CHW / HWC / float / uint8
  W7   _action_to_np flattens to 6 regardless of input shape
  W8   build_observation includes frame0 iff frame0_img is provided
  W9   _summarize aggregates overall + per-slot + per-repo correctly
  W10  _summarize tolerates None slot_correct (when target_idx is None)
  W11  _append_jsonl writes header lazily, ONCE, on first call
  W12  local_root_for resolves vs raises on missing
  W13  _parse_prompts respects EVAL3_VAL_PROMPTS JSON override
  W14  load_cfg env-var precedence + CLI override

Run::

  python tools/eval3_val_watcher_unit_tests.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

# We need to import the watcher BEFORE running tests. Import side-effects
# include applying the slot patch — that's fine, it's idempotent and we don't
# load a real policy in this test.
import numpy as np
import torch

import eval3_val_watcher as VW

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"[{name}] PASS  {detail}")
    else:
        _failed += 1
        print(f"[{name}] FAIL  {detail}")


def main() -> int:
    # ---- W1: slot_from_repo ------------------------------------------------
    print("\n=== W1: slot_from_repo ===")
    cases_w1 = [
        ("RobotLearningVLA/dataset_v4_taylor_left", "left"),
        ("RobotLearningVLA/dataset_v4_taylor_middle", "middle"),
        ("RobotLearningVLA/dataset_v4_yann_right", "right"),
        ("RobotLearningVLA/dataset_v5_synth_algvr_someone_left_full", "left"),
        ("RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_middle_3", "middle"),
        ("RobotLearningVLA/dataset_v2_barack_obama_right_1", "right"),
        ("RobotLearningVLA/totally_unrelated_repo", None),
    ]
    for repo, expected in cases_w1:
        got = VW.slot_from_repo(repo)
        check(f"W1[{expected}]", got == expected, f"{repo!r} -> {got!r}")

    # ---- W2: identity_from_repo --------------------------------------------
    print("\n=== W2: identity_from_repo ===")
    cases_w2 = [
        ("RobotLearningVLA/dataset_v4_taylor_left", "swift"),
        ("RobotLearningVLA/dataset_v4_yann_middle", "lecun"),
        ("RobotLearningVLA/dataset_v4_barack_right", "obama"),
        ("RobotLearningVLA/taylor_swift_1", "swift"),
        ("RobotLearningVLA/dataset_v2_barack_obama_left_1", "obama"),
        ("RobotLearningVLA/dataset_v5_synth_algvr_someone_left_full", None),
    ]
    for repo, expected in cases_w2:
        got = VW.identity_from_repo(repo)
        check(f"W2[{expected}]", got == expected, f"{repo!r} -> {got!r}")

    # ---- W3 + W4: detect_rename_map + is_v17_ckpt --------------------------
    print("\n=== W3+W4: detect_rename_map / is_v17_ckpt ===")
    with tempfile.TemporaryDirectory() as td:
        # v17-style ckpt: has frame0 rename
        v17_dir = Path(td) / "ckpt_v17"
        v17_dir.mkdir()
        (v17_dir / "train_config.json").write_text(json.dumps({
            "rename_map": {
                "observation.images.front": "observation.images.camera1",
                "observation.images.front_frame0": "observation.images.camera2",
            }
        }))
        rmap_v17 = VW.detect_rename_map(str(v17_dir))
        check("W3-v17", rmap_v17 == {
            "observation.images.front": "observation.images.camera1",
            "observation.images.front_frame0": "observation.images.camera2",
        }, f"got {rmap_v17}")
        check("W4-v17", VW.is_v17_ckpt(rmap_v17) is True, "v17 detected")

        # v16-style ckpt: single-cam rename
        v16_dir = Path(td) / "ckpt_v16"
        v16_dir.mkdir()
        (v16_dir / "train_config.json").write_text(json.dumps({
            "rename_map": {"observation.images.front": "observation.images.camera1"}
        }))
        rmap_v16 = VW.detect_rename_map(str(v16_dir))
        check("W3-v16", rmap_v16 == {"observation.images.front": "observation.images.camera1"},
              f"got {rmap_v16}")
        check("W4-v16", VW.is_v17_ckpt(rmap_v16) is False, "v16 NOT detected as v17")

        # No train_config.json at all -> default single-cam fallback
        empty_dir = Path(td) / "ckpt_empty"
        empty_dir.mkdir()
        rmap_default = VW.detect_rename_map(str(empty_dir))
        check("W3-default", rmap_default == {"observation.images.front": "observation.images.camera1"},
              f"got {rmap_default}")

    # ---- W5: sample_frames determinism + bounds ----------------------------
    print("\n=== W5: sample_frames ===")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("RobotLearningVLA/dataset_v4_taylor_left", video_backend="pyav")
    samples_a = VW.sample_frames(ds, episodes_per_repo=3, frames_per_episode=5, seed=0)
    samples_b = VW.sample_frames(ds, episodes_per_repo=3, frames_per_episode=5, seed=0)
    check("W5-deterministic", samples_a == samples_b,
          f"two calls with same seed return same list ({len(samples_a)} samples)")
    # bounds check: every returned frame_idx is inside its episode
    bounds_ok = True
    for ep_idx, oi, local_idx in samples_a:
        f0, f1 = VW._episode_bounds(ds, ep_idx)
        if not (f0 <= oi < f1):
            bounds_ok = False
            break
        if oi - f0 != local_idx:
            bounds_ok = False
            break
    check("W5-bounds", bounds_ok,
          "all returned (ep, frame_idx, local_idx) inside [ep_from, ep_to)")
    # frames_per_episode honored: each episode should contribute <= frames_per_episode rows
    counts: dict[int, int] = {}
    for ep_idx, _, _ in samples_a:
        counts[ep_idx] = counts.get(ep_idx, 0) + 1
    check("W5-frames_cap", all(c <= 5 for c in counts.values()),
          f"max per-episode count {max(counts.values()) if counts else 0} <= 5")

    # ---- W6: _image_to_uint8 -----------------------------------------------
    print("\n=== W6: _image_to_uint8 ===")
    chw_f32 = torch.rand(3, 16, 24)  # CHW float in [0,1]
    out1 = VW._image_to_uint8(chw_f32)
    check("W6-chw_float", out1.shape == (16, 24, 3) and out1.dtype == np.uint8,
          f"chw float32 -> {out1.shape} {out1.dtype}")

    hwc_u8 = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    out2 = VW._image_to_uint8(hwc_u8)
    check("W6-hwc_uint8", out2.shape == (32, 48, 3) and out2.dtype == np.uint8,
          f"hwc uint8 -> {out2.shape} {out2.dtype}")

    chw_u8 = (torch.rand(3, 8, 8) * 255).to(torch.uint8)
    out3 = VW._image_to_uint8(chw_u8)
    check("W6-chw_uint8", out3.shape == (8, 8, 3) and out3.dtype == np.uint8,
          f"chw uint8 tensor -> {out3.shape} {out3.dtype}")

    # ---- W7: _action_to_np -------------------------------------------------
    print("\n=== W7: _action_to_np ===")
    a1 = VW._action_to_np(torch.zeros(6))
    a2 = VW._action_to_np(np.zeros(6))
    a3 = VW._action_to_np(torch.zeros(50, 6))  # action chunk
    a4 = VW._action_to_np([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])  # extra trailing
    check("W7-tensor6", a1.shape == (6,) and a1.dtype == np.float32, f"{a1.shape} {a1.dtype}")
    check("W7-np6", a2.shape == (6,) and a2.dtype == np.float32, f"{a2.shape} {a2.dtype}")
    check("W7-chunk", a3.shape == (6,), f"chunk flattened first 6: {a3.shape}")
    check("W7-trim", a4.shape == (6,) and np.allclose(a4, [1, 2, 3, 4, 5, 6]),
          f"trimmed to first 6: {a4.tolist()}")

    # ---- W8: build_observation ---------------------------------------------
    print("\n=== W8: build_observation ===")
    fake_row = {
        VW.IMAGE_KEY: torch.rand(3, 16, 16),
        VW.STATE_KEY: torch.zeros(6),
    }
    obs_no_f0 = VW.build_observation(fake_row, frame0_img=None)
    check("W8-no_frame0",
          VW.IMAGE_KEY in obs_no_f0 and VW.STATE_KEY in obs_no_f0 and VW.FRAME0_KEY not in obs_no_f0,
          f"obs keys without frame0: {sorted(obs_no_f0)}")
    obs_with_f0 = VW.build_observation(fake_row,
                                       frame0_img=np.zeros((16, 16, 3), dtype=np.uint8))
    check("W8-with_frame0",
          VW.FRAME0_KEY in obs_with_f0 and obs_with_f0[VW.FRAME0_KEY].shape == (16, 16, 3),
          f"obs has frame0: {sorted(obs_with_f0)}")

    # ---- W9 + W10: _summarize ----------------------------------------------
    print("\n=== W9+W10: _summarize aggregation ===")
    rows_left = [{
        "ep_idx": 0, "frame_idx": 10, "local_idx": 10, "identity": "swift", "slot": "left",
        "target_idx": 0, "slot_pred": 0, "slot_correct": 1,
        "action_mae": 1.5, "per_joint_abs_err": [1.0, 1.5, 1.5, 1.5, 2.0, 0.5],
        "prompt_nearest": "swift", "prompt_nearest_correct": 1,
        "cross_prompt_delta": 22.0,
    }, {
        "ep_idx": 0, "frame_idx": 50, "local_idx": 50, "identity": "swift", "slot": "left",
        "target_idx": 0, "slot_pred": 1, "slot_correct": 0,
        "action_mae": 3.0, "per_joint_abs_err": [2.0, 3.0, 3.5, 3.0, 5.0, 1.5],
        "prompt_nearest": "lecun", "prompt_nearest_correct": 0,
        "cross_prompt_delta": 18.0,
    }]
    rows_middle = [{
        "ep_idx": 0, "frame_idx": 5, "local_idx": 5, "identity": "lecun", "slot": "middle",
        "target_idx": 1, "slot_pred": 1, "slot_correct": 1,
        "action_mae": 2.0, "per_joint_abs_err": [1.5, 2.0, 2.0, 2.0, 3.0, 1.5],
        "prompt_nearest": "lecun", "prompt_nearest_correct": 1,
        "cross_prompt_delta": 27.0,
    }]
    per_repo = {"swift_repo": rows_left, "lecun_repo": rows_middle}
    summary = VW._summarize(per_repo, t_start=0.0)
    o = summary["overall"]
    check("W9-overall_n", o["n_frames"] == 3, f"overall n_frames={o['n_frames']}")
    check("W9-overall_slot_acc",
          abs(o["slot_acc"] - (1 + 0 + 1) / 3) < 1e-9, f"slot_acc={o['slot_acc']}")
    check("W9-overall_mae",
          abs(o["action_mae"] - (1.5 + 3.0 + 2.0) / 3) < 1e-9, f"action_mae={o['action_mae']}")
    check("W9-overall_nearest",
          abs(o["prompt_nearest_accuracy"] - (1 + 0 + 1) / 3) < 1e-9,
          f"prompt_nearest_accuracy={o['prompt_nearest_accuracy']}")
    check("W9-overall_delta",
          abs(o["cross_prompt_delta"] - (22 + 18 + 27) / 3) < 1e-9,
          f"cross_prompt_delta={o['cross_prompt_delta']}")
    check("W9-per_joint",
          abs(o["action_mae_per_joint"]["wrist_roll"] - (2.0 + 5.0 + 3.0) / 3) < 1e-9,
          f"wrist_roll per-joint MAE={o['action_mae_per_joint']['wrist_roll']}")
    check("W9-per_slot_left",
          summary["per_slot"]["left"]["n_frames"] == 2 and
          abs(summary["per_slot"]["left"]["slot_acc"] - 0.5) < 1e-9,
          f"left slot_acc={summary['per_slot']['left']['slot_acc']}")
    check("W9-per_slot_right",
          summary["per_slot"]["right"]["n_frames"] == 0,
          "right slot has no rows; agg returns n=0 placeholder")
    check("W9-per_repo_count",
          len(summary["per_repo"]) == 2,
          f"per_repo entries: {[r['repo'] for r in summary['per_repo']]}")

    # W10: slot_correct=None tolerated (no slot logits available)
    rows_no_slot = [{
        "ep_idx": 0, "frame_idx": 5, "local_idx": 5, "identity": "swift", "slot": None,
        "target_idx": None, "slot_pred": None, "slot_correct": None,
        "action_mae": 1.0, "per_joint_abs_err": [1.0] * 6,
        "prompt_nearest": "swift", "prompt_nearest_correct": 1, "cross_prompt_delta": 20.0,
    }]
    summary2 = VW._summarize({"r": rows_no_slot}, t_start=0.0)
    check("W10-none_slot_acc",
          summary2["overall"]["slot_acc"] is None,
          f"slot_acc with all None slot_correct: {summary2['overall']['slot_acc']}")
    check("W10-other_metrics",
          summary2["overall"]["action_mae"] == 1.0 and
          summary2["overall"]["prompt_nearest_accuracy"] == 1.0,
          "other metrics still computed when slot_acc is None")

    # ---- W11: lazy JSONL header --------------------------------------------
    print("\n=== W11: _append_jsonl lazy header ===")
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "subdir" / "val.jsonl"
        cfg = VW.Cfg(
            train_out=Path(td), policy_path=None,
            val_repos=["a", "b"], val_local_repos=set(),
            episodes_per_repo=3, frames_per_episode=30,
            prompts={"swift": "x"}, device="cpu",
            poll_sec=60, idle_sec=600, out_path=out_path,
            seed=0, final_step=None, once=False,
            wandb_enable=False, wandb_project=None, wandb_name=None,
        )
        VW._append_jsonl(cfg, out_path, {"step": 1, "x": 1})
        VW._append_jsonl(cfg, out_path, {"step": 2, "x": 2})
        VW._append_jsonl(cfg, out_path, {"step": 3, "x": 3})
        lines = out_path.read_text().splitlines()
        check("W11-line_count", len(lines) == 4,
              f"1 header + 3 records = 4 lines, got {len(lines)}")
        hdr = json.loads(lines[0])
        check("W11-header_schema", hdr.get("schema") == "eval3_val_watcher/v1",
              f"first line is the schema header (got {hdr.get('schema')})")
        first = json.loads(lines[1])
        check("W11-first_record", first.get("step") == 1, "second line is first record")
        # bonus: 2nd append must NOT re-write the header
        n_headers = sum(1 for li in lines if '"schema"' in li)
        check("W11-header_once", n_headers == 1, f"header appears exactly once ({n_headers}x)")

    # ---- W12: local_root_for ----------------------------------------------
    print("\n=== W12: local_root_for ===")
    with tempfile.TemporaryDirectory() as td:
        # set up datasets/<name>/meta
        ds_dir = Path(td) / "datasets" / "my_local_v"
        (ds_dir / "meta").mkdir(parents=True)
        old_cwd = Path.cwd()
        try:
            os.chdir(td)
            got = VW.local_root_for("org/my_local_v", {"org/my_local_v"})
            check("W12-resolves",
                  got is not None and Path(got).name == "my_local_v",
                  f"resolved to {got}")
            # Repo not in local_set -> None
            check("W12-not_local",
                  VW.local_root_for("org/my_local_v", set()) is None,
                  "no entry in local_set -> None")
            # Repo in local_set but no datasets/<name>/meta -> raises
            raised = False
            try:
                VW.local_root_for("org/missing", {"org/missing"})
            except FileNotFoundError:
                raised = True
            check("W12-raises", raised,
                  "in local_set but missing datasets/<name>/meta -> FileNotFoundError")
        finally:
            os.chdir(old_cwd)

    # ---- W13: _parse_prompts -----------------------------------------------
    print("\n=== W13: _parse_prompts ===")
    # default when env unset
    os.environ.pop("EVAL3_VAL_PROMPTS", None)
    defaults = VW._parse_prompts()
    check("W13-default", set(defaults) == set(VW.DEFAULT_PROMPTS),
          f"default keys: {sorted(defaults)}")
    # override via JSON
    os.environ["EVAL3_VAL_PROMPTS"] = json.dumps({"a": "p1", "b": "p2"})
    try:
        parsed = VW._parse_prompts()
        check("W13-override", parsed == {"a": "p1", "b": "p2"},
              f"override: {parsed}")
    finally:
        os.environ.pop("EVAL3_VAL_PROMPTS", None)
    # bad JSON -> fall back to defaults
    os.environ["EVAL3_VAL_PROMPTS"] = "{not json"
    try:
        parsed = VW._parse_prompts()
        check("W13-bad_json", set(parsed) == set(VW.DEFAULT_PROMPTS),
              "bad JSON falls back to defaults")
    finally:
        os.environ.pop("EVAL3_VAL_PROMPTS", None)

    # ---- W14: load_cfg env + CLI precedence -------------------------------
    print("\n=== W14: load_cfg env + CLI precedence ===")
    # Build a tiny args namespace
    args = argparse.Namespace(
        train_out=None, policy_path=None, once=False,
        val_repos=None, val_local_repos=None,
        episodes_per_repo=None, frames_per_episode=None,
        device=None, poll_sec=None, idle_sec=None, seed=None,
        final_step=None,
        wandb=False, wandb_project=None, wandb_name=None,
    )
    # env-only path
    os.environ["EVAL3_VAL_REPOS"] = "a,b,c"
    os.environ["EVAL3_VAL_EPISODES_PER_REPO"] = "7"
    os.environ["EVAL3_VAL_DEVICE"] = "cpu"
    try:
        cfg = VW.load_cfg(args)
        check("W14-env_repos", cfg.val_repos == ["a", "b", "c"], f"val_repos={cfg.val_repos}")
        check("W14-env_ep", cfg.episodes_per_repo == 7,
              f"episodes_per_repo={cfg.episodes_per_repo}")
        check("W14-env_device", cfg.device == "cpu", f"device={cfg.device}")
    finally:
        os.environ.pop("EVAL3_VAL_REPOS", None)
        os.environ.pop("EVAL3_VAL_EPISODES_PER_REPO", None)
        os.environ.pop("EVAL3_VAL_DEVICE", None)
    # CLI overrides env
    os.environ["EVAL3_VAL_REPOS"] = "z"
    args2 = argparse.Namespace(**{**vars(args), "val_repos": ["x", "y"],
                                   "episodes_per_repo": 11, "device": "mps"})
    try:
        cfg2 = VW.load_cfg(args2)
        check("W14-cli_repos", cfg2.val_repos == ["x", "y"],
              "CLI --val-repos beats env")
        check("W14-cli_ep", cfg2.episodes_per_repo == 11,
              f"CLI --episodes-per-repo beats env: {cfg2.episodes_per_repo}")
        check("W14-cli_device", cfg2.device == "mps",
              "CLI --device beats env")
    finally:
        os.environ.pop("EVAL3_VAL_REPOS", None)

    print(f"\n{'=' * 60}\n  RESULTS: {_passed} passed / {_failed} failed\n{'=' * 60}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

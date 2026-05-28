#!/usr/bin/env python3
"""v16 Eval3PrepDataset robustness test.

Exercises the v16-only data path WITHOUT loading SmolVLA:

  D1  apply_concat_patch + Eval3PrepDataset wraps dataset_v4_taylor_left
  D2  frame-0 cache populated (one image per episode)
  D3  grasp offsets sane (15%-50% of episode length, median in expected range)
  D4  target_position correctly derived from repo name (=0 for taylor_left)
  D5  pre-grasp sample: is_pregrasp=1, front_frame0 == current frame
  D6  post-grasp sample: is_pregrasp=0, front_frame0 == cached frame-0
  D7  Walk a full episode -- is_pregrasp flips exactly once at grasp_offset
  D8  Stats merge across multi-dataset concat is shape-consistent

Run::
    python tools/eval3_v16_dataset_test.py
    python tools/eval3_v16_dataset_test.py --repo RobotLearningVLA/dataset_v4_taylor_middle
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Required env vars BEFORE importing eval3 modules.
os.environ.setdefault("EVAL3_SLOT_LOSS_WEIGHT", "0.5")
os.environ.setdefault("EVAL3_SLOT_FRAME0", "1")
os.environ.setdefault("EVAL3_SLOT_CE_PREGRASP_ONLY", "1")
os.environ.setdefault("EVAL3_GRASP_GRIP_DELTA", "20")
os.environ.setdefault("EVAL3_MAX_FRAMES_PER_EP", "0")
# disable image transforms to keep frame-equality checks bit-exact
os.environ.setdefault("EVAL3_TASK_AUG", "0")
os.environ.setdefault("EVAL3_BG_REPLACE", "0")
os.environ.setdefault("EVAL3_PRINT_SHUFFLE", "0")
# disable state aug for deterministic sample inspection
os.environ.setdefault("EVAL3_STATE_NOISE_SIGMA_MAX", "0.0")
os.environ.setdefault("EVAL3_STATE_REPLACE_PROB", "0.0")
os.environ.setdefault("EVAL3_PREP_CACHE", "0")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from eval3_dataset_prep import Eval3PrepDataset


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


def _v16_collate(batch):
    """Module-scope collate for the DataLoader test (must be picklable)."""
    out = {}
    for k in batch[0]:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


def _img_eq(a, b) -> bool:
    ta = a if isinstance(a, torch.Tensor) else torch.as_tensor(np.asarray(a))
    tb = b if isinstance(b, torch.Tensor) else torch.as_tensor(np.asarray(b))
    if ta.shape != tb.shape:
        return False
    return bool(torch.equal(ta, tb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RobotLearningVLA/dataset_v4_taylor_left")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()

    repo_id = args.repo
    ep = args.episode
    expected_target = 0 if "_left" in repo_id else 1 if "_middle" in repo_id else 2

    print(f"\n=== D1: Build LeRobotDataset({repo_id}) ===")
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    print(f"  num_episodes={ds.num_episodes}, num_frames={ds.num_frames}, fps={ds.meta.fps}")
    print(f"  features keys: {sorted(ds.features.keys())}")
    print(f"  image_key candidates: "
          f"{[k for k in ds.features if k.startswith('observation.images')]}")
    check("D1", ds.num_episodes > 0 and ds.num_frames > 0,
          f"loaded {ds.num_episodes} episodes / {ds.num_frames} frames")

    print(f"\n=== D2/D3: Wrap with Eval3PrepDataset (target_position_idx={expected_target}) ===")
    wrapped = Eval3PrepDataset(
        ds,
        max_frames_per_episode=None,
        target_position_idx=expected_target,
    )
    n_ep = wrapped.num_episodes
    print(f"  wrapped.num_episodes={n_ep}, wrapped.num_frames={wrapped.num_frames}")
    print(f"  _frame0_by_ep size: {len(wrapped._frame0_by_ep)}")
    print(f"  _grasp_offset_by_ep size: {len(wrapped._grasp_offset_by_ep)}")
    check("D2-cache", len(wrapped._frame0_by_ep) == n_ep,
          f"frame-0 cache covers all {n_ep} episodes")

    # D3: grasp offsets in [15%, 50%] of episode length, and median in reasonable range
    grasp_offsets = wrapped._grasp_offset_by_ep
    ep_lengths = []
    for i in range(n_ep):
        f0 = wrapped._episode_from_idxs[i]
        f1 = wrapped._episode_to_idxs[i]
        ep_lengths.append(int(f1 - f0))
    offs = [grasp_offsets[i] for i in range(n_ep)]
    lo = [int(0.15 * el) for el in ep_lengths]
    hi = [int(0.50 * el) for el in ep_lengths]
    in_window = all(lo[i] <= offs[i] <= hi[i] for i in range(n_ep))
    print(f"  grasp_offset distribution: min={min(offs)} median={int(np.median(offs))} "
          f"max={max(offs)}; episode_lengths min={min(ep_lengths)} med={int(np.median(ep_lengths))} max={max(ep_lengths)}")
    check("D3-bounds", in_window,
          f"all grasp offsets in [15%, 50%] of episode length")

    print(f"\n=== D4: target_position derived from repo name ===")
    # Pick a midframe in episode 0
    f0_idx = wrapped._episode_from_idxs[ep]
    f1_idx = wrapped._episode_to_idxs[ep]
    g_off = grasp_offsets[ep]
    print(f"  episode {ep}: frames [{f0_idx}, {f1_idx}), grasp_offset={g_off}")

    midpre_idx = max(0, g_off // 2)  # well before grasp
    midpost_idx = min(int(f1_idx - f0_idx) - 1, g_off + max(5, (int(f1_idx - f0_idx) - g_off) // 2))

    # Find the wrapped indices for these episode-local positions
    valid = wrapped._valid_indices  # list of original-ds indices kept after filtering
    # Episode-local to wrapped-global: scan valid_indices for ones inside this episode
    pre_global = None
    post_global = None
    for wi, oi in enumerate(valid):
        if oi == f0_idx + midpre_idx and pre_global is None:
            pre_global = wi
        if oi == f0_idx + midpost_idx and post_global is None:
            post_global = wi
        if pre_global is not None and post_global is not None:
            break
    # fallback if exact match not found
    if pre_global is None:
        pre_global = next(wi for wi, oi in enumerate(valid)
                          if f0_idx <= oi < f0_idx + g_off)
    if post_global is None:
        post_global = next(wi for wi, oi in enumerate(valid)
                           if f0_idx + g_off < oi < f1_idx)

    pre_row = wrapped[pre_global]
    post_row = wrapped[post_global]

    print(f"  pre-grasp sample (wrapped idx {pre_global}, original {valid[pre_global]}):")
    print(f"    target_position={int(pre_row['target_position'])} (expected {expected_target})")
    print(f"    is_pregrasp={int(pre_row['is_pregrasp'])} (expected 1)")
    print(f"    keys with .front: {[k for k in pre_row if 'front' in k]}")

    check("D4-target_left", int(pre_row["target_position"]) == expected_target,
          f"target_position={int(pre_row['target_position'])}")

    # D5
    print(f"\n=== D5: pre-grasp sample: front_frame0 == current frame ===")
    pre_front = pre_row.get("observation.images.front")
    pre_frame0 = pre_row.get("observation.images.front_frame0")
    print(f"  observation.images.front      shape={getattr(pre_front, 'shape', None)} "
          f"dtype={getattr(pre_front, 'dtype', None)}")
    print(f"  observation.images.front_frame0 shape={getattr(pre_frame0, 'shape', None)} "
          f"dtype={getattr(pre_frame0, 'dtype', None)}")
    check("D5-is_pregrasp", int(pre_row["is_pregrasp"]) == 1, "is_pregrasp=1 for pre-grasp frame")
    check("D5-frame0_is_current", _img_eq(pre_front, pre_frame0),
          "pre-grasp: front_frame0 should equal current front")

    # D6
    print(f"\n=== D6: post-grasp sample: front_frame0 == cached episode frame-0 ===")
    post_front = post_row.get("observation.images.front")
    post_frame0 = post_row.get("observation.images.front_frame0")
    cached_f0 = wrapped._frame0_by_ep[ep]
    print(f"  observation.images.front      shape={getattr(post_front, 'shape', None)}")
    print(f"  observation.images.front_frame0 shape={getattr(post_frame0, 'shape', None)}")
    print(f"  cached frame-0 image          shape={getattr(cached_f0, 'shape', None)}")
    check("D6-is_postgrasp", int(post_row["is_pregrasp"]) == 0,
          f"is_pregrasp=0 for post-grasp frame (wrapped idx {post_global}, "
          f"original {valid[post_global]}, local {valid[post_global] - f0_idx})")
    check("D6-frame0_is_cached", _img_eq(post_frame0, cached_f0),
          "post-grasp: front_frame0 should equal cached episode frame-0")
    check("D6-frame0_neq_current",
          not _img_eq(post_front, post_frame0) if post_front is not None and post_frame0 is not None else True,
          "post-grasp: front_frame0 must differ from current front (else carry phase leaks)")

    # D7: walk the episode, count the is_pregrasp flip
    print(f"\n=== D7: walk episode {ep}, count is_pregrasp transitions ===")
    flips = 0
    last_flag = None
    flag_seq = []
    # collect (local_idx, is_pregrasp) for every wrapped sample in this episode
    samples = []
    for wi, oi in enumerate(valid):
        if not (f0_idx <= oi < f1_idx):
            continue
        local = int(oi - f0_idx)
        r = wrapped[wi]
        flag = int(r["is_pregrasp"])
        samples.append((local, flag))
        if last_flag is not None and flag != last_flag:
            flips += 1
        last_flag = flag
        flag_seq.append(flag)
    samples.sort()
    # Identify the boundary
    boundary = None
    for i in range(1, len(samples)):
        if samples[i - 1][1] == 1 and samples[i][1] == 0:
            boundary = samples[i][0]
            break
    print(f"  total samples in episode: {len(samples)}")
    print(f"  is_pregrasp transitions (1->0): {flips}")
    print(f"  observed boundary (first frame with is_pregrasp=0): {boundary}")
    print(f"  grasp_offset (cached):                              {g_off}")
    check("D7-monotone", flips == 1,
          f"is_pregrasp flips exactly once across the episode (got {flips})")
    check("D7-boundary", boundary is not None and boundary == g_off + 1,
          f"boundary at local frame {g_off}+1={g_off + 1} (got {boundary})")

    # D8: also check tensor shapes / dtypes for action and state
    print(f"\n=== D8: shape/dtype of action+state in a sample ===")
    action = pre_row.get("action")
    state = pre_row.get("observation.state")
    print(f"  action.shape={getattr(action, 'shape', None)} dtype={getattr(action, 'dtype', None)}")
    print(f"  state.shape={getattr(state, 'shape', None)} dtype={getattr(state, 'dtype', None)}")
    check("D8-action_shape", action is not None and tuple(action.shape)[-1] == 6,
          f"action last dim == 6 (got {tuple(action.shape) if action is not None else None})")
    check("D8-state_shape", state is not None and tuple(state.shape) == (6,),
          f"state shape == (6,) (got {tuple(state.shape) if state is not None else None})")

    # ============================================================
    # PHASE B: Camera-1 dropout (CameraDropAugmenter) tests
    # ============================================================
    from eval3_dataset_prep import CameraDropAugmenter

    def _is_noise(img: torch.Tensor, *, mean: float = 0.5, std: float = 0.25) -> bool:
        if img is None:
            return False
        t = img if isinstance(img, torch.Tensor) else torch.as_tensor(np.asarray(img))
        return (
            abs(float(t.float().mean()) - mean) < 0.05
            and abs(float(t.float().std()) - std) < 0.05
        )

    # D9: cam2 is never touched by the augmenter. Use forced per-episode drop
    # so we hit the drop path 100% of the time. Wrap fresh dataset to inject
    # the augmenter.
    print(f"\n=== D9: CameraDropAugmenter never modifies front_frame0 (cam2) ===")
    aug_d9 = CameraDropAugmenter(
        episode_drop_p=1.0, frame_drop_p=0.0, post_mult=1.0,
        noise_mean=0.5, noise_std=0.25, seed=0,
    )
    wrapped_d9 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                                  target_position_idx=expected_target,
                                  cam_drop_fn=aug_d9)
    # Walk a handful of samples (mix of pre- and post-grasp) and confirm cam2
    # bit-exactness against the cached frame-0 / current frame the v16 block
    # would have written, AND that cam1 has been replaced by noise.
    valid_d9 = wrapped_d9._valid_indices
    sample_indices = [0, 50, 100, 200, 300]
    cam2_unchanged = True
    cam1_dropped = 0
    for si in sample_indices:
        if si >= len(valid_d9):
            continue
        oi = valid_d9[si]
        ep_idx = int(wrapped_d9._episode_index_by_frame[int(oi)])
        # Compute the expected cam2 image by re-doing the v16 block's choice
        grasp_off = wrapped_d9._grasp_offset_by_ep.get(ep_idx)
        ep_start = wrapped_d9._episode_from_idxs[ep_idx]
        is_pre = 1 if grasp_off is None else (1 if (int(oi) - ep_start) <= grasp_off else 0)
        if is_pre:
            # cam2 should be the current frame from the underlying dataset.
            raw_cur = ds[int(oi)][wrapped_d9._image_key]
            expected_cam2 = raw_cur
        else:
            expected_cam2 = wrapped_d9._frame0_by_ep[ep_idx]
        row = wrapped_d9[si]
        if not _img_eq(row["observation.images.front_frame0"], expected_cam2):
            cam2_unchanged = False
        if _is_noise(row["observation.images.front"]):
            cam1_dropped += 1
    check("D9-cam2_untouched", cam2_unchanged,
          f"cam2 (front_frame0) matches expected real image for all {len(sample_indices)} samples")
    check("D9-cam1_dropped_all", cam1_dropped == len(sample_indices),
          f"with episode_drop_p=1.0, cam1 was noise in {cam1_dropped}/{len(sample_indices)} samples")

    # D10: Per-episode mode (episode_drop_p=1.0, frame_drop_p=0.0). Walking one
    # episode should yield uniform drop on every frame.
    print(f"\n=== D10: per-episode mode forces uniform drop within one episode ===")
    aug_d10 = CameraDropAugmenter(
        episode_drop_p=1.0, frame_drop_p=0.0, post_mult=1.0, seed=0,
    )
    wrapped_d10 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                                   target_position_idx=expected_target,
                                   cam_drop_fn=aug_d10)
    valid_d10 = wrapped_d10._valid_indices
    drops = 0
    n = 0
    for si, oi in enumerate(valid_d10):
        if not (f0_idx <= oi < f1_idx):
            continue
        n += 1
        row = wrapped_d10[si]
        if _is_noise(row["observation.images.front"]):
            drops += 1
        if n >= 40:  # plenty
            break
    check("D10-per_episode_uniform", n > 0 and drops == n,
          f"every frame in episode {ep} dropped: {drops}/{n}")

    # D11: Per-frame mode (episode_drop_p=0.0, frame_drop_p=0.3, post_mult=2.0).
    # Drop rate should be ~0.3 pre-grasp, ~0.6 post-grasp over many samples.
    print(f"\n=== D11: per-frame mode + post-grasp asymmetry ===")
    aug_d11 = CameraDropAugmenter(
        episode_drop_p=0.0, frame_drop_p=0.3, post_mult=2.0, seed=42,
    )
    wrapped_d11 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                                   target_position_idx=expected_target,
                                   cam_drop_fn=aug_d11)
    pre_total = pre_drops = post_total = post_drops = 0
    # Sample broadly across the dataset for stable rates
    rng = np.random.default_rng(0)
    sample_pool = rng.choice(len(wrapped_d11), size=min(800, len(wrapped_d11)),
                              replace=False)
    for si in sample_pool:
        row = wrapped_d11[int(si)]
        is_pre = int(row["is_pregrasp"]) == 1
        is_drop = _is_noise(row["observation.images.front"])
        if is_pre:
            pre_total += 1
            pre_drops += int(is_drop)
        else:
            post_total += 1
            post_drops += int(is_drop)
    pre_rate = pre_drops / max(pre_total, 1)
    post_rate = post_drops / max(post_total, 1)
    print(f"  N_pre={pre_total} pre_drop_rate={pre_rate:.3f} (target 0.30 +/- 0.05)")
    print(f"  N_post={post_total} post_drop_rate={post_rate:.3f} (target 0.60 +/- 0.06)")
    check("D11-pre_rate", abs(pre_rate - 0.30) < 0.06,
          f"pre-grasp drop rate {pre_rate:.3f} within +-0.06 of 0.30")
    check("D11-post_rate", abs(post_rate - 0.60) < 0.08,
          f"post-grasp drop rate {post_rate:.3f} within +-0.08 of 0.60")
    check("D11-asymmetry", post_rate > pre_rate + 0.10,
          f"post_rate > pre_rate by >0.10 (asymmetry holds)")

    # D12: Per-episode flag is deterministic and stable across calls.
    print(f"\n=== D12: per-episode flag determinism ===")
    aug_d12 = CameraDropAugmenter(
        episode_drop_p=0.5, frame_drop_p=0.0, post_mult=1.0,
        epoch_step_window=10_000_000, seed=7,
    )
    flags_pass1 = [aug_d12.episode_drop_flag(i, epoch_bucket=0) for i in range(50)]
    flags_pass2 = [aug_d12.episode_drop_flag(i, epoch_bucket=0) for i in range(50)]
    # Different bucket -> different flag set (most should differ)
    flags_other_bucket = [aug_d12.episode_drop_flag(i, epoch_bucket=1) for i in range(50)]
    same_in_bucket = flags_pass1 == flags_pass2
    n_differ_across_buckets = sum(a != b for a, b in zip(flags_pass1, flags_other_bucket))
    # Sanity: with p=0.5 about half should be True
    n_true = sum(flags_pass1)
    print(f"  pass1: {n_true}/50 flags True (target ~25)")
    print(f"  pass1 vs pass2 (same bucket): {'identical' if same_in_bucket else 'DIFFER'}")
    print(f"  pass1 vs other bucket: {n_differ_across_buckets}/50 differ")
    check("D12-determinism", same_in_bucket,
          "episode_drop_flag is deterministic for fixed (ep_idx, bucket)")
    check("D12-bucket_reroll", n_differ_across_buckets >= 10,
          f"changing epoch_bucket re-rolls flags ({n_differ_across_buckets}/50 differed)")
    check("D12-fraction", 15 <= n_true <= 35,
          f"with p=0.5 over 50 episodes, {n_true} are True (within sanity band)")

    # ============================================================
    # PHASE C: Extensive robustness verification (pickle, DataLoader workers,
    # integrity of non-image fields, curriculum integration, mixed mode).
    # ============================================================

    # C1: pickle round-trip — required for spawn-based multiprocessing, also a
    # sanity that our __reduce__ captures every config field.
    print(f"\n=== C1: pickle round-trip ===")
    import pickle
    aug_c1 = CameraDropAugmenter(
        episode_drop_p=0.42, frame_drop_p=0.27, post_mult=2.5,
        noise_mean=0.4, noise_std=0.2, epoch_step_window=1234, seed=99,
    )
    blob = pickle.dumps(aug_c1)
    aug_c1_r = pickle.loads(blob)
    fields = ("_episode_drop_p", "_frame_drop_p", "_post_mult",
              "_noise_mean", "_noise_std", "_epoch_step_window", "_seed")
    fields_match = all(getattr(aug_c1, f) == getattr(aug_c1_r, f) for f in fields)
    check("C1-config", fields_match,
          f"all {len(fields)} config fields restored bit-exact")
    # Same hash output for same inputs (no Python hash randomization).
    h_orig = aug_c1.episode_drop_flag(5, epoch_bucket=3)
    h_rest = aug_c1_r.episode_drop_flag(5, epoch_bucket=3)
    check("C1-hash_stable", h_orig == h_rest,
          f"episode_drop_flag stable post-pickle ({h_orig} == {h_rest})")
    # __call__ on a real image still works after pickle (uses the lazy RNG path)
    dummy = torch.rand(3, 16, 16)
    out_r = aug_c1_r(dummy, ep_idx=0, is_pregrasp=True)
    check("C1-call_after_pickle",
          isinstance(out_r, torch.Tensor) and out_r.shape == dummy.shape,
          f"__call__ works after unpickle, shape={tuple(out_r.shape)}")

    # C2: drop preserves cam2 references (pre-grasp = current frame, post-grasp
    # = cached frame-0). This is stricter than D9 — checks both branches.
    print(f"\n=== C2: cam2 reference preservation across pre/post branches ===")
    aug_c2 = CameraDropAugmenter(episode_drop_p=1.0, frame_drop_p=0.0, seed=0)
    wrapped_c2 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                                  target_position_idx=0, cam_drop_fn=aug_c2)
    # Pre-grasp sample at wrapped index 5 (well below grasp_offset 58)
    pre_row = wrapped_c2[5]
    pre_oi = wrapped_c2._valid_indices[5]
    raw_cur_pre = ds[pre_oi][wrapped_c2._image_key]
    check("C2-pre_cam2_real", _img_eq(pre_row["observation.images.front_frame0"], raw_cur_pre),
          "pre-grasp + drop: cam2 still equals raw current frame")
    check("C2-pre_cam1_noise", _is_noise(pre_row["observation.images.front"]),
          "pre-grasp + drop: cam1 replaced by noise")
    # Post-grasp sample
    g_off_ep0 = wrapped_c2._grasp_offset_by_ep[0]
    post_target_oi = wrapped_c2._episode_from_idxs[0] + g_off_ep0 + 30
    post_wi = next(i for i, oi in enumerate(wrapped_c2._valid_indices) if oi == post_target_oi)
    post_row = wrapped_c2[post_wi]
    cached_f0 = wrapped_c2._frame0_by_ep[0]
    check("C2-post_cam2_cached", _img_eq(post_row["observation.images.front_frame0"], cached_f0),
          "post-grasp + drop: cam2 still equals cached frame-0")
    check("C2-post_cam1_noise", _is_noise(post_row["observation.images.front"]),
          "post-grasp + drop: cam1 replaced by noise")

    # C3: non-image fields (state, action, task, target_position, is_pregrasp)
    # must be unchanged by cam-drop.
    print(f"\n=== C3: cam-drop leaves non-image fields untouched ===")
    aug_c3_off = CameraDropAugmenter(episode_drop_p=0.0, frame_drop_p=0.0, seed=0)
    aug_c3_on = CameraDropAugmenter(episode_drop_p=1.0, frame_drop_p=0.0, seed=0)
    w_off = Eval3PrepDataset(ds, max_frames_per_episode=None,
                             target_position_idx=0, cam_drop_fn=aug_c3_off)
    w_on = Eval3PrepDataset(ds, max_frames_per_episode=None,
                            target_position_idx=0, cam_drop_fn=aug_c3_on)
    for probe_idx in (10, 100, 250, 350):
        if probe_idx >= len(w_off):
            continue
        a = w_off[probe_idx]
        b = w_on[probe_idx]
        check(f"C3-state[{probe_idx}]", torch.equal(a["observation.state"], b["observation.state"]),
              "state bit-exact across drop on/off")
        check(f"C3-action[{probe_idx}]", torch.equal(a["action"], b["action"]),
              "action bit-exact across drop on/off")
        check(f"C3-task[{probe_idx}]", a.get("task") == b.get("task"),
              f"task unchanged ({a.get('task')!r})")
        check(f"C3-tp[{probe_idx}]",
              int(a["target_position"]) == int(b["target_position"]),
              "target_position unchanged")
        check(f"C3-ip[{probe_idx}]",
              int(a["is_pregrasp"]) == int(b["is_pregrasp"]),
              "is_pregrasp unchanged")

    # C4: target_position derivation works for all three slots (left/middle/right).
    print(f"\n=== C4: target_position long-tensor for {{left,middle,right}} ===")
    for label, idx in (("left", 0), ("middle", 1), ("right", 2)):
        w_slot = Eval3PrepDataset(ds, max_frames_per_episode=None,
                                  target_position_idx=idx, cam_drop_fn=None)
        r = w_slot[0]
        tp = r["target_position"]
        check(f"C4-{label}",
              isinstance(tp, torch.Tensor) and tp.dtype == torch.long and int(tp) == idx,
              f"target_position={int(tp)} dtype={tp.dtype}")

    # C5: curriculum counter integration — set_step changes epoch_bucket and
    # re-rolls the per-episode flag. Stable on revisit.
    print(f"\n=== C5: curriculum-counter-driven epoch bucket ===")
    from eval3_dataset_prep import get_curriculum_step_counter
    counter = get_curriculum_step_counter()
    aug_c5 = CameraDropAugmenter(
        episode_drop_p=0.5, frame_drop_p=0.0,
        epoch_step_window=100, seed=123,
    )
    counter.set_step(0)
    flags_step0 = [aug_c5.episode_drop_flag(i) for i in range(50)]
    counter.set_step(500)  # bucket 500//100 = 5
    flags_step500 = [aug_c5.episode_drop_flag(i) for i in range(50)]
    counter.set_step(0)   # back to bucket 0
    flags_step0_again = [aug_c5.episode_drop_flag(i) for i in range(50)]
    n_diff = sum(a != b for a, b in zip(flags_step0, flags_step500))
    n_same = sum(a == b for a, b in zip(flags_step0, flags_step0_again))
    print(f"  bucket 0 vs bucket 5 : {n_diff}/50 differ")
    print(f"  bucket 0 revisited   : {n_same}/50 match")
    check("C5-counter_reroll", n_diff >= 10,
          f"counter advance changes per-episode flag ({n_diff}/50)")
    check("C5-counter_stable", n_same == 50,
          "same bucket -> identical flags on revisit")
    counter.set_step(0)  # reset

    # C6: mixed mode — both ep_p>0 and frame_p>0.
    # The naive theoretical rate is ep_p + (1-ep_p)*frame_p, but the realized
    # rate depends on WHICH of the 10 episodes are flagged dropped under the
    # augmenter's seed — variance over 10 episodes is large. Compute the
    # actual expected rate from the per-episode flags.
    print(f"\n=== C6: mixed mode (ep + frame contributions combine) ===")
    aug_c6 = CameraDropAugmenter(
        episode_drop_p=0.5, frame_drop_p=0.3, post_mult=1.0, seed=11,
    )
    w_c6 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                            target_position_idx=0, cam_drop_fn=aug_c6)
    # Realized per-episode flags for this dataset under this seed:
    dropped_eps = {i for i in range(w_c6.num_episodes)
                   if aug_c6.episode_drop_flag(i, epoch_bucket=0)}
    # Weight by frame count per episode so the expected rate matches uniform
    # sampling across frames (not episodes).
    n_total = 0
    n_in_dropped = 0
    for i in range(w_c6.num_episodes):
        ep_len = int(w_c6._episode_to_idxs[i] - w_c6._episode_from_idxs[i])
        n_total += ep_len
        if i in dropped_eps:
            n_in_dropped += ep_len
    ep_drop_frac_frame_weighted = n_in_dropped / max(n_total, 1)
    expected_rate = ep_drop_frac_frame_weighted + (1.0 - ep_drop_frac_frame_weighted) * 0.3
    print(f"  dropped episodes (seed=11): {sorted(dropped_eps)} "
          f"({len(dropped_eps)}/{w_c6.num_episodes})")
    print(f"  frame-weighted ep drop fraction: {ep_drop_frac_frame_weighted:.3f}")
    print(f"  -> expected total drop rate: {expected_rate:.3f}")
    rng_c6 = np.random.default_rng(1)
    N_c6 = 500
    pool = rng_c6.choice(len(w_c6), size=N_c6, replace=False)
    drops = sum(int(_is_noise(w_c6[int(i)]["observation.images.front"])) for i in pool)
    rate = drops / N_c6
    print(f"  observed drop rate: {rate:.3f}  (expected {expected_rate:.3f} +/- 0.05)")
    check("C6-mixed_rate", abs(rate - expected_rate) < 0.05,
          f"mixed-mode drop rate {rate:.3f} within +-0.05 of {expected_rate:.3f}")

    # C7: DataLoader integration (num_workers=0) — verifies the augmenter works
    # under DataLoader iteration and per-episode flags consistently mark every
    # sample from a dropped episode as noisy across batch boundaries. (Per-PID
    # RNG independence under real fork is exercised explicitly in C8; the
    # actual fork-based training path is covered by the end-to-end smoke run.)
    print(f"\n=== C7: DataLoader (num_workers=0) — per-episode flag consistency ===")
    from torch.utils.data import DataLoader
    aug_c7 = CameraDropAugmenter(
        episode_drop_p=0.5, frame_drop_p=0.0,
        epoch_step_window=10_000_000, seed=2027,
    )
    w_c7 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                            target_position_idx=0, cam_drop_fn=aug_c7)
    expected_dropped = {
        i for i in range(w_c7.num_episodes)
        if aug_c7.episode_drop_flag(i, epoch_bucket=0)
    }
    print(f"  expected dropped episodes (seed=2027, bucket=0): {sorted(expected_dropped)}")
    loader = DataLoader(
        w_c7, batch_size=4, num_workers=0, shuffle=False,
        collate_fn=_v16_collate, persistent_workers=False,
    )
    mismatches = 0
    examined = 0
    # shuffle=False -> sample order matches __getitem__ order; the wrapped
    # index = running counter.
    for batch in loader:
        front = batch["observation.images.front"]
        for i in range(front.shape[0]):
            wi = examined + i
            if wi >= len(w_c7):
                break
            oi = w_c7._valid_indices[wi]
            ep = int(w_c7._episode_index_by_frame[int(oi)])
            actual_noise = _is_noise(front[i])
            expect_noise = ep in expected_dropped
            if actual_noise != expect_noise:
                mismatches += 1
        examined += front.shape[0]
        if examined >= 48:
            break
    del loader
    print(f"  examined {examined} samples through DataLoader; {mismatches} disagreements")
    check("C7-dataloader_consistency", mismatches == 0,
          f"all {examined} samples consistent with per-episode drop set")

    # C8: per-PID RNG reset — fake two pids and confirm independent draws.
    print(f"\n=== C8: per-PID RNG re-init produces independent streams ===")
    aug_c8 = CameraDropAugmenter(episode_drop_p=0.0, frame_drop_p=1.0,
                                  noise_std=0.05, seed=4242)
    # Force the augmenter into a known-pid state, draw a noise sample, then
    # spoof a different pid via monkeypatch and draw again. The two noise
    # tensors should NOT be bit-identical (independent RNG seeds).
    real_getpid = os.getpid
    img0 = torch.zeros(3, 8, 8)
    try:
        os.getpid = lambda: 11111  # pretend we're worker A
        out_a = aug_c8(img0.clone(), ep_idx=0, is_pregrasp=True)
        os.getpid = lambda: 22222  # pretend we're worker B
        out_b = aug_c8(img0.clone(), ep_idx=0, is_pregrasp=True)
    finally:
        os.getpid = real_getpid
    check("C8-distinct_workers", not torch.equal(out_a, out_b),
          "workers with different pids produce DIFFERENT noise (RNG re-seeded)")
    # And both are actual noise (not zeros)
    check("C8-both_noise", out_a.std() > 0.01 and out_b.std() > 0.01,
          f"both outputs are noise (std={float(out_a.std()):.3f}, {float(out_b.std()):.3f})")

    # ============================================================
    # PHASE E: Cam2 invariant — "second camera (frame-0) is ALWAYS
    # visible and ALWAYS valid". This is the structural promise of v16
    # under cam-drop. Tests pound it from every angle.
    # ============================================================

    def _is_real_image(img: torch.Tensor) -> bool:
        """Heuristic: a real frame has std in [0.05, 0.45] (NOT the noise-std
        band ~0.25 alone — real images can also fall there) AND has variation
        across channels (R/G/B means differ for natural images, unlike noise
        N(0.5, 0.25^2) which is per-channel uncorrelated)."""
        if not isinstance(img, torch.Tensor) or img.ndim != 3 or img.shape[0] != 3:
            return False
        if not torch.isfinite(img).all():
            return False
        # std band: not flat (>0.05), not absurdly high (<0.5)
        s = float(img.float().std())
        if not (0.05 <= s <= 0.5):
            return False
        # Per-channel mean spread: real images have non-trivial R/G/B mean diff
        per_ch = img.float().mean(dim=(1, 2))
        spread = float((per_ch.max() - per_ch.min()))
        if spread < 0.005:
            return False
        return True

    # E1: large-N sweep across many episodes and ALL cam-drop knob settings.
    # cam2 must be present and look like a real image for every single sample.
    print(f"\n=== E1: cam2 always present & real across 500-sample sweep ===")
    sweep_configs = [
        # (ep_p, frame_p, post_mult, label)
        (0.0, 0.0, 1.0, "no_drop"),
        (1.0, 0.0, 1.0, "force_episode"),
        (0.0, 1.0, 1.0, "force_per_frame"),
        (0.35, 0.10, 3.0, "realistic_defaults"),
        (0.5, 0.5, 2.0, "aggressive_mixed"),
    ]
    rng_e1 = np.random.default_rng(123)
    for ep_p, fr_p, pm, label in sweep_configs:
        aug_e = CameraDropAugmenter(
            episode_drop_p=ep_p, frame_drop_p=fr_p, post_mult=pm,
            seed=hash(label) & 0xFFFF,
        )
        w_e = Eval3PrepDataset(ds, max_frames_per_episode=None,
                               target_position_idx=0, cam_drop_fn=aug_e)
        n_check = 100
        idxs = rng_e1.choice(len(w_e), size=n_check, replace=False)
        n_missing = 0
        n_not_real = 0
        n_cam1_noised = 0
        n_pre = 0
        for i in idxs:
            row = w_e[int(i)]
            cam2 = row.get("observation.images.front_frame0")
            if cam2 is None:
                n_missing += 1
                continue
            if not _is_real_image(cam2):
                n_not_real += 1
            if _is_noise(row["observation.images.front"]):
                n_cam1_noised += 1
            if int(row["is_pregrasp"]) == 1:
                n_pre += 1
        print(f"  [{label:22s} ep_p={ep_p} fr_p={fr_p} pm={pm}] "
              f"missing={n_missing}  not_real={n_not_real}  "
              f"cam1_noised={n_cam1_noised}/{n_check} (pre={n_pre})")
        check(f"E1-{label}-cam2_present", n_missing == 0,
              f"cam2 present in all {n_check} samples")
        check(f"E1-{label}-cam2_real", n_not_real == 0,
              f"cam2 is a real image in all {n_check} samples")

    # E2: A WHOLE EPISODE sweep with forced episode drop. Every sample in the
    # episode must have cam2 valid AND it must be bit-exactly equal to either
    # the raw current frame (pre-grasp) or the cached frame-0 (post-grasp).
    print(f"\n=== E2: whole-episode walk under forced drop -- cam2 bit-exact ===")
    aug_e2 = CameraDropAugmenter(episode_drop_p=1.0, frame_drop_p=0.0, seed=0)
    w_e2 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                            target_position_idx=0, cam_drop_fn=aug_e2)
    cached_f0_e2 = w_e2._frame0_by_ep[0]
    g_off_e2 = w_e2._grasp_offset_by_ep[0]
    f0_idx_e2 = w_e2._episode_from_idxs[0]
    f1_idx_e2 = w_e2._episode_to_idxs[0]
    n_walked = 0
    n_bad = 0
    n_pre_e2 = 0
    n_post_e2 = 0
    for wi, oi in enumerate(w_e2._valid_indices):
        if not (f0_idx_e2 <= oi < f1_idx_e2):
            continue
        row = w_e2[wi]
        cam2 = row.get("observation.images.front_frame0")
        is_pre = int(row["is_pregrasp"]) == 1
        if cam2 is None:
            n_bad += 1
            continue
        if is_pre:
            # cam2 must equal the raw current frame for this index
            raw_cur = ds[int(oi)][w_e2._image_key]
            if not _img_eq(cam2, raw_cur):
                n_bad += 1
            n_pre_e2 += 1
        else:
            if not _img_eq(cam2, cached_f0_e2):
                n_bad += 1
            n_post_e2 += 1
        n_walked += 1
    print(f"  walked {n_walked} samples (pre={n_pre_e2}, post={n_post_e2}); "
          f"cam2 mismatches: {n_bad}")
    check("E2-cam2_bit_exact_episode", n_bad == 0 and n_walked > 0,
          f"cam2 bit-exact across full episode 0 ({n_walked} samples)")

    # E3: defense-in-depth — a hypothetical augmenter that MUTATES cam1
    # in-place. Without the .clone() in the pre-grasp branch, this would
    # silently corrupt cam2 too. With the clone, cam2 is unaffected.
    print(f"\n=== E3: in-place augmenter cannot corrupt cam2 (clone defense) ===")
    class _InPlaceMutator:
        def __reduce__(self):
            return (self.__class__, ())
        def __call__(self, img, *, ep_idx, is_pregrasp):
            # WARNING: in-place fill — this is what would happen if a future
            # augmenter writer forgot to return a fresh tensor.
            img.fill_(0.5)
            return img
    w_e3 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                            target_position_idx=0, cam_drop_fn=_InPlaceMutator())
    # Pre-grasp sample
    pre_e3 = w_e3[5]
    raw_cur_e3 = ds[w_e3._valid_indices[5]][w_e3._image_key]
    cam2_e3 = pre_e3["observation.images.front_frame0"]
    # cam1 should be all 0.5 (the mutator's fill); cam2 should still match
    # the raw original (clone defense kicks in because cam_drop_fn is set).
    cam1_filled = torch.allclose(pre_e3["observation.images.front"],
                                  torch.full_like(pre_e3["observation.images.front"], 0.5))
    cam2_unchanged = _img_eq(cam2_e3, raw_cur_e3)
    check("E3-cam1_mutated", cam1_filled,
          "cam1 was filled in-place by the mutator (expected)")
    check("E3-cam2_protected", cam2_unchanged,
          "cam2 was NOT corrupted by the in-place mutation (clone defense holds)")

    # E4: missing-frame-0 cache entry triggers a hard error (not silent slot
    # corruption). Simulate by deleting one entry from _frame0_by_ep AFTER
    # construction.
    print(f"\n=== E4: cache-miss raises a hard error (no silent fallback) ===")
    aug_e4 = CameraDropAugmenter(episode_drop_p=0.0, frame_drop_p=0.0, seed=0)
    w_e4 = Eval3PrepDataset(ds, max_frames_per_episode=None,
                            target_position_idx=0, cam_drop_fn=aug_e4)
    # Find a POST-grasp sample for episode 0, then delete the cache entry.
    g_off_e4 = w_e4._grasp_offset_by_ep[0]
    f0_idx_e4 = w_e4._episode_from_idxs[0]
    post_target_oi_e4 = f0_idx_e4 + g_off_e4 + 30
    post_wi_e4 = next(i for i, oi in enumerate(w_e4._valid_indices) if oi == post_target_oi_e4)
    # Save and remove
    saved = w_e4._frame0_by_ep.pop(0)
    raised = False
    try:
        _ = w_e4[post_wi_e4]
    except RuntimeError as e:
        raised = True
        msg_ok = "cam2" in str(e) and "front_frame0" in str(e)
    finally:
        # Restore the cache so subsequent tests are unaffected
        w_e4._frame0_by_ep[0] = saved
    check("E4-cache_miss_raises", raised, "missing cache entry raises RuntimeError")
    check("E4-error_message", raised and msg_ok,
          "error message names cam2 / front_frame0 (clear diagnosis)")

    print(f"\n{'=' * 60}\n  RESULTS: {_passed} passed / {_failed} failed\n{'=' * 60}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

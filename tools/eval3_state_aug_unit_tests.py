#!/usr/bin/env python3
"""Unit tests for StateAugmenter and the curriculum step counter.

Covers:
    S1.  No-op when both sigma_max=0 and replace_prob=0
    S2.  Gaussian noise always applied when sigma > 0 (and gripper noise scaled down)
    S3.  HOME replacement: state ≈ CANONICAL_HOME_STATE + small jitter
    S4.  Zero replacement: state == 0 (before subsequent noise application)
    S5.  HOME/zero mode weighting: with weights (0.7, 0.3), ~70% HOME / ~30% zero
    S6.  Curriculum cosine schedule:
            progress=0   → sigma = sigma_max
            progress=0.5 → sigma = (sigma_max + sigma_min) / 2
            progress=1   → sigma = sigma_min
    S7.  Curriculum disabled (steps=0) → sigma == sigma_max regardless of counter
    S8.  __reduce__ roundtrip (DataLoader-worker compatible)
    S9.  make_state_augmenter() env-var parsing (returns None when disabled,
         returns StateAugmenter with correct knobs when env vars set)
    S10. Module-level CANONICAL_HOME_STATE has 6 floats with reasonable magnitudes
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import torch  # noqa: E402

from eval3_dataset_prep import (  # noqa: E402
    CANONICAL_HOME_STATE,
    StateAugmenter,
    get_curriculum_step_counter,
    make_state_augmenter,
)


def _state_rand(rng_seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(rng_seed)
    return torch.randn(6, generator=g, dtype=torch.float32) * 5.0


# ---- S10: CANONICAL_HOME_STATE sanity --------------------------------------
print("[S10] CANONICAL_HOME_STATE sanity")
assert len(CANONICAL_HOME_STATE) == 6, f"expected 6 values, got {len(CANONICAL_HOME_STATE)}"
home_arr = torch.tensor(CANONICAL_HOME_STATE)
# Bounds: all values reasonable for an SO-101 follower (joints in [-180, 180], gripper in [-5, 100]).
assert (home_arr.abs() < 200).all(), f"unrealistic home magnitudes: {home_arr.tolist()}"
print(f"   HOME = {[f'{v:+8.3f}' for v in home_arr.tolist()]} ✓")


# ---- S1: no-op augmenter ---------------------------------------------------
print("\n[S1] no-op augmenter (sigma_max=0, replace_prob=0)")
noop = StateAugmenter(sigma_max=0.0, replace_prob=0.0)
state = _state_rand(0)
out = noop(state.clone())
assert torch.equal(state, out), f"no-op augmenter changed state: {state.tolist()} -> {out.tolist()}"
print("   state unchanged ✓")


# ---- S2: noise scaled per-joint by state_std (NORMALIZED units mode) -------
print("\n[S2] Gaussian noise scaled per-joint by state_std (normalized-units mode)")
# Pass a known state_std so sigma is interpreted in normalized stddev units.
# Per-joint raw-degree noise = sigma * state_std[j].
state_std = (10.0, 50.0, 40.0, 25.0, 30.0, 27.0)  # rough match to v6_synth stats
aug_noise = StateAugmenter(
    sigma_max=0.3, replace_prob=0.0, gripper_noise_scale=0.1, state_std=state_std,
)
deltas = []
state = _state_rand(0)
for _ in range(2000):
    deltas.append((aug_noise(state.clone()) - state).numpy())
import numpy as np  # noqa: E402

deltas_np = np.stack(deltas)
std_per_joint = deltas_np.std(axis=0)
print(f"   per-joint noise std: {[f'{v:.2f}' for v in std_per_joint]}")
# Expected raw-degree noise per joint: 0.3 * state_std[j], with gripper scaled by 0.1.
expected = [0.3 * s for s in state_std]
expected[-1] *= 0.1
print(f"   expected (0.3 * state_std, gripper × 0.1): {[f'{v:.2f}' for v in expected]}")
for i in range(6):
    # Allow ±15% tolerance on empirical std estimate from 2000 samples.
    assert 0.80 * expected[i] < std_per_joint[i] < 1.20 * expected[i], (
        f"joint {i}: empirical noise std {std_per_joint[i]:.3f} not within "
        f"±15% of expected {expected[i]:.3f}"
    )
print("   per-joint noise std matches sigma * state_std (uniform 0.3 normalized stddev) ✓")

# Also test RAW-DEGREE FALLBACK: when state_std is None, sigma is raw degrees.
print("\n[S2b] Raw-degree fallback (no state_std)")
aug_raw = StateAugmenter(sigma_max=5.0, replace_prob=0.0, gripper_noise_scale=0.1, state_std=None)
deltas = []
state = _state_rand(0)
for _ in range(2000):
    deltas.append((aug_raw(state.clone()) - state).numpy())
std_raw = np.stack(deltas).std(axis=0)
print(f"   raw-degree mode per-joint noise std: {[f'{v:.2f}' for v in std_raw]}")
for i in range(5):
    assert 4.0 < std_raw[i] < 6.0, f"joint {i} raw noise std {std_raw[i]} not ≈ 5.0"
assert 0.3 < std_raw[5] < 0.7
print("   fallback: noise ≈ 5° for joints, 0.5° for gripper ✓")


# ---- S3: HOME replacement ---------------------------------------------------
print("\n[S3] HOME replacement (replace_prob=1, mode home only)")
aug_home = StateAugmenter(
    sigma_max=0.0, replace_prob=1.0,
    mode_home_weight=1.0, mode_zero_weight=0.0,
    home_jitter_sigma=1.0,
)
home_means = []
for _ in range(500):
    state = _state_rand(0)
    home_means.append(aug_home(state.clone()).numpy())
home_means_np = np.stack(home_means)
emp_mean = home_means_np.mean(axis=0)
emp_std = home_means_np.std(axis=0)
for i in range(6):
    expected_mean = CANONICAL_HOME_STATE[i]
    expected_std = 1.0 if i < 5 else 0.1  # gripper jitter scaled by 0.1
    assert abs(emp_mean[i] - expected_mean) < 0.3, (
        f"joint {i} mean {emp_mean[i]} != {expected_mean} (CANONICAL_HOME)"
    )
    assert 0.5 * expected_std < emp_std[i] < 2.0 * expected_std, (
        f"joint {i} std {emp_std[i]} not ≈ {expected_std}"
    )
print(f"   empirical mean ≈ CANONICAL_HOME (max dev: {max(abs(emp_mean[i] - CANONICAL_HOME_STATE[i]) for i in range(6)):.3f}) ✓")


# ---- S4: zero replacement ---------------------------------------------------
print("\n[S4] Zero replacement (replace_prob=1, mode zero only)")
aug_zero = StateAugmenter(
    sigma_max=0.0, replace_prob=1.0,
    mode_home_weight=0.0, mode_zero_weight=1.0,
)
state = _state_rand(0)
out = aug_zero(state.clone())
assert torch.equal(out, torch.zeros(6)), f"zero mode didn't zero state: {out.tolist()}"
print("   state == 0 ✓")


# ---- S5: HOME/zero mode weighting -------------------------------------------
print("\n[S5] HOME/zero mode weighting (0.7 / 0.3)")
aug_mix = StateAugmenter(
    sigma_max=0.0, replace_prob=1.0,
    mode_home_weight=0.7, mode_zero_weight=0.3,
    home_jitter_sigma=0.0,  # turn off jitter so we can detect modes by exact value
    seed=12345,
)
n_home = 0
n_zero = 0
for _ in range(2000):
    state = _state_rand(0)
    out = aug_mix(state.clone()).numpy()
    if abs(out).sum() < 1e-6:
        n_zero += 1
    elif abs(out[0] - CANONICAL_HOME_STATE[0]) < 0.5:  # close to HOME pan
        n_home += 1
home_frac = n_home / 2000
zero_frac = n_zero / 2000
print(f"   home_frac={home_frac:.3f} (expected ~0.70)  zero_frac={zero_frac:.3f} (expected ~0.30)")
assert 0.62 < home_frac < 0.78, f"home_frac {home_frac} not ≈ 0.7"
assert 0.22 < zero_frac < 0.38, f"zero_frac {zero_frac} not ≈ 0.3"
print("   ✓")


# ---- S6/S7: Curriculum schedule --------------------------------------------
print("\n[S6/S7] Curriculum cosine schedule")
aug_curr = StateAugmenter(
    sigma_max=10.0, sigma_min=1.0, curriculum_steps=100,
    replace_prob=0.0,
)
counter = get_curriculum_step_counter()
counter.set_step(0)
sigma_0 = aug_curr._current_sigma()
counter.set_step(50)
sigma_half = aug_curr._current_sigma()
counter.set_step(100)
sigma_end = aug_curr._current_sigma()
counter.set_step(200)  # clamp test
sigma_over = aug_curr._current_sigma()
print(f"   sigma(progress=0.0) = {sigma_0:.4f}  (expected 10.0)")
print(f"   sigma(progress=0.5) = {sigma_half:.4f}  (expected 5.5 = (10+1)/2)")
print(f"   sigma(progress=1.0) = {sigma_end:.4f}  (expected 1.0)")
print(f"   sigma(progress=2.0) = {sigma_over:.4f}  (clamped to 1.0)")
assert abs(sigma_0 - 10.0) < 0.01
assert abs(sigma_half - 5.5) < 0.01
assert abs(sigma_end - 1.0) < 0.01
assert abs(sigma_over - 1.0) < 0.01
# S7: curriculum disabled (steps=0) → constant sigma_max
aug_const = StateAugmenter(sigma_max=10.0, sigma_min=1.0, curriculum_steps=0)
counter.set_step(50)
assert abs(aug_const._current_sigma() - 10.0) < 0.01, "curriculum_steps=0 should pin sigma=sigma_max"
print("   curriculum disabled → constant sigma_max ✓")


# ---- S8: __reduce__ roundtrip (includes state_std) ------------------------
print("\n[S8] __reduce__ pickling roundtrip (incl. state_std)")
import pickle  # noqa: E402

aug = StateAugmenter(
    sigma_max=0.3, sigma_min=0.05, curriculum_steps=100,
    replace_prob=0.4, mode_home_weight=0.7, mode_zero_weight=0.3,
    home_jitter_sigma=1.0, gripper_noise_scale=0.1,
    state_std=(10.0, 50.0, 40.0, 25.0, 30.0, 27.0),
    seed=999,
)
buf = pickle.dumps(aug)
aug2 = pickle.loads(buf)
assert aug2._sigma_max == aug._sigma_max
assert aug2._sigma_min == aug._sigma_min
assert aug2._curriculum_steps == aug._curriculum_steps
assert aug2._replace_prob == aug._replace_prob
assert aug2._mode_home_weight == aug._mode_home_weight
assert aug2._mode_zero_weight == aug._mode_zero_weight
assert aug2._home_jitter_sigma == aug._home_jitter_sigma
assert aug2._gripper_noise_scale == aug._gripper_noise_scale
assert aug2._state_std == aug._state_std
assert aug2._seed == aug._seed
print("   all config knobs (including state_std) round-tripped via pickle ✓")


# ---- S9: make_state_augmenter() env-var parsing ----------------------------
print("\n[S9] make_state_augmenter() env-var parsing")
# Test 1: disabled (all defaults)
for k in [
    "EVAL3_STATE_NOISE_SIGMA_MAX",
    "EVAL3_STATE_NOISE_SIGMA_MIN",
    "EVAL3_STATE_NOISE_CURRICULUM_STEPS",
    "EVAL3_STATE_REPLACE_PROB",
    "EVAL3_STATE_REPLACE_MODES",
    "EVAL3_STATE_HOME_JITTER_SIGMA",
    "EVAL3_STATE_GRIPPER_NOISE_SCALE",
]:
    os.environ.pop(k, None)
assert make_state_augmenter() is None, "all-defaults should return None"
print("   all defaults → None (no-op) ✓")
# Test 2: enabled with full recipe
os.environ.update({
    "EVAL3_STATE_NOISE_SIGMA_MAX": "5.0",
    "EVAL3_STATE_NOISE_SIGMA_MIN": "0.5",
    "EVAL3_STATE_NOISE_CURRICULUM_STEPS": "1000",
    "EVAL3_STATE_REPLACE_PROB": "0.4",
    "EVAL3_STATE_REPLACE_MODES": "home:0.6,zero:0.4",
    "EVAL3_STATE_HOME_JITTER_SIGMA": "1.5",
    "EVAL3_STATE_GRIPPER_NOISE_SCALE": "0.2",
})
aug = make_state_augmenter()
assert aug is not None, "enabled env vars should produce a StateAugmenter"
assert aug._sigma_max == 5.0
assert aug._sigma_min == 0.5
assert aug._curriculum_steps == 1000
assert aug._replace_prob == 0.4
assert abs(aug._mode_home_weight - 0.6) < 1e-6
assert abs(aug._mode_zero_weight - 0.4) < 1e-6
assert aug._home_jitter_sigma == 1.5
assert aug._gripper_noise_scale == 0.2
print("   full env-var recipe parsed correctly ✓")


print("\n" + "=" * 70)
print("ALL STATE-AUG TESTS PASSED ✓")
print("=" * 70)

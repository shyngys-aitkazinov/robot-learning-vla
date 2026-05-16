# Eval 3 — compute budget (Brev / lab GPUs)

## Principles

1. **Never start multi-hour jobs** before **teleop QA + single-episode overfit** pass ([train_regimes.md](train_regimes.md)).
2. **256×256** training images — avoid full-res SO-101 frames in the inner loop.
3. Prefer **mid-tier GPUs** when they fit batch + model (TA example: **1×A100 80GB**, effective batch **64** via **gradient accumulation 2**) — re-tune for your backbone.

## Template worksheet

| Item | Value |
|------|-------|
| Cloud provider locked | (e.g. one Brev provider — TA recommendation) |
| GPU type | |
| Hours budget / team | |
| Model backbone | (SmolVLA-class vs larger) |
| Train resolution | 256 |
| Per-step batch | |
| Gradient accumulation steps | |
| Expected sec/step | (measure on 100 steps) |

## Smoke procedure

1. **100 training steps** on **1 GPU** with logging — extrapolate hours before committing.
2. Checkpoint every **N** steps only after loss decreases on tiny subset.

## HF token scopes

Fine-grained tokens must include **read** on **`RobotLearningVLA`** datasets or pulls return **404**.

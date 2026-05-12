# Eval 3 — demo-day rollout CLI

## TA requirement

> CLI where TAs type the task instruction and press **Enter** to run the rollout.

## Provided harness

[`scripts/eval3_rollout.py`](../../scripts/eval3_rollout.py):

- Reads **one line** from **stdin** (the instruction).
- Runs a **20 s** control loop at configurable **FPS**.
- Optional **`--policy-path`**: loads the **BC sanity** checkpoint from [`train_eval3_bc_overfit.py`](../../scripts/train_eval3_bc_overfit.py) or your replacement checkpoint **if keys match**.
- **`--mock-frame-*`**: repeats one dataset frame as observation for **integration testing without robot** (not for scoring).

## Robot wiring (team completes)

Closed-loop on SO-101 requires feeding **live** `observation.images.*` + `observation.state` into your policy and streaming **`action`** to the follower at dataset FPS.

Recommended paths:

1. Extend harness with LeRobot **robot** APIs mirroring `lerobot-record` observation pipeline (maintain feature parity with training).
2. Or delegate to **`lerobot-eval`** once you have a **`PreTrainedPolicy`** checkpoint compatible with real-robot env configs in your `lerobot` version — verify against upstream examples.

## Inference constraints reminder

- **No** cloud VLMs, **no** YOLO/detector APIs at rollout time.
- **Single** RGB stream.

## Logging

The harness writes JSONL metadata under `outputs/eval3_rollouts/` for post-mortem.

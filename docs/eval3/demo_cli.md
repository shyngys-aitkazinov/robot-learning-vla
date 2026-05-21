# Eval 3 demo-day CLI (TA workstation)

Interactive launcher for **demo day on Ubuntu**: loads the SmolVLA checkpoint **once**, connects the SO-101 **once**, then the TA types prompts in a loop (no reload between students).

## Quick start (Ubuntu)

```bash
cd robot-learning-vla
source .venv/bin/activate   # after ./install.sh + EVAL3_INSTALL_SMOLVLA_DEPS=1

# Robot port on Linux is often /dev/ttyUSB0 (not macOS usbmodem)
export FOLLOWER_TTY=/dev/ttyUSB0
export CAM_IDX=0
export EVAL3_POLICY_DEVICE=cuda

./scripts/run_eval3_demo_cli.sh
```

At the `prompt>` line:

| Input | Effect |
|-------|--------|
| `taylor` | `Place the coke on Taylor Swift` |
| `yann` | `Place the coke on Yann LeCun` |
| `obama` | `Place the coke on Barack Obama` |
| full sentence | normalized to canonical form if needed |
| `home` | return arm to pose captured at connect |
| `quit` | disconnect and exit |

Each rollout runs **20 s** (default), logs to `outputs/eval3_rollouts/rollout_<UTC>.jsonl`, then returns home automatically.

## Change the model (one env var)

```bash
export EVAL3_DEMO_POLICY=RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k
./scripts/run_eval3_demo_cli.sh
```

Later swap:

```bash
export EVAL3_DEMO_POLICY=RobotLearningVLA/eval3-vla-v16-real-synth-50k-step50000
EVAL3_DEMO_PRESET=v16 ./scripts/run_eval3_demo_cli.sh
```

`EVAL3_DEMO_PRESET` sets dataset schema + camera rename defaults; `EVAL3_DEMO_POLICY` overrides the Hub path only (train_config is read when possible for v16 two-camera mode).

Presets: `v4slots_expert` (default), `v4slots_full`, `v6_new`, `v16`.

## Pre-flight without hardware

```bash
./scripts/run_eval3_demo_cli.sh --dry-run
```

## Single rollout (non-interactive)

```bash
./scripts/run_eval3_demo_cli.sh --once yann
```

## Knobs

| Env var | Default | Meaning |
|---------|---------|---------|
| `EVAL3_DEMO_POLICY` | v4slots expert Hub id | Checkpoint |
| `EVAL3_DEMO_PRESET` | `v4slots_expert` | Schema / rename recipe |
| `FOLLOWER_TTY` | `/dev/ttyUSB0` | SO-101 serial port |
| `CAM_IDX` | `0` | OpenCV camera index |
| `EVAL3_POLICY_DEVICE` | `auto` | `cuda` on Ubuntu when available |
| `EVAL3_EPISODE_TIME_S` | `20` | Rollout length (seconds) |
| `EVAL3_FPS` | `30` | Control loop rate |

## Before demo

1. Motor bus **7–12 V** supply ON.
2. `lerobot-find-port` if unsure of `FOLLOWER_TTY`.
3. `HF_TOKEN` logged in (private Hub checkpoints).
4. Run `--dry-run` once after any model swap.

## Files

- `scripts/eval3_demo_cli.py` — interactive loop
- `scripts/run_eval3_demo_cli.sh` — env defaults + wrapper

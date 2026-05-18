# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this repo is

A thin sandbox around the upstream
[huggingface/lerobot](https://github.com/huggingface/lerobot) library for the
ETH "Robot Learning" project. The runtime `lerobot` package is installed
from PyPI via `uv pip install` into `./.venv/`. There is **no** local
checkout of lerobot to edit — treat `lerobot` as a third-party dependency.

Local platform target: **macOS 14 / Apple Silicon / Python 3.12** (no CUDA),
so `DEVICE=cpu` is the practical default for any training/E2E runs. Linux
hosts work too.

## Files

- `install.sh` — idempotent bootstrap. Installs `uv` if missing, creates
  `.venv` on Python 3.12, runs `uv pip install ${LEROBOT_SPEC:-lerobot}`,
  optionally logs into Hugging Face if `HF_TOKEN` is set, scaffolds
  calibration directories under `~/.cache/huggingface/lerobot/calibration/`.
- `README.md` — user-facing setup, calibration, teleop, record, replay, and
  HF dataset docs. Keep in sync if behavior changes.
- `pyproject.toml` — minimal metadata only (`requires-python`, name). The
  install is *imperative* (`uv pip install`), not declarative — `dependencies`
  is intentionally empty. Do not switch to `uv sync` without re-validating
  on macOS 14 (some lerobot extras have no wheels there; see below).
- `camera.py` — small OpenCV live-preview script using
  `lerobot.cameras.opencv.OpenCVCamera`. Useful for verifying a camera before
  running `lerobot-record`.
- `.venv/` — uv-managed virtualenv; do not commit.

## Quick orientation

```bash
./install.sh                                # one-time bootstrap (idempotent)
source .venv/bin/activate                   # or prefix commands with `uv run`
lerobot-find-port                           # discover motor-adapter port
lerobot-find-cameras opencv                 # enumerate cameras
lerobot-calibrate --teleop.type=so101_leader --teleop.port=... --teleop.id=my_awesome_leader_arm
lerobot-teleoperate ...
lerobot-record ...
lerobot-replay ...
```

Calibration IDs already in use on the typical setup:
`my_awesome_leader_arm` and `my_awesome_follower_arm`. Reuse those exact
strings on every `lerobot-*` command, or generate new calibration files with
different ids.

## Eval 3 VLA notes

Eval 3 training/deploy code lives in `scripts/eval3_*` and `tools/eval3_*`.
The current retrain wrapper is `scripts/run_eval3_smolvla_aug_train.sh`; it
defaults to the v8 data recipe with dataset_v2 gripper-open label repair and
arm-label smoothing. Deployment should use single-camera SmolVLA compatibility
(`front` -> `camera1`, `policy.empty_cameras=2`) plus the guarded smoothing
flags documented in `docs/eval3/abcd_model_eval.md`.

## Platform gotchas

- **`lerobot[all]` does NOT install on macOS 14 + Python 3.12.** Two
  sub-extras have no compatible wheel: `intelrealsense`
  (`pyrealsense2-macosx ≥ 2.56` needs macOS 15) and `unitree_g1`
  (`onnxruntime 1.26` needs Python 3.13 on macOS-arm). The default
  `LEROBOT_SPEC=lerobot` sidesteps both. Add extras à la carte if you need
  them.
- **No GPU on macOS.** Anything that defaults to CUDA will be slow or fail.
  Use `DEVICE=cpu` (the default in most lerobot configs).
- **`lerobot-record` performance**: on Apple Silicon the record loop often
  runs below the target 30 Hz. Biggest cost is `--display_data=true` (rerun
  streaming). Drop it, lower fps, or lower camera resolution if frames are
  being dropped.
- **macOS Accessibility**: `lerobot-record` uses `pynput` to capture
  arrow-key shortcuts (`→` next, `←` redo, `Esc` stop). On macOS this
  requires granting the terminal Accessibility permission (System Settings →
  Privacy & Security → Accessibility). Without it, keys leak through as
  raw escape codes (`^[[C` etc.) and the shortcuts may not register.

## Hugging Face dataset workflow

Datasets live under `RobotLearningVLA/<name>` on the Hub (see README for
inventory). `lerobot-record` pushes automatically at session end. **Every
new dataset needs a Hub git tag matching `meta/info.json:codebase_version`
(currently `v3.0`) before it can be replayed/trained on** — otherwise
loading fails with `RevisionNotFoundError`, masked by a confusing
`HfHubHTTPError missing 'response'` error on `huggingface_hub` 1.x.

Tag after pushing:

```python
uv run python -c "from huggingface_hub import HfApi; HfApi().create_tag('<org>/<name>', tag='v3.0', repo_type='dataset')"
```

## Common failure modes (quick diagnosis)

| Symptom | Likely cause |
|---|---|
| `FeetechMotorsBus motor check failed … found motor list: {}` | external 7–12 V supply to the motor bus is off, or leader/follower USB ports are swapped. USB alone does not power servos. |
| `FileExistsError … .cache/huggingface/lerobot/<org>/<name>` | a previous failed `lerobot-record` left an empty cache dir. Either pick a new `--dataset.repo_id` or `rm -rf` the offending dir. |
| `RevisionNotFoundError` / `HfHubHTTPError missing 'response'` on replay | the target dataset has no `v3.0` git tag on the Hub. See section above. |
| Record loop runs at < 30 Hz | drop `--display_data=true`, lower fps, or reduce camera resolution. |

## When editing this repo

- Keep `install.sh` idempotent and re-runnable.
- Keep `README.md` and `AGENTS.md` in sync — the README is for humans, this
  file is for the agent. Don't duplicate; cross-reference.
- Don't add lerobot source-level changes here — fork
  [huggingface/lerobot](https://github.com/huggingface/lerobot) for that.

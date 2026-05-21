# Eval 3 — VLA robot: "Place the coke on `<celebrity>`"

> **Course:** ETH Robot Learning · **Hardware:** SO-101 follower arm · **Policy:** fine-tuned [SmolVLA](https://huggingface.co/lerobot/smolvla_base)

This repository is the team's **complete Eval 3 submission**: a Vision-Language-Action
(VLA) policy that drives an SO-101 robot arm to **pick up a Coke can and place it on
the printed photo of the celebrity named in the language prompt**.

It also doubles as the team's `lerobot` hardware sandbox (teleoperation, calibration,
recording, replay). That reference material is kept in
[Part 2 — Hardware & lerobot sandbox](#part-2--hardware--lerobot-sandbox-reference).

**Teammate submitting the form?** Start with **[TEAMMATE_SUBMIT.md](TEAMMATE_SUBMIT.md)** and **[HANDOFF_READY.txt](HANDOFF_READY.txt)**.

## Course submission (Project 1 — VLA)

**Deadline:** Friday 22.05.2026 23:59. See **[docs/PROJECT_SUBMISSION.md](docs/PROJECT_SUBMISSION.md)** for the full checklist, Azure upload commands, and video requirements.

| Run script | Task |
|------------|------|
| [`run_eval3_in_distribution.sh`](run_eval3_in_distribution.sh) | Taylor Swift, Yann LeCun, Barack Obama — `v4slots_expert` |
| [`run_eval3_ood.sh`](run_eval3_ood.sh) | OOD celebrities — `eval3-smolvla-v16-pinsv5-step5k` |

Form summary (under 300 words): paste from [`SUBMISSION_SUMMARY.txt`](SUBMISSION_SUMMARY.txt).

```bash
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
export FOLLOWER_TTY=/dev/ttyACM0   # or Mac usbmodem port
./run_eval3_in_distribution.sh "Place the coke on Taylor Swift"
```

---

## For graders — at a glance

| | |
|---|---|
| **Task** | Given the instruction *"Place the coke on `<celebrity>`"*, the arm grasps a Coke can and places it on the correct printed celebrity photo (one of three slots: left / middle / right). |
| **Policy** | SmolVLA, fine-tuned on the team's **real SO-101 teleoperation data** (`dataset_v4_*`). The vision encoder + VLM language tower are **frozen** — only the action expert (plus a small slot head for v16) is trained. |
| **Final models** | Two checkpoints on the Hugging Face Hub under `RobotLearningVLA/` — see [Final models](#final-models). |
| **Run it** | One command — see [Deploy](#deploy-run-the-policy-on-the-arm). |
| **Reproduce training** | See [Reproduce training](#reproduce-training). |
| **Status** | ✅ Complete. `v16` is the deployed model. |

---

## The task

The workspace has three printed celebrity photos laid out left, middle, and right, and
one Coke can. The policy receives a single RGB camera stream, the arm's joint state, and
a natural-language instruction such as `"Place the coke on Barack Obama"`. It must grasp
the can and release it on top of the **named** celebrity's photo — so the policy has to
actually *bind the language prompt to the right image region*, not just repeat one
motion.

The three "known" (in-distribution) identities are **Taylor Swift, Barack Obama, and
Yann LeCun**. Out-of-distribution celebrities are also tested — see
[Celebrity print sheets](#celebrity-print-sheets).

## Approach

We fine-tune **SmolVLA** on **real teleoperated SO-101 data only** (the `dataset_v4_*`
corpus — 9 datasets, one per celebrity × slot position). The **vision encoder and VLM
language tower are frozen**; we train only SmolVLA's action expert ("expert-only"
fine-tuning, ~101 M of 452 M parameters). Freezing the perception stack keeps the
pretrained celebrity knowledge intact and makes the fine-tune stable on a small robot
dataset.

### Final models

Both models are on the Hugging Face Hub under `RobotLearningVLA/` and are wired into the
deploy battery (`scripts/run_eval3_deploy_battery.sh`).

| Model (HF repo) | What it is | Deploy entry |
|---|---|---|
| `eval3-vla-v6-smolvla-fresh-v4slots-expert-50k` | SmolVLA fine-tuned on the 9 real `dataset_v4_*` slot datasets, expert-only (frozen encoder), fresh from `lerobot/smolvla_base`, 50k steps. The baseline frozen-encoder VLA. | `run_eval3_deploy_battery.sh v4slots_expert` |
| `eval3-smolvla-v16-pinsv5-step5k` | **The deployed model.** v16 *slot-bottleneck* variant — same frozen-encoder recipe plus an architecture change that forces the policy to read the target from the **language prompt**. | `run_eval3_deploy_battery.sh v16` |

### Why v16 (the slot bottleneck)

Earlier checkpoints scored well *offline* but, on the real arm, tended to **ignore the
celebrity name** in the prompt. They had learned a shortcut: read the target from the
*can's motion* after grasping it, rather than from the language. Cross-prompt action
differences were < 1° on training frames — the prompt was effectively unused.

**v16** removes that shortcut. A small slot classifier reads a **frozen frame-0 image**
(captured at the start of the episode, fed through a dedicated second camera input) plus
the language tokens, and commits to one of the three target slots for the whole episode.
Because the frame-0 scene is static and the same 3-face layout appears with all three
prompts, the only way the classifier can lower its loss is to actually use the language.
A `LayerNorm` fix on the classifier inputs was needed to stop its attention softmax from
saturating. Full write-up: **[docs/eval3/v16_playbook.md](docs/eval3/v16_playbook.md)**.

---

## Deploy: run the policy on the arm

### Hardware prerequisites

- SO-101 **follower** arm, calibrated — see [Calibration](#calibration).
- External **7–12 V** supply to the motor bus, powered on. *USB alone does not power the
  servos.*
- One camera pointed at the workspace.
- The celebrity photos printed and placed in the left / middle / right slots — see
  [Celebrity print sheets](#celebrity-print-sheets).

### One-time setup on the deploy machine

```bash
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh        # venv + lerobot + SmolVLA deps
hf auth login --token <YOUR_TOKEN>               # needs read access to RobotLearningVLA/
lerobot-find-port                                # discover the follower TTY port
```

### Deploy command (final v16 model)

**macOS** (the rig defaults — follower TTY + camera index — are baked in):

```bash
./scripts/run_eval3_deploy_battery.sh v16 --task='Place the coke on Taylor Swift'
```

**Linux / CUDA** — override the port, camera index, and device:

```bash
FOLLOWER_TTY=/dev/ttyACM0 \
CAM_IDX=0 \
./scripts/run_eval3_deploy_battery.sh v16 \
  --task='Place the coke on Taylor Swift' \
  --episode_time_s=40 --fps=40 \
  --policy.device=cuda
```

Notes:

- The `v16` entry **defaults to the Hub model** `eval3-smolvla-v16-pinsv5-step5k` — no
  `EVAL3_V16_CKPT` needed. Set `EVAL3_V16_CKPT=<path-or-repo>` only to deploy a different
  checkpoint.
- `FOLLOWER_TTY` / `CAM_IDX` default to the macOS dev rig — **override them on any other
  machine**. Get the port from `lerobot-find-port`, the camera from
  `lerobot-find-cameras opencv`.
- On Linux you **must** append `--policy.device=cuda` — the battery hardcodes `mps`
  (Apple Silicon), which does not exist on Linux. It is appended last, so it wins.
- The `v16` entry sets the two-camera `rename_map` + `policy.empty_cameras=1` itself
  (v16 uses a second camera input for the frame-0 image) — you do **not** add those.
- Add `--display_data=true` to open a live Rerun viewer of the camera + state.
- On Linux, give the user serial-port access once: `sudo usermod -aG dialout $USER`
  (then re-login).

### Pre-flight check (run before plugging in the arm)

```bash
python tools/eval3_check_deploy_command.py \
  --policy-pretrained-path RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k \
  --rename-map '{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}' \
  --task 'Place the coke on Taylor Swift'
```

This validates the `rename_map`, camera keys, and task string. To load the checkpoint
without driving hardware, append `--dry_run=true` to the deploy command.

### Celebrity print sheets

Print these in color and cut out the photos to set up the scene.

| File | Contents |
|---|---|
| `in-distribution-eval-3.pdf` | Course-provided. The 3 in-distribution identities (Taylor Swift, Barack Obama, Yann LeCun), 15 photos. |
| `out-distribution-eval-3-pins.pdf` | 10 out-of-distribution celebrities from the PINS dataset, one photo each, 2 per A4 page (~A5, sized to cut out). Built by `tools/eval3_build_ood_pins_pdf.py`. |
| `datasets/out-distribution-eval-3/` | Wikimedia-sourced OOD portraits of the 3 in-distribution identities (different shoots / years). |

---

## Reproduce training

Full SmolVLA fine-tunes need a CUDA GPU; see [docs/eval3/compute_budget.md](docs/eval3/compute_budget.md).

```bash
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh

# Baseline frozen-encoder VLA  ->  eval3-vla-v6-smolvla-fresh-v4slots-expert-50k
./scripts/run_eval3_smolvla_v4slots_train.sh expert

# v16 slot-bottleneck (final deployed recipe)  ->  eval3-smolvla-v16-...
./scripts/run_eval3_smolvla_v16_real_data_slot_train.sh
```

Each launcher is a thin wrapper that exports env vars and calls
`scripts/train_eval3_smolvla.py` (the `lerobot-train` CLI plus two import-time shims).
Read each launcher's header comment for the exact corpus and knobs.

Quick sanity checks (no GPU, no hardware):

```bash
python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/dataset_v4_taylor_left
python tools/eval3_smolvla_compat.py --repo-id RobotLearningVLA/dataset_v4_taylor_left
```

## Repository map (Eval 3)

- `scripts/run_eval3_deploy_battery.sh` — **the deploy entry point.** Named checkpoint
  shortcuts (`v16`, `v4slots_expert`, …), bakes in the follower port + camera + deploy
  guards. Run with `--help` for the full checkpoint list.
- `scripts/eval3_vla_deploy.py` — closed-loop SmolVLA on the real SO-101.
- `scripts/train_eval3_smolvla.py` — training entry (`lerobot-train` + import shims).
- `scripts/run_eval3_smolvla_*train*.sh` — training launchers (one recipe each).
- `tools/eval3_check_deploy_command.py` — pre-flight validator. Run before every deploy.
- `tools/eval3_build_ood_pins_pdf.py` — builds `out-distribution-eval-3-pins.pdf`.
- `tools/eval3_*.py` — offline audits, dataset inspectors, augmentation builders. None
  touch hardware.
- `docs/eval3/` — design and runbook docs. Highest-value:
  [v16_playbook.md](docs/eval3/v16_playbook.md) (the final model),
  [friend_deploy_handoff.md](docs/eval3/friend_deploy_handoff.md) (deploy recipe),
  [hardware_eval_matrix.md](docs/eval3/hardware_eval_matrix.md) (trial protocol),
  [tensor_contract.md](docs/eval3/tensor_contract.md) /
  [prompt_protocol.md](docs/eval3/prompt_protocol.md) /
  [scene_spec.md](docs/eval3/scene_spec.md) (train/deploy invariants).
- `CLAUDE.md` — engineering notes for the agent working on this repo (load-order
  gotchas, the deploy battery, augmentation env-var contract).

---

# Part 2 — Hardware & lerobot sandbox reference

The rest of this file is the operational reference for the SO-101 hardware and the
`lerobot` toolchain — calibration, teleoperation, recording, replay, and the Hugging
Face dataset workflow. It applies whether or not you are running the Eval 3 policy.

This is a `uv`-managed Python environment for the
[huggingface/lerobot](https://github.com/huggingface/lerobot) library. `lerobot` is
installed from PyPI into a local `.venv/` via `uv pip install`. Works on macOS / Apple
Silicon (no CUDA) and on Linux.

## Quickstart

```bash
./install.sh
```

The script is idempotent — re-run it any time to reconcile dependencies.

### What `install.sh` does

1. Installs [`uv`](https://docs.astral.sh/uv/) if it is not on `PATH`.
2. Creates `./.venv/` on the requested Python (`uv venv --python 3.12`), or reuses it.
3. Runs `uv pip install lerobot` into that venv (the install path verified on macOS 14 /
   Apple Silicon — the declarative `uv sync` path fails there, see
   [Platform caveats](#platform-caveats)).
4. If `HF_TOKEN` is set, runs `hf auth login`.
5. Ensures the calibration directories under
   `~/.cache/huggingface/lerobot/calibration/` exist.

The install is **imperative, not declarative** — re-running reinstalls against the
latest PyPI `lerobot`. It does not write `pyproject.toml` or `uv.lock`.

### Configuration

Override via env vars before running the script:

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | Python minor version pin. Stay on `3.12` on macOS 14 — see [Platform caveats](#platform-caveats). |
| `LEROBOT_SPEC` | `lerobot` | Package spec passed to `uv pip install`. Add extras here. |
| `HF_TOKEN` | unset | If set, the script logs into Hugging Face automatically. |
| `EVAL3_INSTALL_SMOLVLA_DEPS` | unset | When `=1`, also installs `transformers accelerate sentencepiece num2words`. **Required for the Eval 3 pipeline.** |

If an extras install fails (the macOS 14 culprits are `intelrealsense` and
`unitree_g1`), fall back to plain `LEROBOT_SPEC="lerobot"`.

### Verifying the install

```bash
source .venv/bin/activate
lerobot-find-port              # confirms motor adapters enumerate
lerobot-find-cameras opencv    # confirms cameras enumerate
hf auth whoami                 # confirms Hugging Face login
```

## Hugging Face authentication

```bash
hf auth login --token <YOUR_TOKEN> --add-to-git-credential
```

Get a token at <https://huggingface.co/settings/tokens>. For pushing to the
`RobotLearningVLA` org, use a fine-grained token scoped to that org with read+write on
repos/datasets.

## Calibration

Calibration stores the homing offset and joint range of every motor on each arm, in raw
STS-3215 encoder counts (0–4095). `lerobot-calibrate` writes JSON to:

```
~/.cache/huggingface/lerobot/calibration/
  teleoperators/so_leader/my_awesome_leader_arm.json     # leader arm
  robots/so_follower/my_awesome_follower_arm.json        # follower arm
```

The basenames (`my_awesome_leader_arm`, `my_awesome_follower_arm`) are the `--teleop.id`
/ `--robot.id` strings — pass the **exact same ids** to every later `lerobot-teleoperate`
/ `lerobot-record` / `lerobot-replay` call so the matching file is picked up. The
`so_leader` / `so_follower` directory layout is fixed by lerobot — do not rename it.

Each JSON maps motor name → encoder calibration:

```json
{
    "shoulder_pan":  { "id": 1, "drive_mode": 0, "homing_offset":  -872, "range_min": 1108, "range_max": 3041 },
    "wrist_roll":    { "id": 5, "drive_mode": 0, "homing_offset":  1795, "range_min":    0, "range_max": 4095 },
    "gripper":       { "id": 6, "drive_mode": 0, "homing_offset": -1093, "range_min": 1892, "range_max": 3091 }
}
```

`wrist_roll` is the full-turn joint, so its range stays 0–4095; the other five are
bounded by their physical travel during calibration.

### Generating fresh calibrations

```bash
lerobot-find-port            # find each USB-serial port (unplug when prompted)

lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodemXXXXXX \
    --teleop.id=my_awesome_leader_arm

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodemYYYYYY \
    --robot.id=my_awesome_follower_arm
```

The script walks you through homing each joint at its midpoint and moving it through its
full range. Override the calibration root with `HF_LEROBOT_CALIBRATION=/some/dir`.

## Teleoperation

`lerobot-teleoperate` streams positions from the leader arm to the follower arm in real
time, optionally opening a [Rerun](https://www.rerun.io/) viewer.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B140317761 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B141136041 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
```

- `--robot.cameras` takes a draccus-style dict; add keys for more cameras.
- `--display_data=false` skips the Rerun viewer (saves a lot of CPU on Apple Silicon).
- Ports are per-laptop — read them from `lerobot-find-port`, don't copy blindly.

## Recording a dataset

`lerobot-record` runs the teleop loop, captures every frame of state + action + cameras,
and (by default) uploads the result to the Hub at session end.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B140317761 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B141136041 \
    --teleop.id=my_awesome_leader_arm \
    --dataset.repo_id=RobotLearningVLA/<name> \
    --dataset.num_episodes=20 \
    --dataset.single_task="Place the coke on Taylor Swift" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec=h264_videotoolbox
```

| Flag | Meaning |
|---|---|
| `--dataset.repo_id` | Target `<org-or-user>/<name>` on the Hub. When a `--policy` is set (eval rollouts), `<name>` must start with `eval_`. |
| `--dataset.num_episodes` | Episodes to record this session. |
| `--dataset.single_task` | Language label attached to every frame (used by VLA policies). |
| `--dataset.episode_time_s` / `--dataset.reset_time_s` | (Default 60 each) max seconds per episode / scene reset; `→` ends early. |
| `--dataset.streaming_encoding` | Encode videos in a background thread (recommended). |
| `--dataset.vcodec` | `h264_videotoolbox` on macOS; `libsvtav1` / `libx264` otherwise. |
| `--dataset.push_to_hub` | (Default `true`) upload at end. `false` keeps it local-only. |
| `--resume` | **Top-level**, not under `--dataset.`. `true` appends to an existing dataset. |
| `--dataset.root` | Required with `--resume=true`; a writable dir **outside** `~/.cache/huggingface/lerobot/`. |

Keyboard shortcuts while recording: `→` advance phase, `←` re-record episode, `Esc`
stop. On macOS, grant the terminal **Accessibility** permission (System Settings →
Privacy & Security → Accessibility), or the keys leak through as raw escape codes.

## Replaying a dataset

`lerobot-replay` reads a dataset from the Hub and drives the follower arm through the
recorded actions — a quick check that calibration + hardware still match.

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B140317761 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=RobotLearningVLA/dataset_v4_taylor_left \
    --dataset.episode=0
```

## Hugging Face datasets

lerobot datasets live at `https://huggingface.co/datasets/<org-or-user>/<name>` and cache
locally to `~/.cache/huggingface/lerobot/<org-or-user>/<name>/`. The Eval 3 active corpus
is `RobotLearningVLA/dataset_v4_{taylor,yann,barack}_{left,middle,right}`.

Pull a dataset without driving hardware:

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('RobotLearningVLA/dataset_v4_taylor_left')
print(ds.num_episodes, 'episodes,', ds.num_frames, 'frames')
"
```

### The version-tag requirement

When loading a dataset, lerobot looks for a Hub git **tag** matching the dataset's
`meta/info.json:codebase_version` (currently `v3.0`). A missing tag fails with
`RevisionNotFoundError`, masked by a confusing `HfHubHTTPError missing 'response'` on
`huggingface_hub` 1.x. Tag once per dataset after pushing:

```bash
uv run python -c "from huggingface_hub import HfApi; HfApi().create_tag('<org>/<name>', tag='v3.0', repo_type='dataset')"
```

## Platform caveats

`lerobot[all]` will **not** resolve on macOS 14 + Python 3.12 — two sub-extras lack
compatible wheels:

| Sub-extra | Blocking package | Why |
|---|---|---|
| `intelrealsense` | `pyrealsense2-macosx>=2.56` | only ships a `macosx_15` wheel — needs macOS 15 |
| `unitree_g1` | `onnxruntime==1.26.0` | only ships a `cp313` macOS-arm wheel — needs Python 3.13 |

The default `LEROBOT_SPEC=lerobot` sidesteps both. Add extras à la carte if you need
them. Full SmolVLA fine-tunes belong on Linux + CUDA; MPS on Apple Silicon is fine for
inference / smoke runs.

## Other useful CLIs

```bash
lerobot-find-port            # discover motor-adapter port (unplug when prompted)
lerobot-find-cameras opencv  # enumerate cameras + save preview frames
lerobot-setup-motors …       # assign motor IDs 1–6 to a fresh Feetech daisy chain
lerobot-info                 # print env / package versions
```

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `FeetechMotorsBus motor check failed … found motor list: {}` | External 7–12 V supply to the motor bus is off, or leader/follower ports are swapped. USB alone does not power the servos. |
| `FileExistsError … .cache/huggingface/lerobot/<org>/<name>` | A previous failed `lerobot-record` left an empty cache dir. Pick a new `--dataset.repo_id` or `rm -rf` the dir. |
| `RevisionNotFoundError` / `HfHubHTTPError missing 'response'` | The dataset has no `v3.0` git tag — see [the version-tag requirement](#the-version-tag-requirement). |
| Record / deploy loop runs at < 30 Hz | Drop `--display_data=true`, lower fps, or reduce camera resolution. On Apple Silicon, run the policy on MPS. |
| `non-default argument 'backbone_cfg' follows default argument` at import | `lerobot.policies` imported before the shim. Enter via `scripts/train_eval3_smolvla.py` or `scripts/eval3_vla_deploy.py`, never `lerobot-train` directly. |
| `ImportError: Package 'num2words' is required` | SmolVLA extras not installed: `EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh`. |
| Deploy runs but the policy "ignores the celebrity" / grasps the wrong slot | Almost always a missing `--rename_map` — the policy sees black camera frames. Run `tools/eval3_check_deploy_command.py` first. |

---

`CLAUDE.md` holds the detailed engineering notes (Eval 3 pipeline architecture,
load-order gotchas, the deploy battery, the augmentation env-var contract) for anyone —
human or agent — iterating on this repo.

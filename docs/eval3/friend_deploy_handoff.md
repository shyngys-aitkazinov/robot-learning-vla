# Eval 3 — Friend deploy handoff (SmolVLA 50k 3-celebrity model on SO-101)

This document is a self-contained recipe for taking the **`RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh`** checkpoint and running it on an SO-101 follower arm. Follow it top to bottom on the machine that has the robot plugged in. The whole thing is one-time setup + one command to run a rollout.

> ⚠️ **Critical — read this before composing any deploy command** ⚠️
>
> The single most common Eval 3 deploy failure is forgetting the
> `--rename_map` (custom deploy script) / `--dataset.rename_map`
> (`lerobot-record`) flag. Without it, the SmolVLA policy expects
> `observation.images.camera1` but the robot provides
> `observation.images.front` — they never alias and the policy receives
> zero-padded black frames for every camera key. The result is
> "policy doesn't recognise celebrities" AND "grabbing policy is bad".
>
> Validate any deploy command before plugging in the arm with:
>
> ```bash
> python tools/eval3_check_deploy_command.py \
>     --policy-pretrained-path <CHECKPOINT> \
>     --rename-map '{"observation.images.front":"observation.images.camera1"}' \
>     --task "Place the coke on Taylor Swift"
> ```
>
> You want `PASS  (cameras=OK, task=OK)` on the final line. If it FAILs,
> the script prints the corrected command line — copy it verbatim.
>
> For a post-failure remediation walkthrough see
> [`v7_deploy_checklist.md`](v7_deploy_checklist.md).

The model was fine-tuned from `lerobot/smolvla_base` on 44 filtered episodes / 25 553 frames across `taylor_swift_1`, `yann_lecun_1`, and `barack_obama_1` with the full v3 augmentation stack: 10 torchvision image transforms + background replacement (p=0.3) + target-preserving print-position shuffle (p=0.5) + task-string augmentation. 14/58 source episodes were dropped: Swift 6 bad-recording episodes + LeCun/Obama 8 negative-mode wrist_roll episodes (operator inconsistency, both modes physically valid but only one mode kept for clean supervision). Architecture is SmolVLA (≈450 M params, ≈907 MB on disk).

### Behavior changes vs the previous `eval3-smolvla-3way-50k-aug-v1` checkpoint

The previous v1 model collapsed all three prompts to Swift's wrist_roll (≈ −80°) regardless of which celebrity was named. v3 has been retrained from scratch with augmentation that breaks this scene-prompt shortcut. Expected per-prompt `wrist_roll` (action index 4) final-1s behavior:

| prompt | v1 (broken) wrist_roll | v3 (new) expected wrist_roll |
|---|---|---|
| Place the coke on Taylor Swift | −80° to −85° (correct) | similar (−80° to −85°) |
| Place the coke on Yann LeCun | −85° (WRONG, collapsed to Swift) | **+85° to +100°** (180° rotation — gripper visibly flips) |
| Place the coke on Barack Obama | −82° to −92° (WRONG, collapsed to Swift) | **+80° to +95°** (170° rotation) |

So the most visible difference will be that **the gripper orientation is now distinct between Swift and LeCun/Obama rollouts**. After running the 3 prompts, dump the recorded JSONLs and check the final-1s wrist_roll to confirm.

---

## 0. Prerequisites

- **OS**: macOS 14+ (Apple Silicon) or Linux. CUDA optional; if no GPU we fall back to MPS (Apple) or CPU (slow but works).
- **Python**: 3.12 (the install script enforces this via `uv`).
- **Hardware**: SO-101 follower arm + USB serial + one camera (OpenCV-compatible). Same physical arm as the recording rig is best; a different SO-101 with its own calibration also works.
- **Power**: external 7–12 V supply to the motor bus **MUST** be on. USB alone does not power the servos. Without external power, every motor command will silently fail.
- **HF account** with read access to the private `RobotLearningVLA` org (ask Rakhmatillo to invite you at https://huggingface.co/organizations/RobotLearningVLA → Members).

---

## 1. One-time setup

### 1.1 Clone the repo

```bash
git clone https://github.com/shyngys-aitkazinov/robot-learning-vla.git
cd robot-learning-vla
git checkout Tillo
```

### 1.2 Install the Python environment

```bash
./install.sh                              # creates .venv, installs lerobot
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh # adds transformers + accelerate + sentencepiece + num2words
```

Both runs are idempotent and re-runnable. If anything fails, re-run them.

### 1.3 SO-101 motor SDK (required, not in base `lerobot`)

```bash
source .venv/bin/activate
uv pip install 'lerobot[feetech]'
python -c "import scservo_sdk; print('motor SDK ok')"
```

### 1.4 Hugging Face auth

You need a token with **read** access to the `RobotLearningVLA` org. Create one at https://huggingface.co/settings/tokens (fine-grained scope works — pick "read" on `RobotLearningVLA`).

```bash
uv run hf auth login --token <YOUR_HF_TOKEN>
uv run hf auth whoami         # should print your username and 'RobotLearningVLA' under orgs
```

If `whoami` doesn't list `RobotLearningVLA`, message Rakhmatillo to invite your HF username to the org.

### 1.5 Robot calibration

Two cases:

**(A) Same physical SO-101 we used to record.** Rakhmatillo will send you `my_awesome_follower_arm.json`. Drop it at:

```bash
mkdir -p ~/.cache/huggingface/lerobot/calibration/robots/so_follower
cp /path/to/my_awesome_follower_arm.json \
   ~/.cache/huggingface/lerobot/calibration/robots/so_follower/
```

In the deploy command later, use `--robot.id=my_awesome_follower_arm`.

**(B) Different SO-101 arm.** Calibrate it once:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=<your_follower_tty> \
  --robot.id=<your_choice_of_id>
```

In the deploy command later, substitute your chosen ID for `<your_calib_id>`.

---

## 2. Hardware discovery (≈2 min)

### 2.1 Find the USB port the follower is on

```bash
lerobot-find-port
```

Unplug the FOLLOWER's USB when prompted, plug it back. It'll report the device path. **Verify this is the follower and not the leader** — the script can mistake them. You can confirm by replaying an episode (Step 3) and watching which arm moves.

### 2.2 Find the camera index

```bash
lerobot-find-cameras opencv
```

This lists every camera the OS sees. On macOS you'll typically see:

- Index 0 = laptop FaceTime camera
- Index 1 = first external USB camera

**You want the camera pointed at the table.** When in doubt, run `tools/eval3_camera_check.py --camera-index 0` (and try 1) and visually compare the captured frame to `outputs/eval3_deep_analysis_v2/sample_frames/swift/ep00_approach_frame76.png` — the right camera should show a similar table + Coke can + 3 celebrity prints layout.

```bash
python tools/eval3_camera_check.py --camera-index 0
open outputs/eval3_camera_check          # macOS; on Linux use xdg-open or your file manager
```

Note the index that shows the table. You'll use it as `<cam_idx>` below.

---

## 3. Sanity replay (verifies calibration + USB + power, NO policy involved)

This sends pre-recorded joint positions straight to the motors. If this doesn't work, nothing else will. Power must be on, motor IDs must match the calibration, the bus must enumerate.

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=<follower_tty> \
  --robot.id=my_awesome_follower_arm   # or <your_calib_id> for case B above
  --dataset.repo_id=RobotLearningVLA/taylor_swift_1 \
  --dataset.episode=0
```

You'll see a prompt about a calibration mismatch on first run — **press Enter** to use the provided calibration file. Don't type `c` (that would recalibrate and invalidate the trained policy's joint mapping).

Pass criteria: the follower arm smoothly executes a teleop trajectory for ~20–25 seconds. If it jerks or doesn't move, fix the hardware before touching the policy.

---

## 4. Live VLA rollout (the actual deploy)

The first run will download the 907 MB checkpoint from HF into `~/.cache/huggingface/hub/`. After that it's cached.

### 4.1 Run the deploy command

```bash
python scripts/eval3_vla_deploy.py \
  --robot.type=so101_follower \
  --robot.port=<follower_tty> \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{front: {type: opencv, index_or_path: <cam_idx>, width: 640, height: 480, fps: 30}}' \
  --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --policy.path=RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh \
  --policy.device=mps \
  --policy.n_action_steps=25 \
  --interpolation_multiplier=2 \
  --episode_time_s=20 \
  --fps=30
```

Adjust:

- `<follower_tty>` → the port from Step 2.1 (e.g. `/dev/tty.usbmodem5B140317761` on macOS, `/dev/ttyACM0` on Linux).
- `<cam_idx>` → the camera index from Step 2.2.
- `--policy.device=mps` → use `cuda` if you have an NVIDIA GPU, `cpu` if neither.
- `my_awesome_follower_arm` → your calibration ID if Case B in Step 1.5.

### 4.2 Operating the rollout

After the model loads (~10–15 s the first time, ~3 s after caching), you'll see on **stderr**:

```
Enter Eval 3 instruction (e.g. 'Place the coke on Taylor Swift'), then press Enter:
```

Type **one** of these three exactly (case sensitive) and press Enter:

- `Place the coke on Taylor Swift`
- `Place the coke on Yann LeCun`
- `Place the coke on Barack Obama`

The 20-second control loop starts immediately. The arm should reach toward whichever celebrity print matches the prompt.

**Aborting**: press `Esc` to stop early (requires macOS Accessibility permission for your terminal in System Settings → Privacy & Security → Accessibility). Or Ctrl-C from the terminal.

### 4.3 Each rollout writes two artifacts

- `outputs/eval3_rollouts/rollout_<UTC_TIMESTAMP>.jsonl` — header line with metadata + ~600 action records (one per tick).
- `outputs/eval3_rollouts/rollout_<UTC_TIMESTAMP>.firstframe.png` — the first camera frame the policy saw. Useful for confirming the camera is pointed at the table.

**Please send these back to Rakhmatillo after each rollout**, plus a video of the arm motion if you can take one on your phone. Even one prompt's result is enough to start triaging.

---

## 5. What we're trying to learn from your test

We've never live-tested this model. Tell us which of the four outcomes you observed:

- **Outcome A** — robot reaches the correct print on **2 of 3** prompts. Means TOY scenario works on real hardware. Focus shifts to ID-holdout (different photos of same celebs).
- **Outcome B** — robot moves coherently but always misses (e.g., always reaches middle of arc). Means generalization problem — the bench scene differs from training. We'll iterate on scene matching, not the model.
- **Outcome C** — robot wanders / never commits to any print. Means something is wrong end-to-end. Most likely camera index mismatch, USB port mismatch, or calibration drift.
- **Outcome D** — robot doesn't move at all / errors out. Means setup problem (motor bus power off, port wrong, HF auth missing).

---

## 6. Common troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'scservo_sdk'` | Step 1.3 not done | `uv pip install 'lerobot[feetech]'` |
| `ImportError: Package 'num2words' is required …` | Step 1.2's `EVAL3_INSTALL_SMOLVLA_DEPS=1` skipped | `EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh` |
| `FeetechMotorsBus motor check failed … found motor list: {}` | Motor bus 7–12 V supply is OFF, or USB ports swapped | Turn power on; verify with `lerobot-find-port` |
| `RevisionNotFoundError` / `HfHubHTTPError missing 'response'` | HF auth or dataset visibility | Re-run `hf auth login`; ask Rakhmatillo to confirm org membership |
| `OSError: cannot open the camera` | Wrong index, or another app is using the camera | Re-run `lerobot-find-cameras`, close other apps using the camera |
| Loop logs "Loop slower than target FPS — running ~2 Hz" | Inference can't keep up | Expected at first chunk; warn at every chunk = check device flag (`mps`/`cuda`) |
| Arm rotates wildly toward a corner of the table | Camera looking at the wrong scene (e.g. facing user) | Re-run `tools/eval3_camera_check.py` and compare to dataset sample |
| Robot does NOT move during the 20 s window | `--robot.port` wrong (leader vs follower) OR power off | Re-run `lerobot-replay` first to isolate |
| Arrow keys / Esc don't register on macOS | Accessibility permission not granted | System Settings → Privacy & Security → Accessibility → add your terminal |

---

## 7. Quick reference: the one command (after setup)

```bash
cd robot-learning-vla
source .venv/bin/activate
python scripts/eval3_vla_deploy.py \
  --robot.type=so101_follower \
  --robot.port=<follower_tty> \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{front: {type: opencv, index_or_path: <cam_idx>, width: 640, height: 480, fps: 30}}' \
  --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --policy.path=RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh \
  --policy.device=mps \
  --policy.n_action_steps=25 \
  --interpolation_multiplier=2 \
  --episode_time_s=20 \
  --fps=30
```

Type the prompt at the stderr prompt, hit Enter, watch the arm. Send back the JSONL + PNG from `outputs/eval3_rollouts/` afterward.

## 8. Why these flags?

- `--policy.path=RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh` — the SmolVLA fine-tune trained on Swift+LeCun+Obama with image+task augmentation; pulled from private HF repo.
- `--rename_map='{"observation.images.front":"observation.images.camera1"}'` — the dataset has one camera (`front`); SmolVLA was trained expecting `camera1`. This rename + the 2 empty cameras (config.empty_cameras=2) tells the policy "you only have one real camera, pad the other two".
- `--policy.n_action_steps=25` — halves SmolVLA's default 50-action chunk size so the policy re-infers twice as often. Reduces lag at chunk boundaries; smoother motion on real hardware. Doesn't change the model.
- `--interpolation_multiplier=2` — at deploy time, inserts an interpolated waypoint between every model-emitted action, doubling the effective control rate and smoothing high-jerk transitions (especially helpful for Swift's `wrist_roll`).
- `--episode_time_s=20` — Eval 3's wall-clock budget.
- `--fps=30` — matches the recording rate. Drop to 15 if your machine struggles.

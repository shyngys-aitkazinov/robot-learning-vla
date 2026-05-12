# ETH Robot Learning Sandbox

A `uv`-managed Python environment for working with the
[huggingface/lerobot](https://github.com/huggingface/lerobot) library against
SO-100 / SO-101 hardware. Installs `lerobot` from PyPI into a local `.venv/`
via `uv pip install`. Works on macOS / Apple Silicon (no CUDA) and Linux.

## Quickstart

```bash
./install.sh
```

That's it. The script is idempotent — re-run it any time to reconcile
dependencies.

## What `install.sh` does

1. Installs [`uv`](https://docs.astral.sh/uv/) if it's not on `PATH`.
2. Creates `./.venv/` on the requested Python (`uv venv --python 3.12`) — or
   reuses it if already present.
3. Runs `uv pip install lerobot` into that venv. This is the install command
   the team has verified working on macOS 14 / Apple Silicon; the more
   declarative `uv sync` path tends to fail on this combo because of extras
   that lack wheels (see [Platform caveats](#platform-caveats)).
4. If `HF_TOKEN` is set in the environment, runs `hf auth login`.
5. Ensures the calibration directories under
   `~/.cache/huggingface/lerobot/calibration/{teleoperators,robots}/` exist.

The script does **not** write `pyproject.toml` or `uv.lock`. The install is
imperative, not declarative — re-running the script reinstalls exactly the
same way against the latest PyPI lerobot.

## Configuration

All overrideable via env vars before running the script:

| Variable | Default | Purpose |
|---|---|---|
| `PYTHON_VERSION` | `3.12` | Python minor version pin. Stay on `3.12` on macOS 14 — see [Platform caveats](#platform-caveats). |
| `LEROBOT_SPEC` | `lerobot` | The package spec passed to `uv pip install`. Add extras here. |
| `HF_TOKEN` | unset | If set, the script logs into Hugging Face automatically. |

Example with a few extras:

```bash
LEROBOT_SPEC="lerobot[hardware,viz,feetech,dynamixel,kinematics,dev,test]" ./install.sh
```

If an extras install fails (the typical macOS 14 culprits are
`intelrealsense` and `unitree_g1`), fall back to plain `LEROBOT_SPEC="lerobot"`.

## Verifying the install

```bash
source .venv/bin/activate
lerobot-find-port              # confirms motor adapters enumerate
lerobot-find-cameras opencv    # confirms cameras enumerate
hf auth whoami                 # confirms Hugging Face login (if you ran it)
```

## Hugging Face authentication

If you skipped the `HF_TOKEN` env var, log in any time:

```bash
hf auth login --token <YOUR_TOKEN> --add-to-git-credential
```

Get a token at <https://huggingface.co/settings/tokens>. For pushing to an
organization (e.g. `RobotLearningVLA`), use a fine-grained token scoped to that
org with read+write on repos/datasets, and confirm membership at
`https://huggingface.co/organizations/<org>/settings/members`.

## Calibration files

Calibration tells lerobot the homing offset and joint range of every motor on
each arm, in units of raw STS-3215 encoder counts (0–4095). The JSON files
are written by `lerobot-calibrate` to:

```
~/.cache/huggingface/lerobot/calibration/
  teleoperators/so_leader/my_awesome_leader_arm.json     # leader arm
  robots/so_follower/my_awesome_follower_arm.json        # follower arm
```

The basenames (`my_awesome_leader_arm`, `my_awesome_follower_arm`) are the
`--teleop.id` / `--robot.id` strings used on this machine — those are the
**exact ids** you must pass to every later `lerobot-teleoperate` /
`lerobot-record` / `lerobot-replay` call so the matching file is picked up.
The directory layout (`so_leader` / `so_follower`) is fixed by lerobot — do
**not** rename it.

Override the root with `HF_LEROBOT_CALIBRATION=/some/other/dir` or per-arm
with `--teleop.calibration_dir=...` / `--robot.calibration_dir=...`.

### Schema

Each JSON is a mapping of motor name → encoder calibration:

```json
{
    "shoulder_pan":   { "id": 1, "drive_mode": 0, "homing_offset":  -872, "range_min": 1108, "range_max": 3041 },
    "shoulder_lift":  { "id": 2, "drive_mode": 0, "homing_offset":   -86, "range_min":  849, "range_max": 3204 },
    "elbow_flex":     { "id": 3, "drive_mode": 0, "homing_offset":  -208, "range_min":  967, "range_max": 3176 },
    "wrist_flex":     { "id": 4, "drive_mode": 0, "homing_offset": -2015, "range_min":  813, "range_max": 3058 },
    "wrist_roll":     { "id": 5, "drive_mode": 0, "homing_offset":  1795, "range_min":    0, "range_max": 4095 },
    "gripper":        { "id": 6, "drive_mode": 0, "homing_offset": -1093, "range_min": 1892, "range_max": 3091 }
}
```

`wrist_roll` is the full-turn joint, so its range stays at 0–4095. The other
five are bounded; their `range_min` / `range_max` come from physically moving
the joint through its travel during calibration.

### Generating fresh calibrations

```bash
# 1. Find each USB-serial port (unplug the cable when prompted).
lerobot-find-port

# 2. Leader (teleop arm).
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodemXXXXXX \
    --teleop.id=my_awesome_leader_arm     # use this exact id

# 3. Follower (robot arm).
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodemYYYYYY \
    --robot.id=my_awesome_follower_arm    # use this exact id
```

If you pick different ids, all subsequent `--teleop.id` / `--robot.id`
arguments need to match — and the README examples below won't apply
verbatim.

The script walks you through (a) homing each joint at its midpoint and
(b) moving each joint through its full range. After that the JSON is written
and re-used on every subsequent `lerobot-teleoperate` / `lerobot-record` /
`lerobot-replay` call that passes the same `--*.id`.

## Teleoperation

`lerobot-teleoperate` streams positions from the leader arm to the follower
arm in real time and (optionally) opens a [Rerun](https://www.rerun.io/)
viewer for the camera + joint state.

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

Notes:
- `--robot.cameras` takes a draccus-style dict. Multiple cameras: add more
  keys (e.g. `{ front: {...}, wrist: {...} }`).
- `--display_data=false` if you do not want the Rerun viewer (saves a lot of
  CPU on Apple Silicon — see [Troubleshooting](#troubleshooting)).
- The leader / follower ports are the strings reported by
  `lerobot-find-port`. They differ per laptop — do not copy these blindly.

## Recording a dataset

`lerobot-record` runs the same teleop loop, captures every frame of state +
action + cameras, and (by default) uploads the result to the Hugging Face
Hub when the session ends.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B140317761 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5B141136041 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true \
    --dataset.repo_id=RobotLearningVLA/banana_red_bowl_test_shyngys_1 \
    --dataset.num_episodes=2 \
    --dataset.single_task="Pick up the banana and put it into the red bowl" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264_videotoolbox
```

Key flags:

| Flag | Meaning |
|---|---|
| `--dataset.repo_id` | Target `<org-or-user>/<name>` on the Hugging Face Hub. Use your org slug (e.g. `RobotLearningVLA`) to push under the org. |
| `--dataset.num_episodes` | Number of episodes to record in this session. |
| `--dataset.single_task` | Language label attached to every frame (used by VLA policies). |
| `--dataset.episode_time_s` | (Default 60) max seconds per episode. `→` ends early. |
| `--dataset.reset_time_s` | (Default 60) seconds between episodes to reset the scene. `→` ends early. |
| `--dataset.streaming_encoding` | Encode videos in a background thread as frames arrive (recommended). |
| `--dataset.vcodec` | `h264_videotoolbox` on macOS (Apple HW accel), `libsvtav1` / `libx264` otherwise. |
| `--dataset.push_to_hub` | (Default `true`) automatically upload at end. Set `false` to keep local-only. |
| `--dataset.private` | (Default `false`) set `true` to create the Hub repo as private. |

Keyboard shortcuts while recording:

| Key | Action |
|---|---|
| `→` | advance to next phase (end episode / end reset early) |
| `←` | re-record the current episode |
| `Esc` | stop the whole session |

On macOS, grant the terminal **Accessibility** permission (System Settings →
Privacy & Security → Accessibility → add your terminal). Without it the keys
fall through to the shell as raw escape codes (`^[[C` etc.).

## Replaying a dataset

`lerobot-replay` reads an existing dataset from the Hub and drives the
follower arm through the recorded actions — useful for sanity-checking that
calibration + hardware still match what was recorded.

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5B140317761 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=RobotLearningVLA/banana_red_bowl_test_shyngys_1 \
    --dataset.episode=0
```

`--dataset.episode` picks one episode by index. Replay does not stream camera
frames back — it only plays actions on the follower.

## Hugging Face datasets

lerobot datasets live on the Hub under
`https://huggingface.co/datasets/<org-or-user>/<name>`. Locally they cache to
`~/.cache/huggingface/lerobot/<org-or-user>/<name>/`.

### Org inventory (`RobotLearningVLA`)

Current datasets in the team org, as of last check:

| Dataset | Visibility | `v3.0` tag | Episodes | FPS | Robot |
|---|---|---|---|---|---|
| `RobotLearningVLA/banana_blue_bowl_eval1`  | private | yes | 20 | 30 | so_follower |
| `RobotLearningVLA/banana_green_bowl_eval1` | private | yes | 20 | 30 | so_follower |
| `RobotLearningVLA/banana_red_bowl_eval1`   | private | yes | 20 | 30 | so_follower |
| `RobotLearningVLA/taylor_swift_1`          | public  | yes | 20 | 30 | so_follower |

All datasets are `codebase_version=v3.0` and carry the matching `v3.0` git
tag, so `lerobot-replay` / training can pull them directly. New datasets
pushed by `lerobot-record` will need to be tagged the same way before they
can be consumed — see
[The version-tag requirement](#the-version-tag-requirement-for-replay--training).

Refresh the inventory yourself any time:

```bash
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
for d in sorted(api.list_datasets(author='RobotLearningVLA'), key=lambda x: x.id):
    refs = api.list_repo_refs(d.id, repo_type='dataset')
    info = api.repo_info(d.id, repo_type='dataset')
    print(f'{d.id}  private={info.private}  tags={[t.name for t in refs.tags] or \"NONE\"}')
"
```

### Joining the org and pulling a dataset

1. Ask a `RobotLearningVLA` admin to invite your Hugging Face account at
   <https://huggingface.co/organizations/RobotLearningVLA/settings/members>.
   You need at least **read** access to pull, **write** access to push or
   tag.
2. Create a token at <https://huggingface.co/settings/tokens>. For
   org-scoped access, prefer a **fine-grained** token with read+write on
   repos/datasets in `RobotLearningVLA`.
3. Log in locally:

   ```bash
   hf auth login --token <YOUR_TOKEN> --add-to-git-credential
   hf auth whoami            # should list "orgs=...,RobotLearningVLA"
   ```

4. Download / cache a dataset. Three equivalent options:

   ```bash
   # (a) Replay drives the follower arm from a dataset — also caches it.
   lerobot-replay --robot.type=so101_follower --robot.port=... \
       --robot.id=my_awesome_follower_arm \
       --dataset.repo_id=RobotLearningVLA/banana_green_bowl_eval1 \
       --dataset.episode=0

   # (b) Pre-download into ~/.cache/huggingface/lerobot/... without driving hardware.
   uv run python -c "
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset('RobotLearningVLA/banana_green_bowl_eval1')
   print(ds.num_episodes, 'episodes,', ds.num_frames, 'frames')
   "

   # (c) Plain git-style fetch into an explicit dir (raw files, no lerobot parsing).
   hf download RobotLearningVLA/banana_green_bowl_eval1 \
       --repo-type dataset --local-dir ./datasets/banana_green_bowl_eval1
   ```

   (a) and (b) cache to `~/.cache/huggingface/lerobot/RobotLearningVLA/<name>/`
   and are what `lerobot-*` commands read from. (c) is for poking at raw
   parquet/MP4s.

5. If you get `RevisionNotFoundError` (masked by `HfHubHTTPError missing
   'response'`), the dataset isn't tagged yet — see below.

### Pushing

`lerobot-record` pushes automatically at the end of a session if
`--dataset.push_to_hub=true` (the default). You must already be logged in
(see [Hugging Face authentication](#hugging-face-authentication)) and a
member of the org with write access on dataset repos.

To push an existing local dataset manually:

```bash
hf upload <org>/<name> ~/.cache/huggingface/lerobot/<org>/<name> --repo-type dataset
```

### The version-tag requirement (for replay / training)

When loading a dataset, lerobot calls `get_safe_version()`, which looks for a
Hub git **tag** matching the dataset's `meta/info.json:codebase_version`
(typically `v3.0` for the current schema). If the tag is missing, replay /
training fails with `RevisionNotFoundError`, masked by a confusing
`TypeError: HfHubHTTPError missing 'response'` on `huggingface_hub` 1.x.

Add the tag once per dataset after pushing:

```python
uv run python -c "from huggingface_hub import HfApi; HfApi().create_tag('<org>/<name>', tag='v3.0', repo_type='dataset')"
```

Pull the right tag string from the dataset itself if you're unsure:

```python
uv run python -c "
from huggingface_hub import HfApi
import json
p = HfApi().hf_hub_download('<org>/<name>', 'meta/info.json', repo_type='dataset')
print(json.load(open(p))['codebase_version'])
"
```

### Inspecting a dataset

```bash
lerobot-dataset-viz --dataset.repo_id=<org>/<name>     # local viewer (rerun)
hf repo files --repo-type dataset <org>/<name>         # list files on Hub
```

Dataset file layout (on Hub and in local cache):

```
<org>/<name>/
  meta/
    info.json                           # codebase_version, fps, robot_type, totals
    stats.json                          # per-feature mean/std
    tasks.parquet                       # task index
    episodes/chunk-000/file-000.parquet # per-episode metadata
  data/chunk-000/file-000.parquet       # state/action timeseries
  videos/observation.images.<cam>/chunk-000/file-000.mp4
```

## Platform caveats

`lerobot[all]` will **not** resolve on macOS 14 + Python 3.12 because two of its
sub-extras require packages with no compatible wheel:

| Sub-extra | Blocking package | Why |
|---|---|---|
| `intelrealsense` | `pyrealsense2-macosx>=2.56` | only ships `macosx_15_0_arm64` wheel — needs macOS 15 |
| `unitree_g1` | `onnxruntime==1.26.0` | only ships `cp313` macOS-arm wheel — needs Python 3.13 |

The default `LEROBOT_SPEC=lerobot` sidesteps both — bare lerobot installs
fine. If you upgrade to macOS 15 *and* Python 3.13 you can add
`intelrealsense,unitree_g1` back in (3.13 then brings its own build issues —
e.g. `labmaze` needing Bazel — check upstream lerobot's
[CONTRIBUTING.md](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md)
for details).

## Other useful CLIs

```bash
lerobot-find-port            # discover motor-adapter port (unplug when prompted)
lerobot-find-cameras opencv  # enumerate cameras + save preview frames
lerobot-setup-motors …       # assign motor IDs 1–6 to a fresh Feetech daisy chain
lerobot-info                 # print env / package versions
```

All `lerobot-*` entry points are declared in lerobot's own `pyproject.toml`
under `[project.scripts]` — `uv run lerobot-info` will list them after a
successful `uv sync`.

## Troubleshooting

**`FeetechMotorsBus motor check failed … Full found motor list: {}`**
Almost always: external 7–12 V supply to the motor bus is unplugged / off, or
the leader and follower ports are swapped. USB alone does not power the
servos.

**`FileExistsError … /Users/<you>/.cache/huggingface/lerobot/<org>/<name>`**
A previous failed `lerobot-record` left an empty cache dir. Either pick a new
`--dataset.repo_id` or `rm -rf` the offending dir and retry.

**`RevisionNotFoundError` / `HfHubHTTPError missing 'response'` when replaying a dataset**
The dataset has no version tag matching its `codebase_version`. Add one:

```python
from huggingface_hub import HfApi
HfApi().create_tag("<org>/<dataset>", tag="v3.0", repo_type="dataset")
```

The exact tag string lives in `meta/info.json:codebase_version` of the dataset.

**Recording warns `Record loop is running slower (… Hz) than the target FPS (30 Hz)`**
On macOS / CPU-only this is usually `--display_data=true` (rerun streaming) or
camera resolution. Try `--display_data=false`, lower fps, or smaller frames
(`width=320 height=240`).

## Files of note

- `install.sh` — bootstrap script (see [What `install.sh` does](#what-installsh-does)).
- `pyproject.toml` — minimal project metadata. The actual install is driven
  by `install.sh` calling `uv pip install`, not by `uv sync`.
- `.venv/` — provisioned by `install.sh` (git-ignored).
- `camera.py` — quick OpenCV live-preview using `lerobot.cameras.OpenCVCamera`
  (press `q` to quit).
- `CLAUDE.md` — guidance for [Claude Code](https://claude.com/claude-code)
  when iterating on this repo.

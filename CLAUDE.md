# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two things in one tree:

1. A thin sandbox around the upstream
   [huggingface/lerobot](https://github.com/huggingface/lerobot) library —
   teleop / calibration / record / replay on SO-100 / SO-101 hardware. The
   runtime `lerobot` package is installed from PyPI via `uv pip install` into
   `./.venv/`. There is **no** local checkout of lerobot to edit — treat
   `lerobot` as a third-party dependency.
2. The team's **Eval 3 VLA pipeline** (`scripts/eval3_*`, `tools/eval3_*`,
   `docs/eval3/`) — SmolVLA fine-tuning + closed-loop deploy for the "place
   the coke on <celebrity>" task. This is now the bulk of the code in the
   repo. The training stack monkey-patches `lerobot.datasets.factory` and
   pre-loads `lerobot.policies.groot.*` modules with a shim — both are
   load-order-sensitive, see below.

Platform targets:
- **macOS 14 / Apple Silicon / Python 3.12** — primary dev box for teleop /
  record / replay and for small SmolVLA fine-tunes (MPS). The `eval3_*`
  scripts default to MPS via `scripts/eval3_device.py`.
- **Linux + CUDA** — full SmolVLA fine-tunes (the `run_eval3_smolvla_aug_train.sh`
  wrapper is parameterised with `EVAL3_POLICY_DEVICE=cuda`).

## Commonly-used commands

```bash
./install.sh                                # base install: uv + .venv + lerobot + feetech-servo-sdk
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh   # adds transformers/accelerate/sentencepiece/num2words
source .venv/bin/activate                   # or prefix commands with `uv run`

# Hardware bring-up (see README for full flow + flags)
lerobot-find-port
lerobot-find-cameras opencv
lerobot-calibrate ... --teleop.id=my_awesome_leader_arm
lerobot-teleoperate ...
lerobot-record ...
lerobot-replay ...

# Eval 3 — inspection / sanity
python tools/inspect_lerobot_dataset.py                  # default: RobotLearningVLA/taylor_swift_1
python tools/eval3_smolvla_compat.py                     # prints rename_map + empty_cameras flags
python scripts/train_eval3_bc_overfit.py --steps 1500    # tiny BC, pipeline gate (NOT a VLA)

# Eval 3 — SmolVLA fine-tune
./scripts/run_eval3_smolvla_train.sh                     # Swift-only baseline
./scripts/run_eval3_smolvla_aug_train.sh                 # 3-celeb + full augmentation stack
python scripts/train_eval3_smolvla.py ...                # raw entry — same flags as `lerobot-train`

# Eval 3 — closed-loop deploy on SO-101 (see docs/eval3/friend_deploy_handoff.md)
python scripts/eval3_vla_deploy.py --policy.path=... --rename_map='...' --task='...' --episode_time_s=20
python scripts/eval3_vla_deploy.py ... --dry_run         # load checkpoint without driving hardware
```

There is **no** test suite and no lint config. Verification is by running the
inspectors / dry-run paths above.

## Eval 3 pipeline architecture (load-order matters)

Three pieces in `scripts/` work together. They must be entered through one of
the wrapper scripts or `scripts/train_eval3_smolvla.py` / `scripts/eval3_vla_deploy.py`
— calling `lerobot-train` or `lerobot.policies.*` directly will hit one of two
import-time crashes.

1. **`eval3_lerobot_shim.py:apply()`** — must run **before any `lerobot.policies`
   import**. With `transformers` installed, importing `lerobot.policies` cascades
   into `lerobot.policies.groot.groot_n1`, whose dataclass inherits from a
   transformers parent and crashes with "non-default argument 'backbone_cfg'
   follows default argument". The shim's surgical fix: pre-load the tokenizer
   processor (so transformers is cached in `sys.modules`), flip
   `lerobot.utils.import_utils._transformers_available = False` just long
   enough to import the GROOT modules with stub parents, then restore the
   flag. Both `train_eval3_smolvla.py` and `eval3_vla_deploy.py` call
   `apply()` at the top.

2. **`eval3_concat_patch.py:apply_concat_patch()`** — only the trainer needs
   this. When `EVAL3_EXTRA_REPOS` is set (comma-separated repo_ids), it
   monkey-patches `lerobot.datasets.factory.make_dataset` to build one
   `LeRobotDataset` per repo, wrap each in `Eval3PrepDataset`, then
   `ConcatLeRobotDataset` them. Upstream's `MultiLeRobotDataset` raises
   `NotImplementedError` so this patch is the only way to joint-train across
   celebrities. Stats (mean/std/min/max/quantiles) are merged across the
   **filtered** frame set, not the raw datasets — otherwise action
   normalization would be wrong after episode/frame truncation.

3. **`eval3_dataset_prep.py:Eval3PrepDataset`** — proxy that wraps each
   `LeRobotDataset` with four env-var-driven layers (all default ON in
   `run_eval3_smolvla_aug_train.sh`):

   | Env var | Default | Effect |
   |---|---|---|
   | `EVAL3_MAX_FRAMES_PER_EP` | `600` | Truncate each episode to first N frames (= 20s @ 30fps; Eval 3 has a 20s wall-clock budget) |
   | `EVAL3_TASK_AUG` / `EVAL3_TASK_AUG_CANONICAL_P` | `1` / `0.8` | Rewrite `row["task"]` at `__getitem__` (80% canonical demo wording, 20% original) |
   | `EVAL3_BG_REPLACE` / `EVAL3_BG_REPLACE_P` | `1` / `0.3` | Replace background pixels using `outputs/eval3_masks/<slug>/bg_mask.npy` + `outputs/eval3_backgrounds/*.png` |
   | `EVAL3_PRINT_SHUFFLE` / `EVAL3_PRINT_SHUFFLE_P` | `0` / `0.5` | Swap non-target print regions (target stays put to preserve action-image alignment) |
   | `EVAL3_{SWIFT,LECUN,OBAMA}_EPISODE_FILTER` | per-dataset audit lists | Drop bad-recording / wrist-roll-negative episodes (see `run_eval3_smolvla_aug_train.sh` header) |

   Augmenters are picklable (DataLoader workers fork them) via `__reduce__`;
   masks / backgrounds are lazy-loaded per worker.

### SmolVLA single-camera workaround (train AND deploy)

`lerobot/smolvla_base` expects `observation.images.camera{1,2,3}`. The team's
datasets carry one stream, `observation.images.front`. Both training and
deploy must pass:

```text
--rename_map='{"observation.images.front":"observation.images.camera1"}'
--policy.empty_cameras=2
```

Pulling these flags directly from a dataset: `python tools/eval3_smolvla_compat.py --repo-id <repo>`.

If train and deploy use a different `rename_map`, stats normalization
silently mismatches — `eval3_vla_deploy.py` calls `rename_stats(...)` to keep
them aligned and **assumes** the same map was used at train time.

## File map (non-obvious bits)

- `scripts/train_eval3_smolvla.py` — entry point; calls
  `eval3_lerobot_shim.apply()` + `eval3_concat_patch.apply_concat_patch()`
  then delegates to `lerobot.scripts.lerobot_train.main`. Use this instead of
  `lerobot-train` directly.
- `scripts/eval3_vla_deploy.py` — closed-loop SmolVLA on real SO-101. Builds
  the same observation→preprocess→policy→postprocess→robot pipeline as
  `lerobot-record` but driven by policy actions. Supports `--dry_run`,
  `--policy.n_action_steps=25` (halves the default chunk to smooth chunk
  boundaries), `--interpolation_multiplier=2` (inserts midpoint waypoints).
- `scripts/eval3_rollout.py` — offline rollout harness with `--mock-frame-index`
  (re-uses one dataset frame as a stationary observation). For pipeline
  testing only, not for scoring.
- `scripts/train_eval3_bc_overfit.py` + `scripts/eval3_models.py` — a tiny
  CNN+proprio BC head. Sanity gate to prove `LeRobotDataset → resize → MSE`
  works end-to-end. **Not** a course-compliant VLA on its own.
- `scripts/eval3_device.py` — device resolution; prefers MPS > CUDA > CPU.
- `tools/eval3_*.py` — offline audits (`*_compare_models`, `*_audit_*`),
  augmentation pre-build (`eval3_extract_masks`, `eval3_build_background_pool`),
  visualisation (`eval3_visualize_augmentation`, `eval3_render_overlay`),
  synthetic OOD test (`eval3_synthetic_ood_test`). Pure offline — none of
  these talk to hardware.
- `tools/eval3_charuco_*.py` — **experimental** synthetic-on-real pipeline for
  Eval 3 OOD runs (see [docs/eval3/charuco_pipeline.md](docs/eval3/charuco_pipeline.md)):
  - `eval3_make_charuco_board.py` — generates a printable ChArUco PDF (default:
    130×180 mm board content centred on A4 with crop marks, 5×7 grid,
    DICT_4X4_50, 60 mm central green chroma). All knobs are mm-accurate; the
    output PDF MediaBox is exactly the requested paper size at the requested
    DPI. The `--content-mm WxH` flag accepts arbitrary non-standard sizes.
  - `eval3_charuco_check.py` — live camera HUD showing detected markers,
    chessboard corner count, projected board outline + chroma rectangle, and
    homography reprojection RMS. Use to verify markers detect at recording
    distance.
  - `eval3_charuco_compose.py` — live preview of the full per-frame
    compositing pipeline: lock homography on frame 0 (matching the offline
    "homography once per episode" assumption), warp a target image into the
    projected board outline, HSV-key the chroma to derive a can mask, restore
    the can on top. The exported `compose()` function is what the (not yet
    committed) batch post-processor will call per frame. The `m` key in live
    mode tints the masks for HSV tuning.
  All three reuse `cv2.aruco` which ships with the lerobot install (no extra
  `pip install` required). They share `build_board()` / `open_camera()` —
  `compose.py` imports them from `check.py`. The camera opener auto-falls
  back from the lerobot `OpenCVCamera` wrapper to plain `cv2.VideoCapture`.
- `scripts/camera.py` — OpenCV live preview using `lerobot.cameras.opencv`.
  (Moved here from the repo root.)
- `docs/eval3/` — foundation docs. The two highest-value ones for the agent
  are `task3_deploy_readiness.md` (training/deploy compatibility checklist)
  and `friend_deploy_handoff.md` (self-contained recipe for the published
  `RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh` checkpoint, including
  the per-prompt `wrist_roll` expectations that distinguish v3 from the
  broken v1).
- `requirements-eval3-train.txt` — **pointer file, not pip-installable**;
  it just documents the `EVAL3_INSTALL_SMOLVLA_DEPS=1` install path.
- `outputs/` — gitignored. Holds `train/<job>/checkpoints/<step>/pretrained_model/`,
  `eval3_rollouts/rollout_<UTC>.{jsonl,firstframe.png}`, `eval3_masks/<slug>/`,
  `eval3_backgrounds/*.png`, audit reports.

## Platform gotchas

- **`lerobot[all]` does NOT install on macOS 14 + Python 3.12.** Two
  sub-extras have no compatible wheel: `intelrealsense`
  (`pyrealsense2-macosx ≥ 2.56` needs macOS 15) and `unitree_g1`
  (`onnxruntime 1.26` needs Python 3.13 on macOS-arm). The default
  `LEROBOT_SPEC=lerobot` sidesteps both. Add extras à la carte if you need
  them.
- **MPS is the practical inference device on Apple Silicon** for SmolVLA —
  the deploy script and `eval3_device.py` default to it. CPU fallback works
  but is slow enough that the control loop will drop below 30 Hz. Training
  on MPS is fine for smoke runs (`EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=1`);
  full 50k-step runs belong on CUDA.
- **`lerobot-record` performance on macOS**: the record loop often runs
  below the target 30 Hz. Biggest cost is `--display_data=true` (rerun
  streaming). Drop it, lower fps, or lower camera resolution if frames are
  being dropped.
- **macOS Accessibility**: `lerobot-record` and `eval3_vla_deploy.py` both
  use `pynput` for arrow-key / Esc shortcuts. On macOS this requires
  granting the terminal Accessibility permission (System Settings → Privacy
  & Security → Accessibility). Without it, keys leak through as raw escape
  codes (`^[[C`) and shortcuts don't register.

## Hugging Face dataset workflow

Datasets live under `RobotLearningVLA/<name>` on the Hub (README has the
current inventory). `lerobot-record` pushes automatically at session end.
**Every new dataset needs a Hub git tag matching `meta/info.json:codebase_version`
(currently `v3.0`) before it can be replayed/trained on** — otherwise
loading fails with `RevisionNotFoundError`, masked by a confusing
`HfHubHTTPError missing 'response'` error on `huggingface_hub` 1.x.

Tag after pushing:

```bash
uv run python -c "from huggingface_hub import HfApi; HfApi().create_tag('<org>/<name>', tag='v3.0', repo_type='dataset')"
```

Eval 3 active corpus: `RobotLearningVLA/{taylor_swift_1,yann_lecun_1,barack_obama_1}`.
Published checkpoint: `RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh`.

## Common failure modes (quick diagnosis)

| Symptom | Likely cause |
|---|---|
| `FeetechMotorsBus motor check failed … found motor list: {}` | external 7–12 V supply to the motor bus is off, or leader/follower USB ports are swapped. USB alone does not power servos. |
| `FileExistsError … .cache/huggingface/lerobot/<org>/<name>` | a previous failed `lerobot-record` left an empty cache dir. Either pick a new `--dataset.repo_id` or `rm -rf` the offending dir. |
| `RevisionNotFoundError` / `HfHubHTTPError missing 'response'` on replay or training | the target dataset has no `v3.0` git tag on the Hub. See section above. |
| `non-default argument 'backbone_cfg' follows default argument` at import time | `lerobot.policies` was imported before `eval3_lerobot_shim.apply()`. Enter via `scripts/train_eval3_smolvla.py` or `scripts/eval3_vla_deploy.py`. |
| `ImportError: Package 'num2words' is required …` | SmolVLA extras not installed. `EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh` (or `uv pip install transformers accelerate sentencepiece num2words`). |
| `ModuleNotFoundError: No module named 'scservo_sdk'` | feetech servo SDK missing. `install.sh` installs it; rerunning the script fixes it. |
| Record / deploy loop runs at < 30 Hz | drop `--display_data=true`, lower fps, reduce camera resolution, or move policy to MPS/CUDA. |
| `eval3_concat_patch` warns "missing mask or bg dir" | run `tools/eval3_extract_masks.py` + `tools/eval3_build_background_pool.py` to populate `outputs/eval3_masks/<slug>/` and `outputs/eval3_backgrounds/`, or set `EVAL3_BG_REPLACE=0 EVAL3_PRINT_SHUFFLE=0`. |

## When editing this repo

- Keep `install.sh` idempotent and re-runnable. The `EVAL3_INSTALL_SMOLVLA_DEPS=1`
  branch is also expected to be safely re-runnable.
- Keep `README.md` and `CLAUDE.md` in sync — the README is for humans, this
  file is for the agent. Don't duplicate; cross-reference.
- Don't add lerobot source-level changes here — fork
  [huggingface/lerobot](https://github.com/huggingface/lerobot) for that.
  The two existing exceptions (`eval3_lerobot_shim.py`,
  `eval3_concat_patch.py`) work *around* lerobot at import time rather than
  modifying it.
- When changing augmentation defaults, keep the env-var contract: the
  `run_eval3_smolvla_aug_train.sh` header is the canonical place to surface
  knobs, and `eval3_concat_patch._patched_make_dataset` is where they're
  read.
- Train and deploy `--rename_map` must stay identical. If you change one,
  search for the other.

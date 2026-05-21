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

**Project status — Eval 3 is complete.** The approach: SmolVLA fine-tuned on
real `dataset_v4_*` teleop data with the vision encoder + language tower
**frozen** (expert-only). Two final models on the Hub under `RobotLearningVLA/`:
`eval3-vla-v6-smolvla-fresh-v4slots-expert-50k` (baseline frozen-encoder VLA)
and `eval3-smolvla-v16-pinsv5-step5k` (the **deployed** model — the v16
slot-bottleneck variant; deploy with `run_eval3_deploy_battery.sh v16`). See
`README.md` (TA-facing writeup) and `docs/eval3/v16_playbook.md`.

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

# Eval 3 — closed-loop deploy on SO-101 (see docs/eval3/v16_playbook.md)
./scripts/run_eval3_deploy_battery.sh v16 --task='Place the coke on Taylor Swift'   # final model
./scripts/run_eval3_deploy_battery.sh v6_synth_15k --task='Place the coke on Barack Obama'
python scripts/eval3_vla_deploy.py --policy.path=... --rename_map='...' --task='...' --episode_time_s=20
python scripts/eval3_vla_deploy.py ... --dry_run         # load checkpoint without driving hardware

# ALWAYS run this before plugging in the arm — catches missing rename_map / camera-key mismatches
python tools/eval3_check_deploy_command.py --policy-pretrained-path <CHECKPOINT> \
    --rename-map '{"observation.images.front":"observation.images.camera1"}' \
    --task 'Place the coke on Taylor Swift'
```

There is **no** test suite and no lint config. Verification is by running the
inspectors / dry-run paths above. The deploy-command validator above is the
single most important pre-flight check — without `rename_map` the policy
receives black frames and silently fails (see `docs/eval3/friend_deploy_handoff.md`).

## Eval 3 deploy battery (`run_eval3_deploy_battery.sh`)

The primary deploy entry. Wraps `scripts/eval3_vla_deploy.py` with the
follower TTY + camera index baked in for Shyngys's rig
(`FOLLOWER_TTY=/dev/tty.usbmodem5B140317761`, `CAM_IDX=0` — override per-run
via env vars). Selects a named checkpoint and applies the friend-recipe
deploy guards by default. Pass any extra `eval3_vla_deploy.py` flags after the
checkpoint name. The table below is the curated shortlist; `--help` prints the
full inventory (v6_synth/v10/aux/slot intermediate snapshots, etc.).

| Name | Repo | Default biases |
|---|---|---|
| `v16` | `eval3-smolvla-v16-pinsv5-step5k` — **final deployed model**; v16 slot-bottleneck, two cameras (sets the 2-cam `rename_map` + `empty_cameras=1` itself) | **OFF** (raw-policy) |
| `v4slots_expert` | `eval3-vla-v6-smolvla-fresh-v4slots-expert-50k` — baseline frozen-encoder VLA on 9× real `dataset_v4_*` | **OFF** (raw-policy) |
| `v8` | `eval3-vla-v8-gripper-repair-smooth-50k` (3-way, pre-v16 best) | ON |
| `v6_combined` | `eval3-vla-v6-smolvla-fresh-combined88-50k` | ON |
| `v6_new` | `eval3-vla-v6-smolvla-fresh-new66-50k` | ON |
| `v7_d` | `eval3-vla-v7-D-obama-only-10k` (Obama-only — use only with Obama prompt) | ON |
| `v6_synth_25k` / `v6_synth_15k` | `eval3-smolvla-3way-25k-b128-v6-synth-step{25k,15k}` (ChArUco-synth) | **OFF** (raw-policy) |
| `v9_charuco` / `v9_new66_charuco` | `eval3-vla-v9-smolvla-fresh-{charuco,new66-charuco}-50k` | **OFF** (raw-policy) |
| `flower_new66` | FlowerVLA — **exits with error**; deploy script is SmolVLA-only |

`v16` defaults `POLICY_PATH` to the Hub repo `eval3-smolvla-v16-pinsv5-step5k`;
override with `EVAL3_V16_CKPT=<path-or-repo>` to deploy a different checkpoint.

The four friend-recipe **deploy guards** (set in `COMMON_ARGS`) are the
difference between "raw policy" and "deployable":
`--interpolation_multiplier=2` (midpoint waypoints between policy steps),
`--action_smoothing_alpha=0.25` (EMA on outgoing actions),
`--max_action_delta_deg=6` (per-step joint slew limit), and
`--gripper_open_bias_deg=5` (additive bias when policy commands an open above
20°). Each checkpoint's `NO_BIASES` override resets these to 0/1 for evaluating
the raw policy distribution — that pattern is how the v6_synth/v9 entries
intentionally bypass the guards.

### Rollout JSONL contract (trajectory analysis)

Every deploy run writes `outputs/eval3_rollouts/rollout_<UTC>.{jsonl,firstframe.png}`.
The first line is a header (instruction, policy_path/sha, all deploy flags,
checkpoint stats counts). Each subsequent line is one control-loop step with:

- `step`, `t_episode_s`, `dt_s`, `loop_hz`, `ran_policy_inference`
- `state` + `joint_keys` (current robot proprio, joint name order)
- `policy_action_raw` (model output), `policy_action_processed` (after rename/postprocess),
  `policy_action_guarded` (after smoothing/delta-clip/gripper-bias), `sent_action` (what hit the bus)
- `action` (final 6-tuple in `joint_keys` order)

This is the contract for `tools/eval3_audit_live_wrist_roll.py`,
`tools/eval3_audit_gripper_opens.py`, and any new trajectory-analysis tool —
read JSONL line-by-line, the header has the run config, the rest are step
records. Steps where `ran_policy_inference=false` are interpolated/repeated
chunk frames (no `policy_action_*` fields, only `sent_action`).

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

3. **`eval3_smolvla_aux_head.py:apply()`** — optional. Adds a 3-way
   position-classification head (`left` / `middle` / `right` print) on top
   of SmolVLA's action expert `suffix_out` (B, 50, 720). Mean-pools across
   the chunk and computes a CrossEntropy loss against the per-dataset
   target position derived from the repo_id (`*_left_*` → 0, `*_middle_*`
   → 1, `*_right_*` → 2; other repos get `-100` ignore_index).

   **Purpose**: force the expert's hidden state to encode language-image
   binding so the action head can use prompt as a feature. Diagnosed need:
   v6_synth_15k learned the `(observation.state, image) → action` shortcut
   and totally ignored the prompt (cross-prompt Δ < 1° on training frames).

   **Wiring**: patches `VLAFlowMatching.__init__` (adds the head), its
   `forward` (computes aux CE loss, stashes on self), `SmolVLAPolicy.forward`
   (pulls aux loss into the main loss), and
   `lerobot.processor.converters._extract_complementary_data` (so the
   `target_position` label survives `batch_to_transition`). All from the
   eval3 script — no `.venv` edits.

   **Env vars**:

   | Env var | Default | Effect |
   |---|---|---|
   | `EVAL3_AUX_POS_LOSS_WEIGHT` | `0` | Multiplier on the aux CE loss. `0` = patch is a no-op (head exists but contributes no gradient). Set to `0.3`-`0.5` to enable. |
   | `EVAL3_AUX_POS_DROPOUT` | `0.1` | Dropout inside the classification head. |
   | `EVAL3_AUX_POS_HIDDEN` | `256` | Hidden width of the head's MLP. |

   When enabled, `loss_dict` carries `aux_pos_loss`, `aux_pos_acc`,
   `aux_pos_weight` in addition to the usual action losses. Saving a
   checkpoint persists the head's weights; loading later just shows a
   benign "Missing key" warning if the patch isn't re-applied at inference
   (the head's weights load but go unused — inference doesn't need it).

4. **`eval3_dataset_prep.py:Eval3PrepDataset`** — proxy that wraps each
   `LeRobotDataset` with four env-var-driven layers (all default ON in
   `run_eval3_smolvla_aug_train.sh`):

   | Env var | Default | Effect |
   |---|---|---|
   | `EVAL3_MAX_FRAMES_PER_EP` | `600` | Truncate each episode to first N frames (= 20s @ 30fps; Eval 3 has a 20s wall-clock budget) |
   | `EVAL3_TASK_AUG` / `EVAL3_TASK_AUG_CANONICAL_P` | `1` / `0.8` | Rewrite `row["task"]` at `__getitem__` (80% canonical demo wording, 20% original) |
   | `EVAL3_BG_REPLACE` / `EVAL3_BG_REPLACE_P` | `1` / `0.3` | Replace background pixels using `outputs/eval3_masks/<slug>/bg_mask.npy` + `outputs/eval3_backgrounds/*.png` |
   | `EVAL3_PRINT_SHUFFLE` / `EVAL3_PRINT_SHUFFLE_P` | `0` / `0.5` | Swap non-target print regions (target stays put to preserve action-image alignment) |
   | `EVAL3_{SWIFT,LECUN,OBAMA}_EPISODE_FILTER` | per-dataset audit lists | Drop bad-recording / wrist-roll-negative episodes (see `run_eval3_smolvla_aug_train.sh` header) |
   | `EVAL3_GRIPPER_REPAIR` / `EVAL3_GRIPPER_OPEN_TARGET` / `EVAL3_GRIPPER_OPEN_THRESHOLD` | `1` / `55` / `20` | v8 label-repair: lift already-open gripper commands (≥20°) to ≥55° to fight the `dataset_v2` truncated-q90/q99 bug |
   | `EVAL3_ACTION_SMOOTH_WINDOW` / `EVAL3_ACTION_SMOOTH_GRIPPER` | `3` / `0` | v8 arm-label low-pass; gripper excluded so grasp/release timing stays crisp |
   | `EVAL3_NEW_EPISODE_KEEP` | (per-repo lists) | v6.2: positive-keep lists per repo (overrides the negative-filter env vars when set) |

   Augmenters are picklable (DataLoader workers fork them) via `__reduce__`;
   masks / backgrounds are lazy-loaded per worker.

### Training launcher matrix

Each launcher is a thin shell wrapper that exports env vars and shells into
`scripts/train_eval3_smolvla.py`. They diverge in dataset corpus and
augmentation defaults — read the file header for the exact recipe.

| Launcher | Corpus | Notes |
|---|---|---|
| `run_eval3_smolvla_train.sh` | Swift-only | Baseline. No augmentation stack. |
| `run_eval3_smolvla_aug_train.sh` | 3-way (Swift + LeCun + Obama) | **v8 current default** — full layers 1-5 (truncation + task-aug + bg-replace + print-shuffle + gripper-repair + arm-smooth + torchvision transforms). |
| `run_eval3_smolvla_v5_train.sh` | dataset_v2_* (label-repaired) | v5 = original recipe on v2-cleaned data. |
| `run_eval3_smolvla_v6_synth_train.sh` | 9 × `dataset_v3_synth_<celeb>_<position>_2` (ChArUco-synth) | v6_synth — synthetic-on-real backgrounds, larger batch (16) on H100. |
| `run_eval3_smolvla_charuco_train.sh` | ChArUco synthetic-only | v9 charuco corpus. |
| `run_eval3_flower_train.sh` | new66 | FlowerVLA, NOT SmolVLA — separate path, not deployable via `eval3_vla_deploy.py`. |

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
  boundaries), `--interpolation_multiplier=2` (inserts midpoint waypoints),
  `--action_smoothing_alpha`, `--max_action_delta_deg`, `--gripper_open_bias_deg`
  (the four "deploy guards" — see deploy battery section). SmolVLA-only;
  FlowerVLA needs a different deploy entry.
- `scripts/run_eval3_deploy_battery.sh` — preferred deploy entry. Named
  checkpoint shortcuts (`v8` / `v6_combined` / `v6_new` / `v7_d` /
  `v6_synth_{15k,25k}` / `v9_charuco` / `v9_new66_charuco`); bakes
  `FOLLOWER_TTY` + `CAM_IDX` + the deploy guards; per-entry `NO_BIASES`
  override resets guards to 0 for raw-policy evaluation. Pass extra
  `eval3_vla_deploy.py` flags after the checkpoint name.
- `scripts/train_eval3_flower.py` + `scripts/run_eval3_flower_train.sh` —
  FlowerVLA training path. Checkpoint layout is `checkpoint.pt` + raw
  `dataset_statistics.json`, *not* the lerobot processor bundle that
  `eval3_vla_deploy.py` consumes, so a Flower-specific deploy script is still
  needed before these checkpoints can run on the arm.
- `scripts/eval3_smolvla_checkpoint_sweep.py` — offline sweep across
  intermediate checkpoints of one training job (uses
  `EVAL3_SAVE_FREQ` snapshots), produces a ranking. Pairs with
  `tools/eval3_abcd_benchmark.py` for cross-job comparison.
- `scripts/eval3_external_vla_data.py` + `tools/eval3_external_vla_preflight.py`
  — handles the external VLA datasets (OpenVLA / FlowerVLA runs 7-8). See
  `docs/eval3/external_vla_runs_7_8.md`.
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
  these talk to hardware. Key entries:
  - `eval3_check_deploy_command.py` — pre-flight validator (rename_map + task
    + camera keys); run before every hardware deploy.
  - `eval3_deploy_flags_from_checkpoint.py` — prints the `rename_map` /
    `empty_cameras` / `task` flags inferred from a `pretrained_model` dir.
  - `eval3_abcd_benchmark.py` — canonical offline benchmark across the v7
    A/B/C/D checkpoints; emits `OFFLINE_REPORT.md` + JSON scores. See
    `docs/eval3/abcd_model_eval.md`.
  - `eval3_audit_live_wrist_roll.py` / `eval3_audit_gripper_opens.py` —
    consume the rollout JSONL contract; quick sanity checks for the v3
    wrist-roll signature and gripper aperture distribution.
  - `eval3_synth_dataset_gen.py` / `eval3_synth_pins_dataset_gen.py` /
    `run_eval3_synth_dataset_gen.sh` — ChArUco-synth dataset generation
    (produces `RobotLearningVLA/dataset_v3_synth_*`). The Pins variant scales
    out to the Pins face-pool (`tools/build_pins_*.py`,
    `scripts/download_pins_faces.sh`).
  - `eval3_verify_truncation.py` / `eval3_export_truncated_videos.py` — verify
    `EVAL3_MAX_FRAMES_PER_EP` actually took effect, export truncated MP4s for
    visual review.
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
- `docs/eval3/` — foundation docs. The highest-value ones for the agent:
  - `friend_deploy_handoff.md` — self-contained recipe for the published
    `eval3-smolvla-3way-50k-v3-fresh` checkpoint, including the per-prompt
    `wrist_roll` expectations that distinguish v3 from the broken v1.
  - `task3_deploy_readiness.md` — training/deploy compatibility checklist
    (the `front`→`camera1` rename, `empty_cameras=2`, stats alignment).
  - `v7_deploy_checklist.md` — post-failure remediation walkthrough; pair
    with `eval3_check_deploy_command.py`.
  - `abcd_model_eval.md` — canonical offline ranking runbook for A/B/C/D
    checkpoints; defines the shortlist rule used by `eval3_abcd_benchmark.py`.
  - `hardware_eval_matrix.md` — structured hardware-trial protocol after
    offline ranking (prompt order, JSONL post-conditions to verify).
  - `tensor_contract.md`, `prompt_protocol.md`, `scene_spec.md` — the
    invariants that train + deploy + recording must all match.
  - `brev_synth_runbook.md` — the Brev/GPU recipe used to generate the
    `dataset_v3_synth_*` corpus.
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
| Deploy runs, policy moves, but "doesn't recognise celebrities" / grasps wrong | almost always missing `--rename_map` (custom deploy) / `--dataset.rename_map` (`lerobot-record`). SmolVLA sees `camera1=zeros`, all three prompts collapse. Run `tools/eval3_check_deploy_command.py` first; it prints the corrected command. |
| FlowerVLA checkpoint won't load in `eval3_vla_deploy.py` | `eval3_vla_deploy.py` is SmolVLA-only — it expects a lerobot processor bundle, not Flower's `checkpoint.pt` + `dataset_statistics.json`. `run_eval3_deploy_battery.sh flower_new66` exits with the same note. Needs a Flower-specific deploy script. |

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
- New checkpoint going to the deploy battery? Add a `case` arm to
  `scripts/run_eval3_deploy_battery.sh` (set `POLICY_PATH` +
  `DATASET_REPO_ID`; append `NO_BIASES` only if the checkpoint is meant to
  be evaluated raw) and update the header comment list. The README + battery
  header are the canonical inventory of "what's deployable today".
- An `AGENTS.md` also exists in this repo — it's a much shorter Codex-only
  view of the same setup. If you change behaviour that touches install /
  record / deploy, update both files (CLAUDE.md is the long-form; AGENTS.md
  is the lean view).

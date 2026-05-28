# v17 Camera-1 Dropout VLA — Experiment Playbook

The v17 run trains a SmolVLA slot-bottleneck policy that **cannot use the live
camera1 stream alone** to drive the trajectory: cam1 is randomly replaced by
Gaussian noise during training (per-frame, per-episode, or both), so the action
expert must complete the task from **language + frame-0 (cam2) + proprio**. The
slot head's frame-0 input (cam2) is bit-exactly preserved across every drop.

This is the "next step" after v16: v16 fixed the slot DECISION; v17 fixes the
trajectory EXECUTION shortcut.

## 1. Why v17 exists (and how it differs from v16)

v16 forces the SLOT decision through the language path:
- The slot classifier reads only cam2 (the frozen frame-0 view).
- Slot CE loss runs only on pre-grasp frames.
- The h_slot prefix token commits to one of `{left, middle, right}` and the
  action expert attends to it.

But v16 still hands the action expert the **live cam1 stream every step**. On
real teleop data, the can's motion after grasp reveals the target slot, so the
action expert can learn a "watch the can drift" shortcut and ignore both the
language and h_slot for the carry phase. (This is precisely the failure mode
that motivated v16 in the first place; v16 fixed half of it.)

**v17 removes that second shortcut.** During training:

1. **Per-episode drop** (default 35%) — for that fraction of episodes, EVERY
   frame's cam1 is replaced by Gaussian noise. The policy must execute the
   full trajectory using only language + cam2 (frame-0) + proprio.
2. **Per-frame iid drop** (default 10% pre-grasp, 30% post-grasp via 3× mult) —
   for non-dropped episodes, individual frames still get noised, simulating a
   brief sensor outage and amplifying pressure exactly where the can-motion
   shortcut is strongest.
3. **cam2 is NEVER touched** — bit-exact invariant under contract.

Everything else (slot head, frame-0 logic, pre-grasp CE mask, state aug, image
transforms, optimizer, scheduler) is **identical to v16**. v17 is v16 + one
augmenter.

## 2. Architecture

| Input | Training (v16) | Training (v17, sometimes) | Deploy (both v16 and v17) |
|---|---|---|---|
| `camera1` | current frame | current frame OR **Gaussian noise** | current frame |
| `camera2` | current frame (pre-grasp) / cached frame-0 (post-grasp) | **same as v16, untouched** | episode frame-0 (captured at step 0) |
| `camera3` | empty pad | empty pad | empty pad |

The cam2 invariant is enforced by two structural guarantees:
- **`.clone()` in the prep dataset's pre-grasp branch** breaks the shared
  reference between cam1 and cam2, so any future mutation of cam1 cannot
  corrupt cam2.
- **Hard runtime assertion** at the end of the v16 block raises
  `RuntimeError` if cam2 is missing or not a `Tensor` — fails fast on a
  frame-0 cache miss rather than letting the slot head read an empty pad.

The cam-drop logic itself is `CameraDropAugmenter` in
`scripts/eval3_dataset_prep.py`. It runs inside `Eval3PrepDataset.__getitem__`
**after** the cam2 assignment, so its replacement of cam1 can never leak into
cam2.

## 3. The training run

| | |
|---|---|
| Launcher | `scripts/run_eval3_smolvla_v17_real_data_slot_train.sh` |
| Default corpus | 9 real `dataset_v4_*` + 9 synthetic `dataset_v3_synth_pinned_idood_*_3` (18 datasets, ~212k frames / 449 episodes) |
| Optional add-on (v2) | `EVAL3_V17_INCLUDE_V2=1` → also concat 9 legacy `dataset_v2_*` repos (~5k frames extra real signal) |
| Optional add-on (algvr) | `EVAL3_V17_INCLUDE_ALGVR=1` → also concat **all local** `dataset_v5_synth_algvr_*_full` (up to 102 datasets, ~296k frames / ~594 episodes — algvr.com conference identity diversity, see §3.5) |
| Steps / batch | default 10,000 / 128; full-recipe target 50,000 / 256 |
| Init | fresh from `lerobot/smolvla_base` |
| Output | `/ephemeral/outputs/train/eval3_v17_camdrop` |
| wandb | project `eval3-v17-camdrop` |

### Corpus modes (one base mode at a time; `INCLUDE_V2` / `INCLUDE_ALGVR` are additive)

| Mode | Env | Datasets | Frames | Use case |
|---|---|---|---|---|
| **Default** | (none) | 9 real v4 + 9 synth v3_3 | ~212k | Production recipe |
| Real-only | `EVAL3_V17_NO_SYNTH=1` | 9 real v4 | ~31k | Tighter distribution; smoke runs |
| Synth-only | `EVAL3_V17_SYNTH_ONLY=1` | 9 synth v3_3 | ~180k | Isolate synth contribution |
| + v2 add-on | `EVAL3_V17_INCLUDE_V2=1` | (above) + 9 legacy v2 | +~5k | Extra real signal |
| + algvr add-on | `EVAL3_V17_INCLUDE_ALGVR=1` | (above) + up to 102 local `dataset_v5_synth_algvr_*_full` | +~296k | Identity diversity — 34 academic faces from `algvr-conference.json` warped onto v5 charuko boards (see §3.5) |
| + pins30 add-on | _**manual `EVAL3_LOCAL_REPOS` for now**_ — see §3.6 | (above) + up to 90 local `dataset_v5_synth_*_full_pins30` | varies with M | Globally-recognisable identity diversity — 30 Pins celebs (Ronaldo, Swift, Bezos, …), quality-ranked and B&W-penalised, warped onto v5 charuko boards (see §3.6). Launcher toggle is an open item. |

### Launch / relaunch

Smoke (~5 min on MPS):

```bash
EVAL3_V17_NO_SYNTH=1 \
  EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=12 EVAL3_WANDB=0 \
  EVAL3_POLICY_DEVICE=mps \
  EVAL3_TRAIN_OUT=/tmp/eval3_v17_smoke EVAL3_JOB_NAME=eval3_v17_smoke \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh --log_freq=2
```

Full run on CUDA (mirrors v16 50k recipe):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EVAL3_TRAIN_STEPS=50000 EVAL3_BATCH=256 EVAL3_SAVE_FREQ=1000 EVAL3_WANDB=1 \
EVAL3_TRAIN_OUT=/ephemeral/outputs/train/eval3_v17_camdrop_50k \
EVAL3_JOB_NAME=eval3_v17_camdrop_50k \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh --log_freq=100
```

- `expandable_segments:True` is required at batch 256 — same constraint as
  v16. If it OOMs, drop to `EVAL3_BATCH=128`.

### Camera-1 drop knobs (defaults)

| Env var | Default | Effect |
|---|---|---|
| `EVAL3_CAM1_DROP` | `1` | master switch (set to 0 to fall back to v16 behavior) |
| `EVAL3_CAM1_DROP_EPISODE_P` | `0.35` | per-episode drop probability (dominant signal) |
| `EVAL3_CAM1_DROP_FRAME_P` | `0.10` | per-frame iid drop (on non-dropped episodes, pre-grasp) |
| `EVAL3_CAM1_DROP_POSTGRASP_MULT` | `3.0` | multiplier for post-grasp per-frame rate (→ 0.30 post-grasp) |
| `EVAL3_CAM1_DROP_NOISE_MEAN` | `0.5` | Gaussian mean (post-normalize) |
| `EVAL3_CAM1_DROP_NOISE_STD` | `0.25` | Gaussian std |
| `EVAL3_CAM1_DROP_EPOCH_WINDOW` | `5000` | steps per per-episode flag reroll |

### Lever A — h_slot token repeat (`EVAL3_SLOT_TOKEN_REPEAT`, default K=4 in v17)

Appends the same `h_slot` prefix token **K times** instead of once. Boosts the
action expert's attention budget for the slot signal from ~1/n_prefix to
~K/n_prefix without changing `slot_proj` — one adapter is shared by all K
copies, so the *content* of every replicated token is identical. `K=1`
reproduces v16 bit-exactly. K=4–8 is the recommended range; K=4 is the v17
default.

Verify at launch:
- `installed SlotClassifier (... token_repeat=K=4)` in the patch install log
- `h_slot appended K=4 times` in the v16 prefix check log

The slot CE loss is computed on the same `_last_slot_logits` regardless of K
(the K copies only affect the action expert's cross-attention; slot
supervision is unchanged).

### Recommended ablations

```bash
# Stress test: every cam1 dropped (lang + frame-0 + proprio ONLY)
EVAL3_CAM1_DROP_EPISODE_P=1.0 EVAL3_CAM1_DROP_FRAME_P=0.0 \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh

# Per-frame only (no full-episode drops, just sensor flicker)
EVAL3_CAM1_DROP_EPISODE_P=0.0 EVAL3_CAM1_DROP_FRAME_P=0.20 \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh

# v16 baseline (cam-drop OFF, everything else identical)
EVAL3_CAM1_DROP=0 \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh

# Identity-diversity bump: add the 102 algvr-conference synth datasets on top
EVAL3_V17_INCLUDE_ALGVR=1 \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh

# Identity diversity + everything else on (real + v3 synth + v2 + algvr)
EVAL3_V17_INCLUDE_V2=1 EVAL3_V17_INCLUDE_ALGVR=1 \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh

# Pins-quality slate added manually (§3.6); 30 globally-recognisable celebs
EVAL3_LOCAL_REPOS="$(ls -d datasets/dataset_v5_synth_*_full_pins30 \
    | sed 's@datasets/@RobotLearningVLA/@' | paste -sd, -)" \
EVAL3_EXTRA_REPOS="$EVAL3_LOCAL_REPOS" \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh
```

### 3.5 algvr-conference synthetic slate (`EVAL3_V17_INCLUDE_ALGVR=1`)

A 4th, fully **opt-in + local-only** slate that injects 34 academic faces from
[algvr.com/conference](https://algvr.com/conference/) into the corpus. Useful
when you want to test whether the slot bottleneck + cam1 drop still learn
the language-grounded routing decision under a much broader identity
distribution than the 3 TOY celebs (Swift / Obama / LeCun) cover.

**Generated by**: `scripts/run_eval3_synth_algvr_dataset_gen.sh`
(re-runnable; see also `docs/eval3/charuco_pipeline.md` for the underlying
warp pipeline).

| | |
|---|---|
| Pool | `datasets/algvr-conference.json` (34 organizers + invited speakers, 1–4 photos each) |
| Sources | `datasets/dataset_v5_charuko_{left,middle,right}_full` (the 10-ep cross-product captures from `tools/eval3_v5_concat_pairs.py`) |
| Output naming | `dataset_v5_synth_algvr_<celeb_slug>_<position>_full` |
| Default scale | 34 × 3 = **102 datasets**, ~594 episodes, ~296,445 frames, **~1.55 GB** (generator default M=3 distractor scenes per target photo) |
| Task strings | One per dataset, `"Place the coke on <Canonical Name>"` (e.g. `"Place the coke on Marc Pollefeys"`) |
| Schema | LeRobot v3.0 — identical to v4 / v3 synth: 6-DOF `action`/`observation.state`, 480×640 `observation.images.front`, fps=30 |
| Push to Hub | **No** by default — the launcher's slate-4 glob picks them up from `./datasets/` and the v17 trainer adds them to `EVAL3_LOCAL_REPOS` so `eval3_concat_patch` reads from disk. |

**To regenerate** (e.g. with a different M, or after editing the pool JSON):

```bash
# Default: 102 datasets, ~1.5 GB, ~30 min on 8 workers
./scripts/run_eval3_synth_algvr_dataset_gen.sh

# Subset (e.g. only ETH organizers, all 3 slots)
EVAL3_ALGVR_CELEBS=marc_pollefeys,jeannette_bohg,xi_wang,ayse_johannes,roy_yang,alexey_gavryushin \
  ./scripts/run_eval3_synth_algvr_dataset_gen.sh

# More distractor variety per target photo (M=10 → ~5 GB)
EVAL3_ALGVR_M=10 EVAL3_ALGVR_OVERWRITE=1 \
  ./scripts/run_eval3_synth_algvr_dataset_gen.sh
```

The v17 launcher **discovers them by glob at runtime**
(`datasets/dataset_v5_synth_algvr_*_full`) — no need to edit the launcher
when you regenerate with different scope. If the toggle is set but no
datasets are found, the launcher warns and continues without them.

### 3.6 Pins-quality synthetic slate (`EVAL3_V17_INCLUDE_PINS30=1`)

A 5th, fully **opt-in + local-only** slate that injects 30 globally-recognisable
celebrities (Pins-Face-Recognition top-30) onto the v5\_charuko\_full charuco
boards. Useful when you want identity diversity beyond the 3 TOY celebs (Swift /
Obama / LeCun) and beyond the 34 academic faces in the algvr slate, with the
photos already **quality-ranked and B&W-penalised** so the print images are
high-res colour portraits rather than the raw Pins mix (median 185 px, ~20%
B&W / multi-face).

**Generated by**: `tools/eval3_synth_pins_dataset_gen.py` driven by
`tools/pins_quality_filter.py`'s output JSON.

| | |
|---|---|
| Pool | `datasets/pins-face-recognition-top30-quality.json` (30 celebs × top-10 photos each, ranked best-first) |
| Sources | `datasets/dataset_v5_charuko_{left,middle,right}_full` (same as §3.5) |
| Output naming | `dataset_v5_synth_<celeb_slug>_<position>_full_pins30` |
| Default scale | 30 × 3 = **90 datasets**, N×M = 10×50 = **500 episodes per dataset** ⇒ ~45k episodes total (size depends on M; with M=3 it's ~525 episodes total / ~1.4 GB, matching the algvr-default footprint) |
| Task strings | One per dataset, `"Place the coke on <Canonical Name>"` (e.g. `"Place the coke on Cristiano Ronaldo"`) |
| Schema | LeRobot v3.0, identical to the algvr slate |
| Push to Hub | **No** by default — keep local under `./datasets/` so `eval3_concat_patch` reads from disk via `EVAL3_LOCAL_REPOS`. Add `--push-to-hub` if you want them on `RobotLearningVLA/`. |

#### Pool prep — rebuild the quality JSON when the source pool or scorer changes

```bash
python tools/pins_quality_filter.py \
    --pool-json datasets/pins-face-recognition-top30.json \
    --out-json  datasets/pins-face-recognition-top30-quality.json
```

The scorer is:

```
raw = face_area_frac * 100                                            # cap 100 ("big face")
    + clamp((long_edge - 300) / (700 - 300), 0, 1) * 100               # cap 100 ("high res")
    + 10 if portrait (h >= w)
    + 10 if face centered (face_cx within 30% of img_cx horizontally)
color_factor = 0.5  if mean HSV saturation <= 30        ("B&W penalty")
               1.0  if mean HSV saturation >= 60
               linear interp otherwise
score = raw * color_factor
```

Hard filters (drop the photo): exactly 1 Haar face (or ≥ 50% dominance on
double-detect), `face_area_frac >= 0.15`, `long_edge >= 300 px`, `h/w >= 0.85`.
Knobs at the top of the script: `HARD_FACE_AREA_MIN`, `HARD_LONG_EDGE_MIN`,
`RES_SCORE_MIN_PX` / `MAX_PX`, `SAT_BW_THRESHOLD`, `SAT_FULL_COLOR_THRESHOLD`,
`BW_SCORE_FACTOR` (raise from 0.5 to soften the B&W penalty).

Known scorer limits:
- **Haar misses partial / occluded faces** — a second face at the edge of the
  frame can slip past the single-face filter. Swap in MediaPipe Face Detection
  or RetinaFace in `_detect_faces` if it bites.
- **No text-overlay detection** — meme images with caption text rank as normal
  photos (this is what gives Morgan Freeman's new top-1 the "AVE TO FEAR IS…"
  caption). Add Tesseract / PaddleOCR at the top of `_score_one_photo` if
  needed.

#### Generate — full pins30 sweep on v5 sources

```bash
# Recommended: matches the algvr slate's source/output conventions
python tools/eval3_synth_pins_dataset_gen.py \
    --pool-json datasets/pins-face-recognition-top30-quality.json \
    --source-prefix dataset_v5_charuko_ --source-suffix _full \
    --output-prefix dataset_v5_synth_  --output-postfix _full \
    --output-suffix pins30 \
    --target-celebs all --target-positions left,middle,right \
    --max-photos-per-celeb 10 --distractors-per-target-photo 50 \
    --n-workers $(nproc) --vcodec h264 --seed 42 --overwrite
```

For a quick smoke (1 celeb × 1 slot, ~5 min):

```bash
python tools/eval3_synth_pins_dataset_gen.py \
    --pool-json datasets/pins-face-recognition-top30-quality.json \
    --source-prefix dataset_v5_charuko_ --source-suffix _full \
    --output-prefix dataset_v5_synth_  --output-postfix _full \
    --output-suffix pins30 \
    --target-celebs cristiano_ronaldo --target-positions left \
    --max-photos-per-celeb 4 --distractors-per-target-photo 5 \
    --overwrite
```

Or via the existing wrapper script (uses v3 charuco sources, default outputs
`dataset_v3_synth_pins30_<celeb>_<pos>_2` instead of the `_full` naming):

```bash
./scripts/run_eval3_synth_pins_dataset_gen.sh --dry-run            # plan only
EVAL3_PINS_PUSH_TO_HUB=1 ./scripts/run_eval3_synth_pins_dataset_gen.sh
```

The wrapper auto-detects `datasets/pins-face-recognition-top30-quality.json`
when present and falls back to the unfiltered top-30 JSON otherwise — set
`EVAL3_PINS_POOL_JSON` to override.

#### Knobs (CLI-only on the python tool; wrapper env-var equivalents in parens)

| Flag (CLI) | Wrapper env | Default | Effect |
|---|---|---|---|
| `--max-photos-per-celeb` | `EVAL3_PINS_MAX_PHOTOS` | 10 | Top-N photos per celeb from the quality JSON. Drop to 4 for max quality, raise for more variety. |
| `--distractors-per-target-photo` | `EVAL3_PINS_DISTRACTORS_PER_TARGET` | 50 | Distractor scenes per target photo (size scales as `N × M`). |
| `--output-suffix` | `EVAL3_PINS_OUTPUT_SUFFIX` | `pins30` | Name tag; bump if you want pins runs not to clobber each other. |
| `--n-workers` | `EVAL3_PINS_WORKERS` | `nproc` | One worker per output dataset. On Apple Silicon use 8; on Brev box use full nproc. |
| `--push-to-hub` | `EVAL3_PINS_PUSH_TO_HUB` | off | Upload + create `v3.0` tag per dataset. |
| `--overwrite` | `EVAL3_PINS_OVERWRITE` | off | Replace existing output dirs. |
| `--source-prefix` / `--source-suffix` | (CLI only) | `dataset_v3_charuco_` / `_2` | Switch trajectory base — use `dataset_v5_charuko_` / `_full` for v17. |

#### Train-time wiring (NOT YET in the launcher)

The v17 launcher does NOT currently grep for `dataset_v5_synth_*_full_pins30`
the way it does for the algvr slate. Until that wiring lands, point at the
slate manually:

```bash
EVAL3_LOCAL_REPOS="$(ls -d datasets/dataset_v5_synth_*_full_pins30 \
    | sed 's@datasets/@RobotLearningVLA/@' | paste -sd, -)" \
EVAL3_EXTRA_REPOS="$EVAL3_LOCAL_REPOS,<existing extras…>" \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh
```

See §9 "Open items" — there's a placeholder for adding a dedicated
`EVAL3_V17_INCLUDE_PINS30=1` toggle that mirrors `EVAL3_V17_INCLUDE_ALGVR=1`.

## 4. Monitoring

### One-time startup diagnostics (must appear in first ~100 log lines)

```
eval3_concat_patch: camera-1 drop enabled — ep_p=0.350 frame_p=0.100 post_mult=3.0 noise=N(0.50,0.25) epoch_window=5000
eval3_concat_patch: dataset_v4_taylor_left  ... cam_drop=True  ep_filter=None  ← per repo
[eval3_slot_bottleneck] installed SlotClassifier (... frame0=True, ce_pregrasp_only=True)
[eval3_slot_bottleneck] v16 prefix check: 3 images embedded, n_img=192 n_cams=3 tok_per_cam=64, slot reads token slice [64:128]
[eval3_slot_bottleneck] v16 mask check: ... post-grasp samples inside CE mask=0
```

In the v16 prefix check, watch the per-camera (mean, std):

| Camera | Healthy v17 stats | Means |
|---|---|---|
| cam1 (current/noised) | std ∈ [0.50, 0.70] | mix of real (0.65-0.75) and noise (0.50-0.55) frames |
| cam2 (frame-0, untouched) | std ∈ [0.75, 0.80] | always real |
| cam3 (empty pad) | std = 0.0, mean = -1.0 | always empty |

If cam2 std drops below 0.5 or cam2 mean approaches -1.0, the invariant is
broken — STOP the run and investigate.

### wandb / log metrics

- `slot_acc` — should climb past 0.85 by ~step 2k (same trajectory as v16).
- `slot_loss` — falls below `ln 3 ≈ 1.10`.
- `slot_ce_n` — ~25-30% of batch (pre-grasp fraction).
- `loss` — main flow-matching loss; expect the same trajectory as v16, possibly
  slightly higher initial values due to cam1 noise but converging within ~1k
  extra steps.

### Health bands

- **slot_acc flat at 0.33** → slot head broken (LayerNorm fix missing OR cam2
  empty). Check the prefix check log.
- **slot_acc climbing but loss plateauing way above v16** → cam-drop too
  aggressive; consider lowering `EVAL3_CAM1_DROP_EPISODE_P` to 0.25.
- **cam1 std ≈ noise std (0.25) in prefix check** → a batch coincidentally
  drew all-dropped episodes; check the next prefix-check log; persistent =
  bug.

### 4.1 Held-out validation watcher (`tools/eval3_val_watcher.py`)

A single CLI tool that scores SmolVLA checkpoints on user-supplied held-out
LeRobot-format datasets, in a separate process so it doesn't slow training
down. Computes four metrics (overall + per-slot + per-repo + per-joint
action-MAE breakdown) and emits a JSONL line per checkpoint to
`$TRAIN_OUT/val_metrics.jsonl`. Optional wandb sidecar for live curves
alongside the train run.

#### TL;DR

**Add val to a live training run** (drop the two `EVAL3_VAL_*` lines into
your usual launcher invocation):

```bash
EVAL3_VAL_WATCH=1 \
EVAL3_VAL_REPOS=org/val_taylor_left,org/val_taylor_middle,org/val_taylor_right \
EVAL3_VAL_DEVICE=mps \
./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh
```

**Score one checkpoint right now**, no launcher needed:

```bash
EVAL3_VAL_REPOS=org/val_taylor_left,org/val_taylor_middle,org/val_taylor_right \
python tools/eval3_val_watcher.py \
  --policy-path RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k \
  --once
```

Each prints a stdout summary and writes JSONL. Full env-var reference + 4
common workflows + JSONL reading examples below.

#### How it works

The watcher runs in **one of three modes** depending on which flags you
pass. All three share the env-var contract, the JSONL schema, and produce
the same per-checkpoint stdout summary.

| Mode | Trigger | When to use |
|---|---|---|
| **Continuous watch** | `--train-out <dir>` (no `--once`, no `--policy-path`) | Background alongside live training. Polls `<dir>/checkpoints/` and scores each new step until idle timeout. The launcher uses this mode when `EVAL3_VAL_WATCH=1`. |
| **Single-shot from a train dir** | `--train-out <dir> --once` | Score the latest checkpoint that has landed under `<dir>/checkpoints/` and exit. Useful after a crash or for backfilling. |
| **Direct checkpoint** | `--policy-path <ckpt-dir-or-HF-repo>` | Score one specific checkpoint with no polling, no train-out. Accepts a local `pretrained_model` dir OR a Hub repo id. |

For every mode the watcher:

1. **Auto-detects** the checkpoint's `rename_map` from `train_config.json`
   (so v16 single-cam and v17 frame-0 two-cam checkpoints both work without
   flags).
2. **Loads** each val repo via `LeRobotDataset` (from Hub or, via
   `EVAL3_VAL_LOCAL_REPOS`, from `./datasets/<name>/`).
3. **Samples** `EVAL3_VAL_EPISODES_PER_REPO` episodes per repo (even stride),
   then `EVAL3_VAL_FRAMES_PER_EPISODE` frames per episode (uniform stride).
4. For each sampled frame: predicts an action chunk under all 3 prompts
   (the correct celebrity + two distractors), reads the slot head's logits,
   and computes the per-frame metrics below.
5. Aggregates **overall + per-slot + per-repo + per-joint action-MAE**;
   writes one JSONL record per checkpoint; optionally logs to wandb.

**Four metrics per checkpoint** (overall + per-slot + per-repo breakdown):

| Metric | What it tells you |
|---|---|
| `slot_acc` | `argmax(model._last_slot_logits) == target_position`. Direct slot-head accuracy on **unseen** scenes — the language-binding test. Healthy ≥ 0.85 by step 5k. |
| `action_mae` / `action_mae_per_joint` | L1 between predicted action and recorded action at the same frame, joint-wise mean + per-joint breakdown. The per-joint table surfaces *which joint* is wrong (wrist_roll spikes = wrong-slot trajectory; gripper spikes = grasp-timing mismatch). |
| `prompt_nearest_accuracy` | For each frame, predict actions under all 3 prompts; check whether the **correct-prompt** prediction is closest L2 to the recorded action. Direct "does the language drive the trajectory" test. ≥ 0.75 is healthy. |
| `cross_prompt_delta` | Mean pairwise L2 across the 3 prompts. Sanity gate that the policy reacts to language at all. ≥ 20° healthy; < 5° = prompt-collapsed. |

#### Env-var contract

| Env var | Default | Purpose |
|---|---|---|
| `EVAL3_VAL_WATCH` | `0` | Master switch (set to `1` to background the watcher from the launcher). |
| `EVAL3_VAL_REPOS` | (required) | Comma-separated list of LeRobot repos to score on. Repo names ending `_left_*`/`_middle_*`/`_right_*` get auto-assigned a target slot. |
| `EVAL3_VAL_LOCAL_REPOS` | (none) | Subset of `EVAL3_VAL_REPOS` to load from `./datasets/<name>/` (mirrors `EVAL3_LOCAL_REPOS`). |
| `EVAL3_VAL_EPISODES_PER_REPO` | `3` | Episodes sampled per val repo (even stride over the dataset's episode range). |
| `EVAL3_VAL_FRAMES_PER_EPISODE` | `30` | Frames per episode (uniform stride). |
| `EVAL3_VAL_PROMPTS` | (auto) | JSON `{slug: prompt_template}` override; otherwise defaults to the three eval celebrities. |
| `EVAL3_VAL_DEVICE` | inherits `EVAL3_POLICY_DEVICE` | Per-watcher device override (e.g. CPU/MPS while training holds CUDA). |
| `EVAL3_VAL_POLL_SEC` | `60` | Checkpoint poll interval. |
| `EVAL3_VAL_IDLE_SEC` | `600` | Auto-stop after this many seconds of no new checkpoint. |
| `EVAL3_VAL_OUT` | `<train_out>/val_metrics.jsonl` | JSONL output path. |
| `EVAL3_VAL_SEED` | `0` | Seed for episode/frame sampling. |

#### CLI flag reference

Every env var above has a matching CLI flag. **CLI flags override env
vars**, env vars override built-in defaults. Useful when you want to invoke
the watcher with one-off settings without polluting your shell environment.

| CLI flag | Env-var equivalent |
|---|---|
| `--train-out PATH` | (no env equivalent — required for watch/--once modes) |
| `--policy-path STR` | (no env equivalent — direct-checkpoint mode) |
| `--once` | (flag-only) |
| `--val-repos REPO [REPO …]` | `EVAL3_VAL_REPOS` |
| `--val-local-repos REPO [REPO …]` | `EVAL3_VAL_LOCAL_REPOS` |
| `--episodes-per-repo N` | `EVAL3_VAL_EPISODES_PER_REPO` |
| `--frames-per-episode N` | `EVAL3_VAL_FRAMES_PER_EPISODE` |
| `--device cpu/mps/cuda` | `EVAL3_VAL_DEVICE` |
| `--poll-sec N` | `EVAL3_VAL_POLL_SEC` |
| `--idle-sec N` | `EVAL3_VAL_IDLE_SEC` |
| `--seed N` | `EVAL3_VAL_SEED` |
| `--final-step N` | (no env equivalent — usually set by the launcher) |
| `--wandb` | `EVAL3_VAL_WANDB=1` |
| `--wandb-project NAME` | `EVAL3_VAL_WANDB_PROJECT` |
| `--wandb-name NAME` | `EVAL3_VAL_WANDB_NAME` |

`python tools/eval3_val_watcher.py --help` prints the same list with
descriptions.

#### Where outputs land

| File | Contents |
|---|---|
| `$EVAL3_VAL_OUT` (default `<train_out>/val_metrics.jsonl`) | Always written. One header line + one record per evaluated checkpoint. The durable record. |
| Stdout (or watcher log when backgrounded) | Human-readable per-checkpoint summary table. Tail this during a run for live feedback. |
| `outputs/train/logs/<JOB>_val_watch.log` | When the launcher backgrounds the watcher, this is where its stdout goes (via `nohup`). |
| Wandb project `<EVAL3_WANDB_PROJECT>` run `<JOB>_val` | When `EVAL3_VAL_WANDB=1`. Step axis = checkpoint step; metric keys = `val/slot_acc`, `val/action_mae`, `val/per_slot/<slot>/<metric>`, `val/action_mae_per_joint/<joint>`. |

#### Wandb sidecar (`EVAL3_VAL_WANDB=1`)

Opens a **separate wandb run** in the same project as training, name = `<EVAL3_JOB_NAME>_val`. Logs `val/slot_acc`, `val/action_mae`, `val/action_mae_per_joint/<joint>`, `val/prompt_nearest_accuracy`, `val/cross_prompt_delta`, plus a `val/per_slot/{left,middle,right}/*` breakdown — all with the **checkpoint step as the x-axis**. In the wandb UI, multi-select train + val runs to overlay curves on the same step axis. No coordination needed with the training process; the sidecar is fully self-contained.

#### JSONL output schema

Header line + one record per evaluated checkpoint. Example record:

```json
{
  "step": 5000,
  "checkpoint": "outputs/train/eval3_v17.../checkpoints/005000/pretrained_model",
  "wall_time_s": 23.7,
  "n_frames_evaluated": 810,
  "overall": {
    "slot_acc": 0.87,
    "action_mae": 2.31,
    "action_mae_per_joint": {"shoulder_pan": 1.82, "wrist_roll": 4.20, ...},
    "prompt_nearest_accuracy": 0.83,
    "cross_prompt_delta": 24.6
  },
  "per_slot": {"left": {...}, "middle": {...}, "right": {...}},
  "per_repo": [{"repo": "...", "slot_acc": 0.88, ...}, ...]
}
```

#### Common workflows

**A. Live val on a running training (launcher hook)** — the most common
mode. Both env vars must be set; the launcher backgrounds the watcher and
moves on with training.

```bash
EVAL3_VAL_WATCH=1 \
EVAL3_VAL_REPOS=org/val_taylor_left,org/val_taylor_middle,org/val_taylor_right \
EVAL3_VAL_EPISODES_PER_REPO=3 EVAL3_VAL_FRAMES_PER_EPISODE=30 \
EVAL3_VAL_DEVICE=mps \
./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh
```

*Tip:* if training runs on CUDA, set `EVAL3_VAL_DEVICE=mps` or `cpu` so the
watcher doesn't fight the training process for GPU memory.

**B. Backfill val on a completed training run** — point at a finished
`<train_out>` dir to score every checkpoint in one pass.

```bash
# Score the latest checkpoint and exit
EVAL3_VAL_REPOS=org/val_repo_a,org/val_repo_b \
python tools/eval3_val_watcher.py \
  --train-out outputs/train/some_finished_job \
  --once

# Score ALL checkpoints, one at a time, then exit (no polling delay).
# The watcher discovers each checkpoint in order and stops when there are
# no more new ones for EVAL3_VAL_IDLE_SEC (default 600s).
EVAL3_VAL_REPOS=org/val_repo_a,org/val_repo_b \
EVAL3_VAL_POLL_SEC=1 EVAL3_VAL_IDLE_SEC=30 \
python tools/eval3_val_watcher.py \
  --train-out outputs/train/some_finished_job
```

**C. Score one specific checkpoint** — point at any HF repo or local
`pretrained_model` directory. No training-out needed.

```bash
EVAL3_VAL_REPOS=org/val_repo_a,org/val_repo_b \
EVAL3_VAL_OUT=outputs/eval/one_off.jsonl \
python tools/eval3_val_watcher.py \
  --policy-path RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k \
  --once
```

**D. Compare checkpoints offline** — call mode C twice (or once with
`--train-out`) and diff the JSONL.

```bash
for ckpt in v17-step10k v17-step25k v17-step50k; do
  EVAL3_VAL_REPOS=... EVAL3_VAL_OUT=outputs/eval/${ckpt}.jsonl \
  python tools/eval3_val_watcher.py --policy-path org/${ckpt} --once
done
```

#### Val dataset naming convention (REQUIRED for `slot_acc`)

The watcher derives the **ground-truth slot** for each val repo from the
repo name via the same regex used by training (`_slot_from_repo` in
`scripts/eval3_concat_patch.py`):

| Repo name contains | Inferred `target_position` |
|---|---|
| `_left_` or trailing `_left` / `_left_full` | `0` |
| `_middle_` or trailing `_middle` / `_middle_full` | `1` |
| `_right_` or trailing `_right` / `_right_full` | `2` |
| none of the above | `None` (slot_acc reports as `null` for this repo's rows) |

So when you create your val datasets, name them like
`dataset_v6_val_taylor_left`, `dataset_v6_val_taylor_middle`, etc. If you
can't follow this convention, pass `EVAL3_VAL_PROMPTS` to map identities and
the other three metrics (`action_mae`, `prompt_nearest_accuracy`,
`cross_prompt_delta`) still work — only `slot_acc` will be `null`.

The watcher also derives the **identity** (Swift / LeCun / Obama) from the
repo name (`taylor`/`swift`, `yann`/`lecun`, `barack`/`obama` substring
matches). Identity is what determines which of the 3 prompts is "the
correct prompt" for `prompt_nearest_accuracy`. If the repo name doesn't
match a known identity, that repo's `prompt_nearest_accuracy` reports as
`null`.

#### Custom prompts (`EVAL3_VAL_PROMPTS`)

By default the watcher uses three hardcoded prompts (`"Place the coke on
Taylor Swift"` etc.). Override with a JSON dict `{slug: prompt_template}`:

```bash
EVAL3_VAL_PROMPTS='{"swift":"Place the coke on Taylor Swift",
                    "lecun":"Place the coke on Yann LeCun",
                    "obama":"Place the coke on Barack Obama",
                    "messi":"Place the coke on Lionel Messi"}' \
EVAL3_VAL_REPOS=...messi_left...,...messi_middle...,...messi_right... \
python tools/eval3_val_watcher.py --policy-path <ckpt> --once
```

The slug keys (`swift`, `lecun`, `obama`, `messi`, …) must match what
`identity_from_repo` would return for your repo names. Add new identity
strings if you extend the celebrity pool — see `identity_from_repo` in
`tools/eval3_val_watcher.py`.

#### Reading the JSONL output

The file is one JSON object per line: line 1 is a header, lines 2…N are
per-checkpoint records. Examples:

```bash
# Show all step + slot_acc + action_mae pairs:
jq -c '. | select(.step != null) | {step, slot_acc: .overall.slot_acc, action_mae: .overall.action_mae}' \
  outputs/train/my_job/val_metrics.jsonl

# Plot slot_acc over training step in Python:
python - <<'PY'
import json, matplotlib.pyplot as plt
with open("outputs/train/my_job/val_metrics.jsonl") as f:
    records = [json.loads(L) for L in f if '"step"' in L]
steps = [r["step"] for r in records]
slot_acc = [r["overall"]["slot_acc"] for r in records]
plt.plot(steps, slot_acc); plt.xlabel("training step"); plt.ylabel("val slot_acc")
plt.savefig("val_slot_acc.png")
PY

# Compare per-slot performance at the latest checkpoint:
tail -1 outputs/train/my_job/val_metrics.jsonl | python -c "
import sys, json
rec = json.loads(sys.stdin.read())
for slot, m in rec['per_slot'].items():
    print(f'{slot:6s} n={m[\"n_frames\"]:3d}  slot_acc={m[\"slot_acc\"]}  mae={m[\"action_mae\"]}')"
```

For wandb users, set `EVAL3_VAL_WANDB=1` and read curves directly in the
wandb UI — see the Wandb sidecar subsection above.

#### Expected metric trajectories

Reference smoke (4 steps, batch 2, MPS, fresh init — i.e., effectively random):

- `slot_acc` ≈ chance (0.33). Wandering high or low depending on the random init's argmax bias.
- `action_mae` ≈ 10-12° / joint. `wrist_roll` typically the largest at 15-25° because it's the high-variance joint.
- `prompt_nearest_accuracy` ≈ 0.40-0.55. Above 1/3 chance because some early gradients already mildly bias the language path.
- `cross_prompt_delta` ≈ 15-25°. The policy reacts to prompt from step 1.

For a converged 50k v17 checkpoint, expect:

- `slot_acc` ≥ 0.90
- `action_mae` ≤ 3°
- `prompt_nearest_accuracy` ≥ 0.85
- `cross_prompt_delta` ≥ 25°

#### Smoke-testing the watcher

Three concentric tests that together cover unit logic, single-checkpoint
scoring, and end-to-end interaction with a real training run. Run in order
before touching shared infra (CI, the deploy box, etc.).

**(1) Unit tests** — pure logic, no SmolVLA load, ~5 s on CPU. Covers
slot/identity regexes, train_config rename-map detection (v16 vs v17),
sample-frame determinism + bounds, image/action coercion, summary
aggregation, lazy JSONL header, env+CLI precedence, picklability of
parsers. 57 checks across W1-W14.

```bash
python tools/eval3_val_watcher_unit_tests.py
# Expected tail: "RESULTS: 57 passed / 0 failed"
```

**(2) Single-shot `--once` against an existing checkpoint** — exercises the
full policy load + 3-prompt inference + metric computation + JSONL writer
on real data. Uses a v4 training repo as stand-in val data; needs an
already-trained checkpoint (or run a 2-step train first to seed one).

```bash
EVAL3_VAL_REPOS=RobotLearningVLA/dataset_v4_taylor_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_barack_right \
EVAL3_VAL_EPISODES_PER_REPO=2 EVAL3_VAL_FRAMES_PER_EPISODE=4 EVAL3_VAL_DEVICE=mps \
python tools/eval3_val_watcher.py \
  --train-out outputs/train/eval3_v17_K4_smoke \
  --once
```

Expect a stdout block of the form:

```
[val] step=N  n_frames=24  wall=~70s  slot_acc=...  action_mae=...  prompt_nearest_acc=...  cross_prompt_delta=...
    [left  ] n=8  slot_acc=...  action_mae=...  prompt_nearest_acc=...  cross_prompt_delta=...
    [middle] n=8  ...
    [right ] n=8  ...
```

and a JSONL header + 1 record written to `<train_out>/val_metrics.jsonl`.

**(3) End-to-end with the launcher hook** — verifies the launcher correctly
backgrounds the watcher, the watcher discovers new checkpoints as they
land, and the lazy JSONL header doesn't race with `lerobot_train`'s
output-dir validator.

```bash
rm -rf outputs/train/eval3_v17_val_smoke outputs/train/eval3_v17_val_smoke.log \
       outputs/train/logs/eval3_v17_val_smoke_val_watch.log

EVAL3_V17_NO_SYNTH=1 \
  EVAL3_TRAIN_STEPS=4 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=2 EVAL3_WANDB=0 \
  EVAL3_POLICY_DEVICE=mps EVAL3_WARMUP_STEPS=1 \
  EVAL3_VAL_WATCH=1 EVAL3_VAL_DEVICE=mps \
  EVAL3_VAL_REPOS=RobotLearningVLA/dataset_v4_taylor_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_barack_right \
  EVAL3_VAL_EPISODES_PER_REPO=1 EVAL3_VAL_FRAMES_PER_EPISODE=3 \
  EVAL3_VAL_POLL_SEC=15 EVAL3_VAL_IDLE_SEC=300 \
  EVAL3_TRAIN_OUT=outputs/train/eval3_v17_val_smoke \
  EVAL3_JOB_NAME=eval3_v17_val_smoke \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh --log_freq=1 \
  > outputs/train/eval3_v17_val_smoke.log 2>&1
```

Verify (after the launcher returns + ~3 min for the watcher to drain):

```bash
# (a) launcher backgrounded the watcher
grep "val watcher backgrounded" outputs/train/eval3_v17_val_smoke.log

# (b) training completed all steps
grep -E "step:[0-9]+ " outputs/train/eval3_v17_val_smoke.log | tail

# (c) watcher discovered checkpoints + wrote metrics
tail -20 outputs/train/logs/eval3_v17_val_smoke_val_watch.log

# (d) JSONL contract: 1 header + 1 record per saved checkpoint
wc -l outputs/train/eval3_v17_val_smoke/val_metrics.jsonl
head -1 outputs/train/eval3_v17_val_smoke/val_metrics.jsonl | python -m json.tool

# (e) the watcher is no longer running (or will exit on its idle timer)
ps -ef | grep eval3_val_watcher | grep -v grep
```

Expected: launcher line present, training reached step 4 without errors,
watcher log shows per-checkpoint stdout summaries, JSONL has header +
records (the schema/v1 marker in line 1, `step:N` records following).

#### Known gotchas

- **macOS background process**: the launcher uses `nohup python … &`. On
  macOS, when the parent shell exits the orphan continues running but is
  no longer visible to that shell. If you re-invoke the launcher quickly,
  two watchers can race on the same `val_metrics.jsonl`. Kill stale ones
  via `pgrep -f eval3_val_watcher | xargs kill` before starting a fresh
  end-to-end smoke.
- **MPS policy load takes ~30s per checkpoint** — the watcher therefore lags
  behind training by `~30s + frames * ~2s` per checkpoint. With
  `EVAL3_SAVE_FREQ=1000` this means the watcher reports each metric
  roughly 1-2 minutes after the checkpoint lands.
- **Per-repo target_idx is None when the repo name doesn't match
  `_left_/_middle_/_right_`**. The watcher reports a warning at startup and
  emits `slot_acc=None` for those rows; other three metrics still compute.

## 5. Deploy

**v17 is a v16 checkpoint** for deploy purposes — same `policy.empty_cameras=1`,
same 2-camera rename_map, same auto-detect logic in `eval3_vla_deploy.py`. The
cam-drop augmentation is training-only; at inference the camera1 frame is
always the live capture.

### Preferred: deploy battery

```bash
EVAL3_V16_CKPT=<path-or-hf-repo-for-v17-ckpt> \
  ./scripts/run_eval3_deploy_battery.sh v16 --task='Place the coke on Taylor Swift'
```

(Yes, use the `v16` battery arm — v17 checkpoints have the same artifact
layout. Alternatively, add a `v17` case arm to the battery if you want a
dedicated entry.)

### Pre-flight (run before every hardware deploy)

```bash
python tools/eval3_check_deploy_command.py \
  --policy-pretrained-path <ckpt> \
  --rename-map '{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}' \
  --task 'Place the coke on Taylor Swift'
```

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: cam2 (observation.images.front_frame0) missing or invalid` | frame-0 cache miss for that episode. Check that the underlying LeRobotDataset can read frame 0 (delete cache + retry). The assertion fires by design — better a loud crash than silent slot corruption. |
| `slot_acc` flat at ~0.38 | LayerNorm fix missing from SlotClassifier — check `img_ln`/`lang_ln` exist (`scripts/eval3_smolvla_slot_bottleneck.py:120`). |
| cam1 std in prefix check ≈ 0 instead of ≈0.5 | cam1 is being replaced by empty pad rather than noise. Check `EVAL3_CAM1_DROP` env var is `1` in the launcher's environment. |
| OOM mid-run at batch 256 | same as v16; relaunch at `EVAL3_BATCH=128`. |
| `slot_ce_n=0` | grasp detection failed — check `EVAL3_GRASP_GRIP_DELTA`, gripper state column. |
| `EVAL3_LOCAL_REPOS` repo not found | one of the synth `_3` dataset dirs is missing under `./datasets/`. Either fetch them or use `EVAL3_V17_NO_SYNTH=1`. |
| `WARN EVAL3_V17_INCLUDE_ALGVR=1 but no datasets/dataset_v5_synth_algvr_*_full found` | The algvr slate is empty. Build it first: `./scripts/run_eval3_synth_algvr_dataset_gen.sh` (~30 min, ~1.5 GB local), or drop the toggle. |
| Algvr-included run has wildly different `loss` curve from default | Expected — algvr adds ~296k frames / 34 new identities, ~2.4× the v17 default frame budget. Adjust `EVAL3_TRAIN_STEPS` (or `EVAL3_BATCH`) if you want the same wall-clock to cover the same fraction of epochs. |
| `WARN EVAL3_VAL_WATCH=1 but EVAL3_VAL_REPOS is empty — watcher NOT started.` | Set `EVAL3_VAL_REPOS=...` alongside `EVAL3_VAL_WATCH=1`. The launcher refuses to start an unconfigured watcher. |
| Val watcher's `slot_acc=null` (or `None`) for every repo | Repo names don't match the `_left_/_middle_/_right_` regex. Rename val repos or accept that slot_acc is unavailable — the other 3 metrics still work. See §4.1 "Val dataset naming convention". |
| Val watcher's `prompt_nearest_accuracy=null` for some repos | Repo name doesn't match a known identity (`taylor`/`yann`/`barack`). Add the identity slug + prompt via `EVAL3_VAL_PROMPTS` JSON (see §4.1 "Custom prompts"). |
| Two `eval3_val_watcher` processes racing on the same JSONL | A previous launcher invocation left an orphan. Kill stale ones via `pgrep -f eval3_val_watcher \| xargs kill` before starting fresh. macOS `nohup` doesn't surface orphans in the new shell. |
| `RepositoryNotFoundError` from val watcher on one repo | The watcher logs a warning and **skips that repo**, continues with the rest. If you see all repos skipped, the JSONL won't get a record for that checkpoint. |

## 7. Pre-merge verification

The cam-drop pipeline has 73 unit checks + 3 end-to-end smokes passing:

```bash
# 73-check unit test (D1-D12 baseline + C1-C8 robustness + E1-E4 cam2 invariant)
python tools/eval3_v16_dataset_test.py

# 2-step end-to-end smoke (defaults)
EVAL3_V17_NO_SYNTH=1 \
  EVAL3_TRAIN_STEPS=2 EVAL3_BATCH=4 EVAL3_SAVE_FREQ=2 EVAL3_WANDB=0 \
  EVAL3_POLICY_DEVICE=mps \
  EVAL3_TRAIN_OUT=outputs/train/eval3_v17_premerge_smoke \
  EVAL3_JOB_NAME=eval3_v17_premerge_smoke \
  ./scripts/run_eval3_smolvla_v17_real_data_slot_train.sh --log_freq=1 \
  2>&1 | tee outputs/train/eval3_v17_premerge_smoke.log

# Verify the v16 prefix check shows cam2 std > 0.7
grep "v16 prefix check" outputs/train/eval3_v17_premerge_smoke.log
```

The val watcher has its own test suite (57 unit checks + 2 integration smokes):

```bash
# 57-check unit tests (W1-W14): regex coverage, rename-map detection,
# sample-frame determinism, image/action coercion, summary aggregation,
# lazy JSONL header, env+CLI precedence. Pure logic, ~5s on CPU.
python tools/eval3_val_watcher_unit_tests.py
# Expected tail: "RESULTS: 57 passed / 0 failed"

# --once smoke against an existing checkpoint (~70s on MPS).
EVAL3_VAL_REPOS=RobotLearningVLA/dataset_v4_taylor_left,RobotLearningVLA/dataset_v4_yann_middle,RobotLearningVLA/dataset_v4_barack_right \
EVAL3_VAL_EPISODES_PER_REPO=2 EVAL3_VAL_FRAMES_PER_EPISODE=4 EVAL3_VAL_DEVICE=mps \
python tools/eval3_val_watcher.py \
  --train-out outputs/train/eval3_v17_K4_smoke --once

# End-to-end smoke with the launcher hook (~3 min on MPS).
# Full recipe in §4.1 "Smoke-testing the watcher".
```

## 8. Key files

- `scripts/run_eval3_smolvla_v17_real_data_slot_train.sh` — this launcher
- `scripts/eval3_dataset_prep.py` — `CameraDropAugmenter` (class) + `.clone()` defense + cam2 assertion
- `scripts/eval3_concat_patch.py` — env-var parsing + per-repo augmenter wiring
- `scripts/eval3_smolvla_slot_bottleneck.py` — slot classifier + LayerNorm fix + camera2 slice (unchanged from v16)
- `scripts/eval3_vla_deploy.py` — closed-loop deploy (no cam-drop, no changes vs v16)
- `tools/eval3_v16_dataset_test.py` — 73-check robustness suite (D + C + E phases)
- `docs/eval3/v16_playbook.md` — companion playbook for the v16 slot bottleneck (architectural prerequisite)
- `scripts/run_eval3_synth_algvr_dataset_gen.sh` — builds the algvr-conference synth slate consumed by `EVAL3_V17_INCLUDE_ALGVR=1` (see §3.5)
- `tools/eval3_synth_pins_dataset_gen.py` — Pins-pool generator the algvr launcher wraps (accepts `--source-prefix` / `--source-suffix` / `--output-prefix` / `--output-postfix` so it can target v5 charuko sources)
- `scripts/run_eval3_synth_pins_dataset_gen.sh` — env-var wrapper around the same generator with v3 charuco defaults; auto-detects the quality JSON (§3.6)
- `tools/pins_quality_filter.py` — Haar-based scorer that produces the quality-ranked, B&W-penalised pool JSON (§3.6)
- `datasets/pins-face-recognition-top30-quality.json` — the ranked pool the pins30 slate samples from (regenerate via `pins_quality_filter.py`)
- `datasets/algvr-conference.json` — 34-person celeb pool the algvr slate samples from
- `tools/eval3_val_watcher.py` — held-out validation watcher (§4.1). Polls the train output for new checkpoints + scores each on user-configurable LeRobot-format datasets. Three modes: continuous watch, `--once`, direct `--policy-path`.
- `tools/eval3_val_watcher_unit_tests.py` — 57-check unit suite (W1-W14) for the val watcher: regex coverage, rename-map detection, frame sampling, summary aggregation, lazy JSONL header, env+CLI precedence.

## 9. Open items

- **Push the final v17 checkpoint to the Hub** once full-run results land
  (~10-50k steps). Update `EVAL3_V17_CKPT` deploy entry once available.
- **Build dedicated v17 val datasets** in LeRobot v3.0 format, named
  `_left_/_middle_/_right_` so the watcher (§4.1) auto-derives
  `target_position`. Until then, point `EVAL3_VAL_REPOS` at the held-out
  `dataset_v3_synth_pinned_idood_*_2` synth slate (NOT in the v17 training
  corpus, which uses `_3` — clean held-out set on the Hub) or at a small
  number of `dataset_v4_*` episodes as a stand-in.
- **Cam-drop sensitivity sweep**: ablate `EVAL3_CAM1_DROP_EPISODE_P` over
  `{0.0, 0.20, 0.35, 0.50, 0.65}` to find the sweet spot between robustness
  and final loss.
- **Optional `v17` battery arm**: if v17 deploys regularly, add a dedicated
  case in `scripts/run_eval3_deploy_battery.sh` so the user doesn't have to
  use the `v16` arm.
- **`EVAL3_V17_INCLUDE_PINS30=1` launcher toggle** — wire the 90-dataset
  pins30 slate from §3.6 into the v17 launcher the same way `INCLUDE_ALGVR`
  is wired: glob-discover `datasets/dataset_v5_synth_*_full_pins30`, append
  to `EVAL3_LOCAL_REPOS`, warn-and-continue if the slate is empty. Until
  this lands, use the manual `EVAL3_LOCAL_REPOS=…` recipe at the end of
  §3.6.
- **Stronger face detector for the pins quality filter.** Haar's recall on
  partial / profile / occluded faces leaks bad photos through the
  single-face gate (§3.6 known limits). Drop-in upgrade: MediaPipe Face
  Detection (`pip install mediapipe`) or YOLOv8-face in
  `tools/pins_quality_filter.py:_detect_faces`.
- **OCR pre-filter** for the pins pool to catch meme captions / watermarks
  (Morgan Freeman's new top-1 in v17 has this problem).

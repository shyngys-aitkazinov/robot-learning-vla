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

## 8. Key files

- `scripts/run_eval3_smolvla_v17_real_data_slot_train.sh` — this launcher
- `scripts/eval3_dataset_prep.py` — `CameraDropAugmenter` (class) + `.clone()` defense + cam2 assertion
- `scripts/eval3_concat_patch.py` — env-var parsing + per-repo augmenter wiring
- `scripts/eval3_smolvla_slot_bottleneck.py` — slot classifier + LayerNorm fix + camera2 slice (unchanged from v16)
- `scripts/eval3_vla_deploy.py` — closed-loop deploy (no cam-drop, no changes vs v16)
- `tools/eval3_v16_dataset_test.py` — 73-check robustness suite (D + C + E phases)
- `docs/eval3/v16_playbook.md` — companion playbook for the v16 slot bottleneck (architectural prerequisite)
- `scripts/run_eval3_synth_algvr_dataset_gen.sh` — builds the algvr-conference synth slate consumed by `EVAL3_V17_INCLUDE_ALGVR=1` (see §3.5)
- `tools/eval3_synth_pins_dataset_gen.py` — Pins-pool generator the algvr launcher wraps (now accepts `--source-prefix` / `--source-suffix` / `--output-prefix` / `--output-postfix` so it can target v5 charuko sources)
- `datasets/algvr-conference.json` — 34-person celeb pool the algvr slate samples from

## 9. Open items

- **Push the final v17 checkpoint to the Hub** once full-run results land
  (~10-50k steps). Update `EVAL3_V17_CKPT` deploy entry once available.
- **Held-out evaluation**: re-use `dataset_v3_synth_pinned_idood_*_2` synth
  scenes (not in the v17 corpus → clean held-out set) for offline scoring.
- **Cam-drop sensitivity sweep**: ablate `EVAL3_CAM1_DROP_EPISODE_P` over
  `{0.0, 0.20, 0.35, 0.50, 0.65}` to find the sweet spot between robustness
  and final loss.
- **Optional `v17` battery arm**: if v17 deploys regularly, add a dedicated
  case in `scripts/run_eval3_deploy_battery.sh` so the user doesn't have to
  use the `v16` arm.

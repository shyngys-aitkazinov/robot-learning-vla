# v16 Slot-Bottleneck VLA — Experiment Playbook

The v16 run trains a SmolVLA slot-bottleneck policy that **commits to its
initial language+image slot guess** instead of drifting onto the observed coke
motion mid-episode. This playbook covers what it is, how to run/monitor it, and
how to deploy the result.

## 1. What v16 is (and why)

Prior slot runs (v15, v6-synth) hit high `slot_acc` only because the slot
classifier read the *current* frame — post-grasp, the carried can reveals the
target, so the head learned a motion shortcut and ignored the prompt. v16
removes that shortcut and forces the slot decision to come from language + a
static scene.

Two changes make it work:

1. **Architecture (frame-0 slot).** The slot classifier reads a dedicated
   `camera2` input instead of the current frame. `h_slot` (the prefix token the
   action expert reads) is a function of that frozen frame-0 scene, so the slot
   decision is constant for the whole episode — the policy commits.
2. **Input LayerNorm fix.** SmolVLA scales image *and* language prefix tokens by
   √960 ≈ 31 (`modeling_smolvla.py:659,685`). Fed raw into the slot
   classifier's cross-attention, the score std reached ~960 → the softmax
   saturated → `q_proj` got ~zero gradient → the head collapsed to uniform
   output (`slot_acc` stuck at chance). Adding a `LayerNorm` on the classifier's
   image and language streams unsaturates it. Verified: with the fix `slot_acc`
   climbs 0.34 → 0.55 (step ~100) → 0.78 (step ~500); without it, flat at 0.38.

## 2. Architecture

| Input | Training | Deploy |
|---|---|---|
| `camera1` | current frame | current frame |
| `camera2` | current frame on pre-grasp frames; cached episode frame-0 on post-grasp frames | episode frame-0, captured at step 0, frozen for the whole episode |
| `camera3` | empty pad (`policy.empty_cameras=1`) | empty pad |

- The `SlotClassifier` reads **only camera2's 64-token slice** + the language
  tokens → 3-way slot logits + the `h_slot` prefix token.
- Slot CE loss is applied **only on pre-grasp frames** (`EVAL3_SLOT_CE_PREGRASP_ONLY=1`).
- `target_position` (0/1/2 = left/middle/right) is derived per-dataset from the
  repo name (`*_left_*`→0, `*_middle_*`→1, `*_right_*`→2).
- Vision encoder + VLM language tower are **frozen** (`train_expert_only`,
  `freeze_vision_encoder=true`) — only the action expert + slot head train
  (~101M of 452M params).

Design/spec: `docs/superpowers/specs/2026-05-20-*-design.md`,
plan: `docs/superpowers/plans/2026-05-20-v16-slot-bottleneck-fix.md`.

## 3. The training run

| | |
|---|---|
| Launcher | `scripts/run_eval3_smolvla_v16_real_data_slot_train.sh` |
| Corpus | 9 real `dataset_v4_*` + 9 synthetic `dataset_v3_synth_pinned_idood_*_3` (18 datasets, ~212k frames / 449 episodes) |
| Steps / batch | 50,000 / 256, checkpoint every 1,000 |
| Init | fresh from `lerobot/smolvla_base` |
| Output | `/ephemeral/outputs/train/eval3_v16_real_synth_50k` ⚠ `/ephemeral` is non-durable |
| wandb | project `eval3-v16-real-synth-50k` |

The synthetic `_3` datasets are **not on the Hub** — they load from
`./datasets/<name>` via `EVAL3_LOCAL_REPOS` (see `eval3_concat_patch._local_root`).
The real `dataset_v4_*` load from the Hub as usual.

### Launch / relaunch

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EVAL3_TRAIN_STEPS=50000 EVAL3_BATCH=256 EVAL3_SAVE_FREQ=1000 EVAL3_WANDB=1 \
EVAL3_WANDB_PROJECT=eval3-v16-real-synth-50k \
EVAL3_TRAIN_OUT=/ephemeral/outputs/train/eval3_v16_real_synth_50k \
EVAL3_JOB_NAME=eval3_v16_real_synth_50k \
  ./scripts/run_eval3_smolvla_v16_real_data_slot_train.sh --log_freq=100
```

- `expandable_segments:True` is required at batch 256 — it leaves only ~2 GB
  GPU headroom (vs ~0.7 GB without). If it OOMs, drop to `EVAL3_BATCH=128`
  (~14 h to 50k, ~25 GB headroom).
- Corpus toggles: `EVAL3_V16_NO_SYNTH=1` → real-only; `EVAL3_V16_SYNTH_ONLY=1`
  → synth-only (both used for the isolation smokes).

### Monitor

- **wandb dashboard** — `slot_acc`, `slot_loss`, `loss` curves.
- Log: `grep "ot_train.py:451" <log> | tail` — each line has `slot_acc`,
  `slot_loss`, `slot_ce_n`, `loss`, `lr`.
- One-time startup diagnostics confirm wiring: `v16 prefix check` (camera2 must
  be a real image, std > 0; camera3 must be the empty pad, std 0.0) and
  `v16 mask check` (`post-grasp samples inside CE mask` must be 0).

### Expected behavior (health markers)

- `slot_acc` sits at chance (~0.33) through lr warmup (step ~250), then climbs.
  Reference smokes: real-only ~0.55 by step 100; real+synth ~0.78 by step 500.
- Healthy = `slot_acc` climbing toward ≥0.85; `slot_loss` falling below ln 3
  (1.10). Flat at ~0.38 = the LayerNorm fix is missing or mis-applied.
- `slot_ce_n` ≈ 25-30% of batch size (the pre-grasp fraction). 0 = grasp
  detection failed.

## 4. Deploy

v16 is a **two-camera** checkpoint — it needs the 2-camera rename_map and
`policy.empty_cameras=1` (NOT 2). The deploy code is v16-aware as of 2026-05-21.

### Preferred: deploy battery

```bash
EVAL3_V16_CKPT=<checkpoint-path-or-hf-repo> \
  ./scripts/run_eval3_deploy_battery.sh v16 --task='Place the coke on Taylor Swift'
```

The `v16` case arm sets the 2-camera rename_map + `empty_cameras=1` (overriding
the single-camera `COMMON_ARGS`). `EVAL3_V16_CKPT` defaults to the local
`/ephemeral` checkpoint; set it to an HF repo once the checkpoint is pushed.
Default mode is `raw` (no deploy guards); append `_smooth` for the friend-recipe
biases.

### Direct

```bash
python scripts/eval3_vla_deploy.py \
  --policy.path=<ckpt> --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}' \
  --task='Place the coke on Taylor Swift' --dry_run=true   # drop --dry_run to drive hardware
```

`eval3_vla_deploy.py` auto-detects v16 from the checkpoint's `train_config.json`
and **auto-adopts** the rename_map if `--rename_map` omits the
`front_frame0→camera2` half — so a forgotten flag warns rather than silently
feeding camera2 black frames. It captures the camera frame at step 0 and injects
it as `observation.images.front_frame0` every step (→ camera2).

### Pre-flight (run before every hardware deploy)

```bash
python tools/eval3_check_deploy_command.py \
  --policy-pretrained-path <ckpt> \
  --rename-map '{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}' \
  --task 'Place the coke on Taylor Swift'
```

The validator is v16-aware: it reports `camera2` as "v16 frame-0 fed
(deploy-injected)" and passes the camera check (it does not need a robot
camera).

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| OOM mid-run at batch 256 | fragmentation; relaunch at `EVAL3_BATCH=128` |
| `slot_acc` flat at ~0.38 | LayerNorm fix missing — check `img_ln`/`lang_ln` in `SlotClassifier` |
| `slot_ce_n=0` | grasp detection failed — check `EVAL3_GRASP_GRIP_DELTA`, gripper state column |
| Deploy: policy ignores the prompt / grasps wrong slot | camera2 fed black frames — rename_map missing `front_frame0→camera2` |
| Deploy warns `empty_cameras=2 expected` | stale check — v16 needs `empty_cameras=1`; verify the deploy code is the 2026-05-21 version |
| `EVAL3_LOCAL_REPOS` repo not found | the synth `_3` dataset dir is missing under `./datasets/` |

## 6. Key files

- `scripts/run_eval3_smolvla_v16_real_data_slot_train.sh` — launcher (corpus, knobs)
- `scripts/eval3_smolvla_slot_bottleneck.py` — slot classifier + LayerNorm fix + camera2 slice
- `scripts/eval3_dataset_prep.py` — frame-0 cache, grasp detection, `is_pregrasp`
- `scripts/eval3_concat_patch.py` — 18-dataset concat + `EVAL3_LOCAL_REPOS` local loading
- `scripts/eval3_vla_deploy.py` — closed-loop deploy (v16 frame-0 detection + auto-adopt)
- `scripts/run_eval3_deploy_battery.sh` — `v16` deploy entry
- `tools/eval3_check_deploy_command.py` — v16-aware pre-flight validator

## 7. Open items

- **Push the final checkpoint to the Hub** — it currently lives only on
  `/ephemeral` (non-durable). After the push, set `EVAL3_V16_CKPT` to the HF repo.
- **Post-hoc validation** — a checkpoint scorer on held-out `dataset_v3_synth_pinned_idood_*_2`
  synth scenes (the `_2` datasets are not in the v16 training corpus, so they
  are a clean held-out set).

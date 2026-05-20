# v16 Slot-Bottleneck Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SmolVLA slot-bottleneck VLA actually use the language prompt to pick the target celebrity, instead of ignoring it and collapsing to one direction.

**Architecture:** Two coupled fixes, both monkey-patched from `scripts/` (no `.venv` edits — same pattern as the existing `eval3_smolvla_slot_bottleneck.py`). **Slot half:** the slot classifier reads the episode's *frame-0* image (a static, pre-motion scene) instead of the current frame — carried through the existing, already-configured `camera2` slot — so its h_slot token is naturally constant per episode and the CE loss is forced onto the prompt (frame 0 is counterfactual: same 3-face scene appears with all 3 prompts). **Expert half:** temporal loss weighting up-weights the late timesteps of the 50-step action chunk, where the directional commitment lives, so committing to the mean is expensive.

**Tech Stack:** Python 3.12, PyTorch, lerobot (PyPI), SmolVLA. No pytest in this repo — verification is by smoke runs, `--dry-run`, the existing `tools/eval3_slot_bottleneck_unit_tests.py`, and `tools/eval3_t0_mean_seeking_probe.py`.

---

## Background (why each change)

- The v4 datasets are **confirmed counterfactual**: each physical 3-face layout was recorded with all 3 placements, so the same frame-0 scene appears with 3 prompts → 3 actions. The data is correct; the training recipe is what fails.
- Deploy logs show the slot head **locks onto the coke's visible motion** post-grasp (`L 1.00`). Reading frame-0 only (no motion) removes that shortcut.
- The probe showed the action expert rides state-continuation on the carry, but the **directional commitment** lives in the *tail* of the 50-step chunk on pre-divergence frames, where its loss is tiny → mean-seeking. Temporal loss weighting makes that tail expensive.

## Key design decision — frame-0 rides the `camera2` slot

SmolVLA base already has `image_features = [camera1, camera2, camera3]`. v15 fed only `camera1` and let `empty_cameras=2` pad cam2/cam3 with `-1` images (`prepare_images`, `modeling_smolvla.py:404-444`). v16 feeds the **episode's frame-0 image as `camera2`** (real), with `empty_cameras=1`. Consequences, all favorable:

- The existing preprocessor (resize/normalize/imagenet-stats) handles `camera2` identically to `camera1` — **no new preprocessing code**.
- h_slot becomes a function of `camera2` = the episode's frame-0 image, which is **constant within an episode** → h_slot is "frozen per episode" automatically at train *and* deploy — **no caching/freezing logic needed**.
- `prepare_images` embeds cameras in order → prefix image tokens are `[cam1 …, cam2 …, cam3-empty …]`. The slot classifier reads the **cam2 slice**.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `scripts/eval3_smolvla_slot_bottleneck.py` | the slot patch | Modify: classifier reads cam2 token slice; add temporal loss weighting; add warm-up action-loss schedule |
| `scripts/eval3_dataset_prep.py` | dataset wrapper | Modify: cache per-episode frame-0 images; attach `observation.images.front_frame0` to every row |
| `scripts/eval3_vla_deploy.py` | closed-loop rollout | Modify: capture frame-0 camera image, feed it as `camera2` every step |
| `scripts/run_eval3_smolvla_v16_real_data_slot_train.sh` | launcher | Create |
| `tools/eval3_slot_bottleneck_unit_tests.py` | offline checks | Modify: add v16 checks |

`scripts/eval3_concat_patch.py` needs **no change** — it already constructs `Eval3PrepDataset`; the frame-0 attach is internal to that class. `scripts/train_eval3_smolvla.py` needs **no change** — it already selects the slot patch when `EVAL3_SLOT_LOSS_WEIGHT>0`.

## Env-var contract (new knobs, all default to no-op)

| Env var | Default | Effect |
|---|---|---|
| `EVAL3_SLOT_FRAME0` | `0` | `1` = classifier reads the `camera2` (frame-0) token slice instead of all image tokens |
| `EVAL3_TEMPORAL_LOSS_W_LATE` | `1.0` | weight on the *last* chunk timestep (linear ramp from 1.0 at t=0); `>1` enables |
| `EVAL3_SLOT_WARMUP_STEPS` | `0` | steps during which the action loss is multiplied by 0 (classifier-only warm-up) |

---

## Task 1: Temporal loss weighting

**Files:**
- Modify: `scripts/eval3_smolvla_slot_bottleneck.py` (inside `apply()`, after the existing patches)

`VLAFlowMatching.forward` (`modeling_smolvla.py:763-799`) returns per-element `losses` of shape `(B, T, action_dim)` *before* `SmolVLAPolicy.forward` reduces it with `.mean()` (`modeling_smolvla.py:355-401`). Wrapping `VLAFlowMatching.forward` and scaling `losses` by a `(1,T,1)` ramp applies the weighting cleanly upstream of the reduction.

- [ ] **Step 1: Add the env helper + patch** in `apply()`, after the MetricsTracker block, before `_APPLIED = True`:

```python
    # ---- 5. temporal loss weighting -------------------------------------
    w_late = _env_float("EVAL3_TEMPORAL_LOSS_W_LATE", 1.0)
    if w_late != 1.0:
        _orig_vlaf_forward = VLAFlowMatching.forward

        def _patched_vlaf_forward(self, *args, **kwargs):
            losses = _orig_vlaf_forward(self, *args, **kwargs)  # (B, T, A)
            T = losses.shape[1]
            ramp = torch.linspace(1.0, w_late, T, device=losses.device, dtype=losses.dtype)
            ramp = ramp / ramp.mean()  # keep overall loss scale unchanged
            return losses * ramp.view(1, T, 1)

        VLAFlowMatching.forward = _patched_vlaf_forward
        logging.info("[eval3_slot_bottleneck] temporal loss weighting: 1.0 -> %.2f over %d steps", w_late, 0)
```

- [ ] **Step 2: Verify it imports** — `python scripts/eval3_smolvla_slot_bottleneck.py` (the `__main__` block applies the patch). Expected: `[eval3_slot_bottleneck] applied OK`, no traceback.

- [ ] **Step 3: Commit** — `git add scripts/eval3_smolvla_slot_bottleneck.py && git commit -m "v16: temporal loss weighting on the action chunk tail"`

---

## Task 2: Per-episode frame-0 image cache + row attach

**Files:**
- Modify: `scripts/eval3_dataset_prep.py` — `Eval3PrepDataset.__init__` (ends ~line 1267) and `__getitem__` (lines 1344-1386)

- [ ] **Step 1: Build the frame-0 cache at the end of `__init__`** (after the cache-save block, ~line 1267). It is gated on `EVAL3_SLOT_FRAME0=1` so the default path is untouched:

```python
        # v16: per-episode frame-0 image cache. Read once here (~88 decodes
        # for the v4 corpus); attached to every row in __getitem__ so the slot
        # classifier can read the static, pre-motion scene.
        self._frame0_by_ep: dict[int, Any] = {}
        if os.environ.get("EVAL3_SLOT_FRAME0", "0") == "1":
            for ep_idx, f0 in enumerate(self._episode_from_idxs):
                try:
                    self._frame0_by_ep[ep_idx] = self._ds[int(f0)][self._image_key]
                except Exception as exc:
                    logging.warning("eval3_prep: frame-0 cache miss ep %d (%s)", ep_idx, exc)
            logging.info("eval3_prep: cached %d frame-0 images for slot head (%s)",
                         len(self._frame0_by_ep), self._ds.repo_id)
```

(`import os` already present at module top — confirm; add if missing.)

- [ ] **Step 2: Attach the frame-0 image in `__getitem__`**, immediately before `return row` (after the `target_position` line, ~1385):

```python
        # v16: attach this episode's frame-0 image so the slot classifier
        # reads a static pre-motion scene (renamed to camera2 by rename_map).
        if self._frame0_by_ep:
            ep = self._episode_index_by_frame[int(original_idx)]
            f0img = self._frame0_by_ep.get(int(ep))
            if f0img is not None:
                row["observation.images.front_frame0"] = f0img
```

- [ ] **Step 3: Verify** — load one prepped dataset and check the key is present:

```bash
python -c "
import sys, os; sys.path.insert(0,'scripts'); os.environ['EVAL3_SLOT_FRAME0']='1'
import eval3_lerobot_shim; eval3_lerobot_shim.apply()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from eval3_dataset_prep import Eval3PrepDataset
ds = Eval3PrepDataset(LeRobotDataset('RobotLearningVLA/dataset_v4_taylor_left', video_backend='pyav'),
                      max_frames_per_episode=None, target_position_idx=0)
r = ds[50]
assert 'observation.images.front_frame0' in r, r.keys()
print('OK frame0 shape', tuple(r['observation.images.front_frame0'].shape))
"
```
Expected: `OK frame0 shape (3, H, W)` — no assertion error.

- [ ] **Step 4: Commit** — `git add scripts/eval3_dataset_prep.py && git commit -m "v16: cache + attach per-episode frame-0 image for the slot head"`

---

## Task 3: Slot classifier reads the camera-2 (frame-0) token slice

**Files:**
- Modify: `scripts/eval3_smolvla_slot_bottleneck.py` — `_patched_embed_prefix` (currently lines ~199-230)

Currently the classifier reads `embs[:, :n_img, :]` — *all* image tokens (cam1+cam2+cam3). With frame-0 in cam2, the classifier must read only the cam2 slice. Cameras are embedded in fixed order and equal token counts, so `tokens_per_cam = n_img // num_cams` and the cam2 slice is `[tokens_per_cam : 2*tokens_per_cam]`.

- [ ] **Step 1: Add the frame-0 toggle near the other env reads** in `apply()`:

```python
    frame0_mode = _env_bool("EVAL3_SLOT_FRAME0", False)
```

- [ ] **Step 2: In `_patched_embed_prefix`, slice the cam2 tokens** — replace the `img_embs = embs[:, :n_img, :]` line with:

```python
        img_embs = embs[:, :n_img, :]
        if frame0_mode:
            # cameras are embedded in order [cam1, cam2(frame-0), cam3]; read cam2.
            n_cams = int(getattr(self.config, "empty_cameras", 0)) + \
                     len([k for k in self.config.image_features if k]) - \
                     int(getattr(self.config, "empty_cameras", 0))
            n_cams = max(1, len(self.config.image_features))
            tok_per_cam = n_img // n_cams
            if tok_per_cam > 0 and n_cams >= 2:
                img_embs = embs[:, tok_per_cam:2 * tok_per_cam, :]
            else:
                logging.warning("[eval3_slot_bottleneck] frame0 mode: cannot slice cam2 "
                                "(n_img=%d n_cams=%d); using all image tokens", n_img, n_cams)
```

(`self.config.image_features` is the ordered camera list; `len(...)` is the camera count incl. empties. Confirm against the loaded config in the smoke test — Step 3.)

- [ ] **Step 3: Verify** via the smoke run in Task 7 — the log line `[eval3_slot_bottleneck] installed SlotClassifier` must still appear and `slot_loss/slot_acc` must still be finite (Task 7 Step 2).

- [ ] **Step 4: Commit** — `git add scripts/eval3_smolvla_slot_bottleneck.py && git commit -m "v16: slot classifier reads the frame-0 (camera2) token slice"`

---

## Task 4: Classifier warm-up (action-loss schedule)

**Files:**
- Modify: `scripts/eval3_smolvla_slot_bottleneck.py` — `_patched_policy_forward` (lines ~237-263)

Phase 1 of the warm-up = freeze the action expert while the classifier bootstraps. Implemented by multiplying the **action** loss by 0 for the first `EVAL3_SLOT_WARMUP_STEPS` steps (zero gradient to the expert = effectively frozen); the slot CE loss still flows. The current training step is already available via the curriculum counter (`_get_curriculum_counter()`).

- [ ] **Step 1: Read the knob** in `apply()`:

```python
    warmup_steps = _env_int("EVAL3_SLOT_WARMUP_STEPS", 0)
```

- [ ] **Step 2: In `_patched_policy_forward`, scale the action loss** — replace `loss = loss + weight * slot_loss` with:

```python
                action_w = 1.0
                if warmup_steps > 0:
                    counter = _get_curriculum_counter()
                    step = int(counter.get_step()) if counter is not None else warmup_steps
                    action_w = 0.0 if step < warmup_steps else 1.0
                loss = action_w * loss + weight * slot_loss
                loss_dict["action_loss_weight"] = float(action_w)
```

(`_get_curriculum_counter` is defined later in `apply()`; move its definition above `_patched_policy_forward`, or reference it lazily as the existing code already does inside `_patched_mt_setattr`. Lazy reference inside the closure is fine — it is only called at forward time.)

- [ ] **Step 3: Verify** — Task 7 smoke run with `EVAL3_SLOT_WARMUP_STEPS=6 EVAL3_TRAIN_STEPS=12`: early steps must show `action_loss_weight: 0.000`, later steps `1.000`.

- [ ] **Step 4: Commit** — `git add scripts/eval3_smolvla_slot_bottleneck.py && git commit -m "v16: classifier warm-up via action-loss schedule"`

---

## Task 5: v16 launcher

**Files:**
- Create: `scripts/run_eval3_smolvla_v16_real_data_slot_train.sh` (start from a copy of `run_eval3_smolvla_v15_real_data_slot_train.sh`)

- [ ] **Step 1: Copy and edit.** Key diffs from the v15 launcher:
  - `RENAMES='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}'`
  - `--policy.empty_cameras=1` (was 2)
  - export `EVAL3_SLOT_FRAME0=1`
  - export `EVAL3_TEMPORAL_LOSS_W_LATE="${EVAL3_TEMPORAL_LOSS_W_LATE:-4.0}"`
  - export `EVAL3_SLOT_WARMUP_STEPS="${EVAL3_SLOT_WARMUP_STEPS:-500}"`
  - `OUT=/ephemeral/outputs/train/eval3_v16_real_data_slot`, `JOB=eval3_v16_real_data_slot`
  - `EVAL3_WANDB_PROJECT=eval3-v16-real-data-slot`
  - keep: 9 `dataset_v4_*` real datasets, batch 128, 10000 steps, save_freq 1000, fresh from `lerobot/smolvla_base`, existing stochastic state aug knobs **unchanged**.

- [ ] **Step 2: Dry-run** — `EVAL3_DRY_RUN=1 ./scripts/run_eval3_smolvla_v16_real_data_slot_train.sh`. Expected: prints the config, exits 0.

- [ ] **Step 3: Commit** — `git add scripts/run_eval3_smolvla_v16_real_data_slot_train.sh && git commit -m "v16: launcher (frame-0 camera2, temporal weighting, warm-up)"`

---

## Task 6: Deploy — feed frame-0 as camera2

**Files:**
- Modify: `scripts/eval3_vla_deploy.py` — observation-build path (near `predict_action` calls, lines ~441 and ~476)

- [ ] **Step 1:** On the first control-loop tick, capture the camera frame into `frame0_img` (alongside the existing "Home position captured" / first-frame-save logic, ~line 425).

- [ ] **Step 2:** When building each observation dict, add `observation["observation.images.front_frame0"] = frame0_img` so the rename map routes it to `camera2`. Apply on every tick (frame-0 is constant).

- [ ] **Step 3:** The deploy `rename_map` must include the second entry — accept it from `--rename_map` (the launcher/battery passes it) and confirm `eval3_check_deploy_command.py` still passes.

- [ ] **Step 4: Verify** — `python scripts/eval3_vla_deploy.py --policy.path=<v16 ckpt> --rename_map='{"observation.images.front":"observation.images.camera1","observation.images.front_frame0":"observation.images.camera2"}' --dry_run`. Expected: checkpoint loads, slot patch applies, no traceback.

- [ ] **Step 5: Commit** — `git add scripts/eval3_vla_deploy.py && git commit -m "v16: deploy feeds captured frame-0 as camera2"`

---

## Task 7: Verification

**Files:**
- Modify: `tools/eval3_slot_bottleneck_unit_tests.py` — add v16 checks (temporal-weight ramp mean≈1; frame0 cam2 slice indices; warm-up schedule 0→1)

- [ ] **Step 1: Unit tests** — `python tools/eval3_slot_bottleneck_unit_tests.py`. Expected: all checks pass.

- [ ] **Step 2: Smoke run** — datasets cached from v15, so this is fast:

```bash
EVAL3_TRAIN_STEPS=12 EVAL3_BATCH=2 EVAL3_SAVE_FREQ=12 EVAL3_WANDB=0 \
EVAL3_SLOT_WARMUP_STEPS=6 \
EVAL3_TRAIN_OUT=/ephemeral/outputs/train/eval3_v16_smoke EVAL3_JOB_NAME=eval3_v16_smoke \
  ./scripts/run_eval3_smolvla_v16_real_data_slot_train.sh --log_freq=2 2>&1 | tee /tmp/v16_smoke.log
```
Expected in the log: `joining datasets` lists 9 v4 repos; `installed SlotClassifier`; `slot_loss`/`slot_acc` finite; `action_loss_weight` = 0.0 for steps <6 then 1.0; `End of training`; a checkpoint saved with `model.slot_clf.*` keys.

- [ ] **Step 3: t=0 probe** on the smoke checkpoint (sanity, not a score) — `python tools/eval3_t0_mean_seeking_probe.py --checkpoint /ephemeral/outputs/train/eval3_v16_smoke/checkpoints/000012/pretrained_model --celeb taylor_swift --device cpu`. Expected: runs without error and prints the table.

- [ ] **Step 4: Commit** — `git add tools/eval3_slot_bottleneck_unit_tests.py && git commit -m "v16: unit-test coverage for the v16 changes"`

---

## Risks / open items

- **`camera2` acceptance:** SmolVLA base's `image_features` already lists `camera2` (v15 padded it empty), so `prepare_images` should pick up a real `camera2`. If the smoke run (Task 7 Step 2) errors on image features, the fallback is the **CE-mask** variant: keep the classifier on the current-frame tokens, add a per-sample `is_pregrasp` flag (plumbed like `target_position`) and mask the slot CE loss to pre-grasp frames, plus freeze h_slot at rollout. Documented here so the executor doesn't improvise.
- **Token-per-camera count:** Task 3 assumes equal token counts per camera and `image_features` order = embed order. Confirm in the smoke log; if `n_img` is not divisible by the camera count, log and fall back to all-image-tokens (the code already does).
- **Deploy state consistency:** v16 keeps the stochastic state aug only — no deterministic state→home — so there is no train/deploy state mismatch to handle (decision reached in brainstorming).

## Self-review

- Spec coverage: slot-frame0 (Tasks 2,3,5,6) ✓; CE loss + detach unchanged (kept) ✓; warm-up (Task 4) ✓; temporal loss weighting (Task 1) ✓; stochastic state aug kept (Task 5, explicit) ✓; deploy freeze — achieved structurally via camera2 (Task 6) ✓; launcher (Task 5) ✓.
- No placeholders: every code step has concrete code; the one residual ambiguity (token-per-camera) has an explicit in-code fallback.
- Naming consistency: `EVAL3_SLOT_FRAME0`, `EVAL3_TEMPORAL_LOSS_W_LATE`, `EVAL3_SLOT_WARMUP_STEPS`, `observation.images.front_frame0` used identically across all tasks.

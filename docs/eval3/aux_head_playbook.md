# Eval 3 — Auxiliary head + state/visual/language augmentation (playbook)

The combined intervention for the `v6_synth_*` failure mode where the policy
learned a `(observation.state, image) → action` shortcut and totally ignored
the language prompt. Five orthogonal pressures that all push the model
toward language-image binding:

1. **Aux head** — 3-way classifier on `suffix_out`, gives a direct gradient
   signal that forces the action expert's cross-attention to encode
   language-image position binding.
2. **State Gaussian noise** with cosine-decay curriculum — blocks exact
   state-as-lookup-table; calibrated in normalized stddev units so the
   perturbation is uniform across joints.
3. **State replacement (HOME / zero)** — with probability `p`, replace state
   with the canonical HOME pose (or zero). Same HOME pose ↔ different
   actions across positions, so only language can disambiguate.
4. **Language augmentation** — 21 wording templates (70% canonical demo
   prompts, 30% varied including "image of"/"photo of"/"picture of" phrasings)
   robustify the text encoder against demo-day prompt drift.
5. **Visual augmentation** — torchvision `RandomErasing` + ColorJitter +
   small geometric transforms add visual robustness.

This doc is the operating manual: recommended hyperparameters, what each
knob does, how to verify, and FAQ.

---

## TL;DR

The launcher `scripts/run_eval3_smolvla_v6_synth_train.sh` exports all the
recommended defaults. To run with everything enabled:

```bash
./scripts/run_eval3_smolvla_v6_synth_train.sh
```

To run with the aux head + augmentations **disabled** (ablation baseline):

```bash
EVAL3_AUX_POS_LOSS_WEIGHT=0 \
EVAL3_STATE_NOISE_SIGMA_MAX=0 \
EVAL3_STATE_REPLACE_PROB=0 \
EVAL3_TASK_AUG_CANONICAL_P=1.0 \
./scripts/run_eval3_smolvla_v6_synth_train.sh
```

All knobs are env vars; defaults in the launcher are evidence-based starting
points (see "Recommended hyperparameters" below).

The metric to watch in the periodic `INFO step:N` log line:
- `aux_pos_loss`: 3-way CE on the position classifier. Random = 1.10,
  converged ≈ 0.05.
- `aux_pos_acc`: 3-way classification accuracy. Random = 0.33,
  converged ≈ 1.00.
- `loss`: TOTAL loss (action MSE + `aux_pos_loss_weight` × aux CE).

Both auto-flow to wandb when `--wandb.enable=true`.

---

## What it patches

`scripts/eval3_smolvla_aux_head.py:apply()` installs six monkey-patches at
training/deploy startup:

| # | Target | Effect |
|---|---|---|
| 0 | `lerobot.processor.converters._extract_complementary_data` | Allows `target_position` to survive `batch_to_transition() → transition_to_batch()` (lerobot otherwise drops unknown batch keys) |
| 1 | `VLAFlowMatching.__init__` | Adds `self.position_clf_head` — a `Linear(720→256) → GELU → Dropout(0.1) → Linear(256→3)` MLP; reads env-var config |
| 2 | `VLAFlowMatching.forward` | Adds an optional `target_position` kwarg; mean-pools `suffix_out` across 50 chunk steps; computes CE + accuracy; stashes results on `self._last_aux_*` |
| 3 | `SmolVLAPolicy.forward` | Pops `target_position` from the batch (so it doesn't flow into `prepare_state`/`prepare_action`); passes it to `model.forward`; adds the weighted aux loss to the main loss; surfaces metrics in `loss_dict` |
| 4 | `MetricsTracker.__init__` | Auto-adds `aux_pos_loss` + `aux_pos_acc` `AverageMeter`s so they appear in the per-step INFO log line and `to_dict()` (wandb) |
| 5 | `MetricsTracker.__setattr__` (side-channel pump) + `SmolVLAPolicy.__init__` (auto-register) | When the train loop does `train_tracker.loss = loss` (once per step), the hook also updates the aux meters from the policy's last forward |

`scripts/eval3_dataset_prep.py:Eval3PrepDataset` was extended to emit a
per-row `target_position` (long tensor) derived from the dataset's repo
name; `scripts/eval3_concat_patch.py` plumbs the derivation through when
each dataset is wrapped.

---

## Files in this feature

| Path | Role |
|---|---|
| `scripts/eval3_smolvla_aux_head.py` | The monkey-patch. ~310 lines. Apply by calling `apply()` once at import time. |
| `scripts/eval3_dataset_prep.py` | `Eval3PrepDataset.__init__` takes `target_position_idx`; `__getitem__` emits `row["target_position"]` |
| `scripts/eval3_concat_patch.py` | Derives slot index from `_slot_from_repo(d.repo_id)` and passes it to each `Eval3PrepDataset` |
| `scripts/train_eval3_smolvla.py` | Calls `eval3_smolvla_aux_head.apply()` after `eval3_concat_patch.apply_concat_patch()` |
| `scripts/eval3_vla_deploy.py` | Also calls `apply()` so checkpoints trained with the aux head load cleanly at deploy time |
| `scripts/run_eval3_smolvla_v6_synth_train.sh` | Exports `EVAL3_AUX_POS_LOSS_WEIGHT` / `EVAL3_AUX_POS_DROPOUT` / `EVAL3_AUX_POS_HIDDEN` |
| `tools/eval3_aux_head_unit_tests.py` | 10 unit tests covering every edge case (weight=0, no-label, partial-ignore, save/load round-trip, inference path, MetricsTracker integration, dtype matching). Run before/after any change. |
| `tools/eval3_aux_head_cross_prompt_test.py` | The diagnostic test: feeds 2 specific training frames through 3 prompts and reports `Δ_max(prompt)` per frame. Used to detect language-conditioning collapse. |

---

## Enable / configure

Set these env vars before launching training (defaults in parens):

| Env var | Default | Recommended | Purpose |
|---|---|---|---|
| `EVAL3_AUX_POS_LOSS_WEIGHT` | `0.0` (off) | **`0.3`** | Weight of the aux CE loss in the total. `0` = patch is a runtime no-op. Sweep `0.1` / `0.3` / `1.0`. |
| `EVAL3_AUX_POS_DROPOUT` | `0.1` | `0.1` | Dropout inside the MLP head. Higher (`0.3`) discourages the head from solving the task purely from arm-pose features. |
| `EVAL3_AUX_POS_HIDDEN` | `256` | `256` | Hidden width of the head. Larger (`512`) is unlikely to help; the task is 3-way. |

The launcher `scripts/run_eval3_smolvla_v6_synth_train.sh` already exports
these. To override per-run:

```bash
EVAL3_AUX_POS_LOSS_WEIGHT=0.5 ./scripts/run_eval3_smolvla_v6_synth_train.sh
```

To disable the patch entirely (it's still imported but has no effect):

```bash
EVAL3_AUX_POS_LOSS_WEIGHT=0 ./scripts/run_eval3_smolvla_v6_synth_train.sh
```

---

## Verify the patch works

### Step 1 — Unit tests (~3 min on MPS)

Run before/after any change to the patch:

```bash
source .venv/bin/activate
python tools/eval3_aux_head_unit_tests.py
```

You want `ALL TESTS PASSED ✓` at the end. The 10 tests cover:

| Test | What it asserts |
|---|---|
| T1 | weight=0 ⇒ no aux contribution (no `aux_pos_loss` in loss_dict) |
| T2 | `target_position` absent in batch ⇒ no aux contribution |
| T3 | all-ignore batch (all `-100`) ⇒ no aux contribution, no spurious metric updates |
| T4 | partial-ignore batch ⇒ aux runs only on the valid subset |
| T5 | `apply()` is idempotent (re-applying is a no-op) |
| T6 | save+load round-trip preserves head weights to the byte |
| T7 | `policy.select_action` (inference path) does NOT invoke the aux head |
| T8 | mixed-label batch ⇒ head receives gradient (`grad_norm > 0`) |
| T9 | `MetricsTracker` has `aux_pos_loss` + `aux_pos_acc` meters; pump updates them; they appear in `__str__` and `to_dict()` (wandb-ready) |
| T10 | `decision_h.dtype` matches `head_param.dtype` (defensive cast in place) |

### Step 2 — Smoke training (~5 min on MPS)

A 50-step training run that exercises the full pipeline:

```bash
rm -rf outputs/train/eval3_aux_smoke

EVAL3_AUX_POS_LOSS_WEIGHT=0.5 \
EVAL3_EXTRA_REPOS="RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2" \
EVAL3_MAX_FRAMES_PER_EP=0 EVAL3_TASK_AUG=1 EVAL3_TASK_AUG_CANONICAL_P=1.0 \
EVAL3_BG_REPLACE=0 EVAL3_PRINT_SHUFFLE=0 EVAL3_GRIPPER_REPAIR=0 \
PYTHONUNBUFFERED=1 python -u scripts/train_eval3_smolvla.py \
  --policy.path=RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k \
  --policy.push_to_hub=false --policy.compile_model=false \
  --policy.device=mps --policy.empty_cameras=2 \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --dataset.repo_id=RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2 \
  --dataset.video_backend=pyav --use_policy_training_preset=false \
  --optimizer.type=adamw --optimizer.lr=1e-4 --optimizer.weight_decay=1e-10 \
  --optimizer.grad_clip_norm=10.0 --scheduler.type=cosine_decay_with_warmup \
  --scheduler.peak_lr=1e-4 --scheduler.decay_lr=1e-6 \
  --scheduler.num_warmup_steps=5 --scheduler.num_decay_steps=45 \
  --job_name=eval3_aux_log_smoke --output_dir=outputs/train/eval3_aux_smoke \
  --steps=50 --save_freq=50 --batch_size=1 --num_workers=0 --log_freq=10
```

Healthy output should include lines like:

```
INFO ... [eval3_smolvla_aux_head] installed position_clf_head (in_dim=720, hidden=256, dropout=0.10, loss_weight=0.500)
INFO ... step:10 smpl:10 ep:0 epch:0.00 loss:0.50 grdn:6.7 lr:3.1e-05 aux_pos_loss:0.95 aux_pos_acc:0.50 ...
INFO ... step:50 smpl:50 ep:1 epch:0.00 loss:0.02 grdn:2.1 lr:1.0e-06 aux_pos_loss:0.01 aux_pos_acc:1.00 ...
INFO ... End of training
```

What to confirm:
- `installed position_clf_head` log line appears once at startup.
- `aux_pos_loss` and `aux_pos_acc` appear in EVERY periodic INFO line.
- `aux_pos_loss` decreases monotonically (it's a simple 3-class classifier; should hit < 0.1 within 50-100 steps).
- `aux_pos_acc` rises to ~1.0 within 30-50 steps.
- Total `loss` stays bounded (action MSE doesn't explode).

### Step 3 — Cross-prompt sensitivity on the saved checkpoint (~30 sec)

Use the diagnostic test to measure whether the trained model now uses the
prompt:

```bash
python tools/eval3_aux_head_cross_prompt_test.py \
    --checkpoint outputs/train/eval3_aux_smoke/checkpoints/000050/pretrained_model
```

Healthy output (model uses language):
```
LeCun_left_ep0   400   +29.05    +29.10  +20.50  +18.50    20.60  ✓ uses lang
Swift_right_ep1  400   -20.18    +18.10  -19.95  +12.30    38.05  ✓ uses lang
Summary: mean Δ_max(prompt) ≈ 25-50°  ← language is being respected
```

Collapsed output (model ignores language):
```
LeCun_left_ep0   400   +29.05    +28.91  +29.15  +28.97     0.24  ✗ ignores
Summary: mean Δ_max(prompt) ≈ 1-2°  ← language still ignored
```

Baseline (v6_synth_15k, before any fix): mean Δ = 0.6°, max = 1.95°.

---

## Reading the metrics

The patched `MetricsTracker` reports two new fields in every periodic INFO
line (and `wandb.log()` payload):

- **`aux_pos_loss`** — the **raw** cross-entropy loss of the 3-way classifier
  on `suffix_out.mean(dim=1)`. Random-init starts at ≈ `−ln(1/3) = 1.10`.
  Anywhere below `0.5` means the head is learning the task. Below `0.1`
  means the head is essentially solving it. **Not multiplied by
  `aux_pos_loss_weight`** — that's so the metric is comparable across runs
  with different weights.
- **`aux_pos_acc`** — classification accuracy of the head on the current
  batch's valid samples. Range `[0, 1]`. A 3-way task has random baseline
  `0.33`. Reaches `1.0` quickly on the training distribution.

Total `loss` (also in the log) IS the weighted sum:
```
loss = action_mse + EVAL3_AUX_POS_LOSS_WEIGHT * aux_pos_loss
```

So if you see `loss=0.5` with `aux_pos_loss=0.9, aux_pos_weight=0.5`, you can
back out `action_mse ≈ 0.5 - 0.5 * 0.9 ≈ 0.05`.

---

## State augmentation — the missing piece

**The aux head alone is necessary but not sufficient** for the v6_synth
corpus. Empirical evidence (300-step MPS run from `v6_synth_15k`):

| Metric | Value | Reading |
|---|---|---|
| Aux head training accuracy | **100%** | ✅ Cross-attention CAN bind language → image position |
| Aux head loss (mean) | 0.032 (was 1.10 random) | ✅ Head fully converged |
| Cross-prompt Δ on action pan | 0.6° → 1.11° | ❌ Action head still ignores language |

The aux head puts language into `suffix_out`. But the action head,
sharing the same `suffix_out`, doesn't *use* those features because the
state shortcut (`observation.state → action`) is still cheaper. To break
the shortcut, pair the aux head with **`StateAugmenter`** — implemented
in `scripts/eval3_dataset_prep.py:StateAugmenter`.

### What it does

Three orthogonal pressures, all opt-in via env vars:

1. **Gaussian noise on `observation.state`** with cosine-decay curriculum.
   At training progress `p ∈ [0, 1]` (= current step / curriculum_steps):
   ```
   sigma(p) = sigma_min + 0.5 * (sigma_max - sigma_min) * (1 + cos(π * p))
   ```
   So p=0 → sigma_max; p=1 → sigma_min. Encourages reliance on visual+language
   features early when state is unreliable, while letting the policy learn from
   clean state late. **The gripper component gets `0.1×` the noise** (it's
   near-binary; full noise destroys grasp/release labels).

2. **State replacement** with two modes, weighted:
   - **HOME mode** — replace `observation.state` with `CANONICAL_HOME_STATE +
     jitter`. The same HOME state now appears across left/middle/right
     trajectories; only the prompt distinguishes the action target.
   - **Zero mode** — replace with all zeros. More aggressive — forces the
     model to predict from image + language alone for that frame.

3. **Curriculum step counter** — a `multiprocessing.Value` updated by the
   train loop's per-step hook (in `eval3_smolvla_aux_head.py`), readable by
   DataLoader worker processes via fork-shared memory.

### Env vars

| Env var | Default | Recommended | Effect |
|---|---|---|---|
| `EVAL3_STATE_NOISE_SIGMA_MAX` | `0` (off) | **`5.0`** | σ at training start (degrees). Joints get this; gripper gets 0.1× this. |
| `EVAL3_STATE_NOISE_SIGMA_MIN` | `0` | **`0.5`** | σ at training end. |
| `EVAL3_STATE_NOISE_CURRICULUM_STEPS` | `0` (off) | **`$STEPS`** (full training duration) | Period over which σ decays cosine-style from max→min. `0` = constant σ_max. |
| `EVAL3_STATE_REPLACE_PROB` | `0` (off) | **`0.4`** | Fraction of frames whose `observation.state` gets replaced (HOME or zero, weighted). |
| `EVAL3_STATE_REPLACE_MODES` | `home:0.7,zero:0.3` | same | Per-mode weights within the replacement probability. |
| `EVAL3_STATE_HOME_JITTER_SIGMA` | `1.0` | `1.0` | Noise σ around HOME (so HOME-masked frames aren't byte-identical sentinels). |
| `EVAL3_STATE_GRIPPER_NOISE_SCALE` | `0.1` | `0.1` | Multiplier applied to gripper-component noise (both Gaussian and HOME-jitter). |

The launcher `scripts/run_eval3_smolvla_v6_synth_train.sh` exports all of
these. To disable state augmentation completely:

```bash
EVAL3_STATE_NOISE_SIGMA_MAX=0 EVAL3_STATE_REPLACE_PROB=0 \
  ./scripts/run_eval3_smolvla_v6_synth_train.sh
```

### Canonical HOME

The HOME state used in replacement is the global average of
`observation.state` across the first 5 frames of every episode in
`dataset_v3_charuco_{left,middle,right}_2` (30 episodes × 5 frames = 150
samples). The value lives as `CANONICAL_HOME_STATE` in
`scripts/eval3_dataset_prep.py`:

```python
CANONICAL_HOME_STATE = (1.3574, -102.8120, 96.3487, -99.8464, 7.2586, 0.6771)
```

Why GLOBAL and not per-position: the per-position wrist_roll at frame 0
already carries directional info (left: +9.3°, middle: +0.2°,
right: +12.3°). A per-position HOME would leak this cue. The global
average erases it.

### Visual augmentation (lighter)

`RandomErasing` is now appended to the launcher's `TFS_JSON`:

```json
"random_erasing": {
  "weight": 1.0, "type": "RandomErasing",
  "kwargs": {"p": 0.25, "scale": [0.02, 0.15], "ratio": [0.3, 3.3], "value": 0}
}
```

This is a lightweight visual regularizer — randomly masks small rectangular
regions. Cheap, well-tested, pairs naturally with the state-aug stack.

A more targeted **print region cutout** (mask non-target prints specifically,
forcing the model to use language to find the prompted celebrity) was
considered but **deferred** — it requires per-dataset board bounding boxes
which need extra extraction infrastructure on the inpainted synth data.

### Verification

```bash
python tools/eval3_state_aug_unit_tests.py
```

10 tests cover: no-op when disabled, noise statistics match σ, HOME
replacement uses canonical pose, zero mode produces zeros, mode weighting
follows env config, cosine curriculum hits expected values at progress
0 / 0.5 / 1, `__reduce__` roundtrip for DataLoader workers, env-var
parser builds the right config.

## Recommended hyperparameters (evidence-based defaults)

These are the launcher's current defaults. Reasoning for each value is in the
"Why these values" section below.

```bash
# ---- Aux head -------------------------------------------------------
EVAL3_AUX_POS_LOSS_WEIGHT=0.3       # multiplier on CE loss. Sweep [0.2, 0.3, 0.5] if needed
EVAL3_AUX_POS_HIDDEN=256             # MLP hidden width
EVAL3_AUX_POS_DROPOUT=0.1            # head's dropout; raise to 0.3 if overfitting

# ---- State noise (NORMALIZED stddev units) --------------------------
EVAL3_STATE_NOISE_SIGMA_MAX=0.3      # 30% of one stddev at training start
EVAL3_STATE_NOISE_SIGMA_MIN=0.05     # 5% at training end (cosine decay)
EVAL3_STATE_NOISE_CURRICULUM_STEPS=$STEPS   # match training duration

# ---- State replacement (HOME / zero modes) --------------------------
EVAL3_STATE_REPLACE_PROB=0.4         # fraction of frames whose state is replaced
EVAL3_STATE_REPLACE_MODES="home:0.7,zero:0.3"   # within replaced frames
EVAL3_STATE_HOME_JITTER_SIGMA=1.0    # noise around HOME (raw degrees)
EVAL3_STATE_GRIPPER_NOISE_SCALE=0.1  # 10x less noise on the gripper component

# ---- Language augmentation -----------------------------------------
EVAL3_TASK_AUG=1
EVAL3_TASK_AUG_CANONICAL_P=0.7       # 70% canonical demo wording, 30% varied

# ---- Visual augmentation (in TFS_JSON) ----------------------------
# Already baked into run_eval3_smolvla_v6_synth_train.sh:
#   ColorJitter brightness 2.0x weight
#   ColorJitter contrast 2.0x
#   ColorJitter saturation
#   ColorJitter hue
#   SharpnessJitter
#   RandomAffine ±2° deg, ±0.02 translate
#   RandomPerspective distortion=0.12 p=0.3
#   RandomErasing p=0.25, scale=[0.02, 0.15]
```

## Why these values

### Aux head — `EVAL3_AUX_POS_LOSS_WEIGHT=0.3`

Goal: aux CE should be comparable in magnitude to action MSE so both
gradients have similar influence.

Empirical scales from smoke runs:
- Action MSE: starts ~0.5–1.0 (with state aug active), settles ~0.05
- Aux CE: starts ~1.10 (random init), settles ~0.05 (converged head)

At weight `0.3`:
- Random init: aux contribution = `0.3 × 1.10 = 0.33` — comparable to early action loss, drives the head to learn quickly
- Converged: aux contribution = `0.3 × 0.05 = 0.015` — modest pressure that keeps language features alive without distorting the action head

Lower (0.1) is too weak — aux signal dies once the head converges. Higher
(1.0) risks destabilizing action quality. **0.3 is the sweet spot**; sweep
to 0.5 only if cross-prompt Δ doesn't rise after 5k+ steps of full training.

### Aux head — CE loss (not MSE, not focal, not label-smoothed)

Plain `F.cross_entropy(logits, target, ignore_index=-100)` is exactly right
for this task:
- **3 mutually-exclusive classes** (left/middle/right)
- **Class-balanced corpus** (9 synth datasets, 3 positions × 3 celebs)
- **Zero label noise** (positions derived deterministically from repo
  names)
- **`ignore_index=-100`** for old single-celeb datasets without a slot in
  their name (e.g. `taylor_swift_1`) — their samples contribute no aux
  gradient but still contribute action MSE

We considered and rejected:
- **MSE on softmax probs**: known anti-pattern for classification — slow
  convergence, vanishing gradients at saturation, worse than CE
- **Label smoothing (α=0.1)**: useful for noisy labels; we have none
- **Focal loss**: useful for class imbalance; we have balance
- **Margin losses (ArcFace etc.)**: for fine-grained classification with
  many classes; irrelevant for 3 classes

### Aux head — `EVAL3_AUX_POS_HIDDEN=256`

Linear(720→256) → GELU → Dropout → Linear(256→3). For 3-way classification
on a 720-d input, even a single linear (no hidden) would mathematically
suffice. The hidden + GELU + dropout provides modest capacity slack.

- **128**: also works, slightly less prone to overfitting
- **256**: default — verified at 100% accuracy on training data
- **512**: overkill, may overfit

### Aux head — `EVAL3_AUX_POS_DROPOUT=0.1`

Light dropout. Prevents the head from solving the classification purely
from arm-pose-derived features in `suffix_out` (which would let the action
head ignore language and still get high aux_acc on training data).

If full training shows aux_acc=1.0 but cross-prompt Δ stays low, raise to
**0.3** — forces the head to use multiple "channels" in `suffix_out`,
which biases the cross-attention more strongly toward language-aware
features.

### State noise — `SIGMA_MAX=0.3`, `SIGMA_MIN=0.05`, units = NORMALIZED stddev

**Sigma is in normalized stddev units, not raw degrees.** σ=0.3 means
"perturb each joint by 0.3 of its per-joint training std". Raw-degree
equivalent at a typical std=30° is 9° of effective noise. Per-joint std is
pulled from the dataset's `meta.stats["observation.state"]["std"]` and
passed into the augmenter at concat-patch time.

Why normalized: the lerobot normalizer applies `(x - mean) / std` AFTER my
aug runs. A uniform σ in raw degrees produces 3.7× more effective noise on
low-std joints (shoulder_pan std≈14°) than high-std joints (shoulder_lift
std≈50°). Normalized units make the effect uniform across joints.

Why 0.3: large enough to make state-as-lookup-table genuinely unreliable
(0.3 stddev shifts the input meaningfully), but small enough that the
underlying action prediction can still learn from state when it's needed
for execution (vs decision).

Why 0.05 minimum: as training converges, want the model to use clean state
for fine-grained execution. Don't decay all the way to 0 because a tiny
amount of noise keeps the model from drifting back into the exact-state
lookup mode.

### State replacement — `REPLACE_PROB=0.4`, modes `home:0.7,zero:0.3`

40% of frames have state replaced. Of those, 70% get HOME, 30% get zero.

Why 40%: aggressive enough to substantially break the state-as-lookup
shortcut (40% of training samples have non-trajectory state), but not so
aggressive that the model can't learn the underlying state→action mapping
at all (60% of frames have real state).

Why HOME-weighted: HOME is the physical deploy-start condition. Training
on HOME-anchored frames teaches the policy "from HOME state + scene +
prompt → action" — exactly the deploy-time question. Zero is more
aggressive (state vanishes entirely) and serves as a small diversity
addition.

Why HOME-jitter `σ=1.0°`: prevents the model from detecting "state ==
canonical HOME exact" and treating those frames as a special case. With
σ=1°, HOME-replaced state looks like any nearby-HOME pose (which is what
real deploy state at episode start looks like).

### State noise curriculum — cosine decay from σ_max to σ_min

```
σ(progress) = σ_min + 0.5 × (σ_max - σ_min) × (1 + cos(π × progress))
progress = current_step / curriculum_steps   (clamped to [0, 1])
```

So at progress=0: σ = σ_max (5x noise). At progress=1: σ = σ_min.
Encourages reliance on visual+language features early when state is
unreliable, while letting the policy learn from clean state late.

The `multiprocessing.Value` step counter is updated by the train-loop hook
(in `eval3_smolvla_aux_head.apply()`) and is fork-shared with DataLoader
workers.

### Gripper noise scale = 0.1

Gripper position is near-binary (closed/open). σ=0.3 × std≈27° = 8° of
noise would destroy the grasp/release labels. The 0.1× scaling keeps
gripper perturbation at 0.8° — small enough to preserve label integrity.

### Language augmentation — `canonical_p=0.7`, 21 templates

- 70% canonical demo prompts: "Place the coke on X" (50%) / "Place the
  coke on the X" (20%) — matches what the demo TA will say.
- 30% varied wordings (19 templates): "Put the coke on X", "Drop the coke
  on X", "Move the coke to X", and crucially "Place the coke on the image
  of X", "Place the coke on the photo of X", "Place the coke on the
  picture of X", etc. — these match what the robot literally sees (a
  printed photo, not the person) and help the text encoder learn
  image-grounded language.

Why 0.7 (not 0.5 or 0.9): canonical wording should dominate (it's the
deploy condition), but 30% variation is enough to robustify the text
encoder without destabilizing it.

### Visual augmentation — RandomErasing `p=0.25, scale=[0.02, 0.15]`

torchvision v2 `RandomErasing`. Masks rectangular regions of the image
with value=0 (black). Pairs with the state-aug stack as a visual
regularizer.

`p=0.25`: 25% of frames have erasing applied. Conservative — higher (0.5)
could erase too much of the can or target print.

`scale=[0.02, 0.15]`: erased area is 2-15% of the image. Small enough to
preserve the celebrity prints and the can; large enough to force feature
diversity.

## Frequently asked questions

### Q: Why mean-pool the chunk for the aux head, not just step 0?

Each chunk step is its own action prediction conditioned on the same
(image, state, prompt). Language matters at every step, not just the
"decision moment". Mean-pool supervises all 50 hidden states to encode
the language-image binding, which gives 50× more supervisory positions
per batch and prevents the model from siloing language to one chunk
position.

### Q: Why is HOME state ~3.9 stddev out-of-distribution after normalization?

The lerobot normalizer's stats include the FULL trajectory (HOME → target
→ place → return). Mean(trajectory) is somewhere mid-action, so HOME (the
trajectory START) is several stddevs away. **This is intended behavior**:
HOME IS the deploy-start condition, and training on HOME-masked frames
specifically teaches the policy "from this OOD-relative-to-training state,
use language + image to pick a direction".

### Q: What if cross-prompt Δ stays low even after full training with all augmentation?

The aux head alone reaches 100% classification accuracy in <300 steps but
the action head can still ignore those features. If full training (5k+
steps) doesn't rise:

1. **Increase aux loss weight to 0.5**. Stronger pressure on the shared
   features.
2. **Increase state replacement prob to 0.6**. More aggressive shortcut
   breaking.
3. **Add print-region cutout** (G, currently deferred) — explicitly mask
   non-target prints so the model must use language to identify which
   celebrity is being asked for.
4. **Mix in real-data datasets** (`dataset_v2_*_1`) which have
   per-celebrity action variation baked in by the operator.

### Q: How do I check whether the aux head is actually working at deploy time?

Run `tools/eval3_aux_head_cross_prompt_test.py --checkpoint <path>`. It
feeds 8 specific training frames through 3 prompts and reports the
shoulder_pan spread `Δ_max(prompt)`. Reference values:
- v6_synth_15k baseline (no fix): mean Δ ≈ 0.6°
- Healthy language-conditioned model: Δ ≥ 20° on these frames
- 50-step smoke (no real training): Δ ≈ 2.5° (just confirms pipeline works)

### Q: Should I unfreeze the VLM (text + vision encoders)?

Default: NO. `train_expert_only=True` in SmolVLA config — only the action
expert + projections are trained, VLM stays frozen. The aux head's
gradient flows through the expert's cross-attention K/V projections,
which is sufficient to learn the binding without needing to update the
VLM itself. Unfreezing the VLM costs ~3× memory and risks destabilizing
the well-pretrained text/vision features.

### Q: What's the most important knob to sweep first?

`EVAL3_AUX_POS_LOSS_WEIGHT`. Try `[0.2, 0.3, 0.5]`. Higher = stronger
language pressure but risk of action degradation.

### Q: Does this affect deploy speed?

No. Inference (`policy.select_action`) calls `model.sample_actions`, which
does NOT touch the aux head. The head exists in the checkpoint but adds
zero deploy-time compute.

---

## Architecture reference

Where does `suffix_out` come from? From SmolVLA's
[`modeling_smolvla.py:763-797`](../../.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py):

```
images ─SigLIP────► ┌────────────────────────────┐
"Place the coke..." │   VLM (16 layers, frozen)  │   kv cache    ┌────────────────┐
       embed──────► │   hidden=960               ├──────────────►│ ACTION EXPERT  │
state─proj(960)───► │   produces prefix tokens   │               │ (16 layers,    │
                    └────────────────────────────┘               │  hidden=720,   │
                                                                  │  trained)      │
noisy_actions──►action_in_proj(720)──────────────────────────────► self+cross attn│
chunk_size=50                                                     └──────┬─────────┘
                                                                          │
                                                          suffix_out      │  (B, 50, 720)
                                                          ────────────────┤
                                                          aux head reads ─┤
                                                          here via mean   │
                                                          pool over 50    │
                                                                          ▼
                                                              action_out_proj ──► v_t (flow velocity)
```

The aux head takes `suffix_out.mean(dim=1)` → MLP → 3 logits → CE against
`target_position`. Gradient flows back through the expert's cross-attention
into the VLM kv cache (the K/V projections in the expert are trainable;
the VLM itself stays frozen due to `train_expert_only=True`).

---

## Quick reference card

```bash
# enable in any v6_synth training launcher:
EVAL3_AUX_POS_LOSS_WEIGHT=0.3 ./scripts/run_eval3_smolvla_v6_synth_train.sh

# verify the patch isn't broken:
python tools/eval3_aux_head_unit_tests.py

# verify a trained checkpoint uses language at deploy:
python tools/eval3_aux_head_cross_prompt_test.py --checkpoint <path-or-hub-id>

# disable without removing the patch:
EVAL3_AUX_POS_LOSS_WEIGHT=0 ./scripts/run_eval3_smolvla_v6_synth_train.sh
```

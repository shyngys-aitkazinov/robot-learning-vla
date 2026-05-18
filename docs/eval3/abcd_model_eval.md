# Eval 3 A/B/C/D Model Evaluation

This runbook evaluates only the canonical A/B/C/D checkpoints and produces a
hardware shortlist. It excludes the incomplete `D-12500` checkpoint and the
duplicate D names (`2.5k`, `5k`, `7.5k`, `10k`).

## Offline Ranking

Create the model manifest:

```bash
uv run python tools/eval3_abcd_benchmark.py --manifest-only
```

Run the full offline benchmark:

```bash
uv run python tools/eval3_abcd_benchmark.py \
  --device=mps \
  --n-frames-per-dataset=4
```

Outputs:

- `outputs/eval3_abcd_eval/model_manifest.json`
- `outputs/eval3_abcd_eval/offline_scores.json`
- `outputs/eval3_abcd_eval/OFFLINE_REPORT.md`

The shortlist rule is:

- best A, best B, best C, best D by offline score
- plus any runner-up within 5% of the best overall score
- maximum 5 models

## Dataset Gripper Audit

Run:

```bash
uv run python tools/eval3_dataset_gripper_audit.py
```

Before any new retrain, run the same audit with the proposed label repair:

```bash
uv run python tools/eval3_dataset_gripper_audit.py \
  --simulate-gripper-repair \
  --repair-open-target=55 \
  --repair-open-threshold=20
```

Outputs:

- `outputs/eval3_abcd_eval/dataset_audit.json`
- `outputs/eval3_abcd_eval/DATASET_AUDIT.md`

Flags to pay attention to:

- `dataset_gripper_q90_low`
- `dataset_gripper_q99_low`
- `pre_approach_gripper_not_open`
- `gripper_action_state_lag_high`
- `action_jerk_high`

The default thresholds intentionally match the suspected failure mode:
release gripper q90 must be at least `45 deg`, and q99 must be at least
`50 deg`.

If the simulated repair does not clear those thresholds, do not spend GPU time
on that data recipe. Re-record or materially fix the labels first.

## Retrain Fixes

`scripts/run_eval3_smolvla_aug_train.sh` now defaults to the v8 data recipe:

- repair dataset_v2 gripper-open labels: `EVAL3_GRIPPER_REPAIR=1`
- lift already-open commands to at least `EVAL3_GRIPPER_OPEN_TARGET=55`
- trigger repair only when the source command is already open:
  `EVAL3_GRIPPER_OPEN_THRESHOLD=20`
- smooth arm action labels with `EVAL3_ACTION_SMOOTH_WINDOW=3`
- do not smooth gripper labels: `EVAL3_ACTION_SMOOTH_GRIPPER=0`

For the selector/classifier architecture, keep the motor policy spatial by
enabling slot prompts during retrain:

```bash
EVAL3_SLOT_TASK_AUG=1 EVAL3_SLOT_TASK_P=0.7 ./scripts/run_eval3_smolvla_aug_train.sh
```

That trains the policy to obey `left/middle/right` target prompts. A separate
target selector can then map the celebrity prompt to a slot before deployment;
pass the selected slot with `--target_slot=left|middle|right`.

## Hardware Stage 1

For each shortlisted model, run the three TOY prompts:

- `Place the coke on Taylor Swift`
- `Place the coke on Yann LeCun`
- `Place the coke on Barack Obama`

Use the same command shape for every model:

```bash
python scripts/eval3_vla_deploy.py \
  --robot.type=so101_follower \
  --robot.port=<follower_tty> \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{front: {type: opencv, index_or_path: <cam_idx>, width: 640, height: 480, fps: 30}}' \
  --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \
  --rename_map='{"observation.images.front":"observation.images.camera1"}' \
  --policy.path=<MODEL_REPO> \
  --policy.device=mps \
  --policy.num_steps=20 \
  --policy.n_action_steps=25 \
  --interpolation_multiplier=2 \
  --action_smoothing_alpha=0.25 \
  --max_action_delta_deg=6 \
  --gripper_open_bias_deg=5 \
  --gripper_open_bias_threshold_deg=20 \
  --episode_time_s=20 \
  --fps=30
```

Optional guarded knobs:

- `--policy.num_steps=<n>` increases SmolVLA flow/denoising refinement inside
  each chunk. Start with `20`; if loop FPS collapses, return to `10`.
- `--max_action_delta_deg=<deg>` clamps per-command joint jumps after smoothing.
- `--gripper_open_bias_deg=<deg>` adds a positive bias only when the gripper is
  already commanded open.
- `--gripper_open_bias_threshold_deg=<deg>` controls that open-command cutoff.

True SmolVLA RTC is not the same as `--interpolation_multiplier`. Upstream
RTC requires chunk inference via `predict_action_chunk`; the normal
`select_action()` path rejects RTC. Treat RTC as a separate deploy-loop change,
not a CLI-only tweak.

Every rollout log records:

- raw policy action
- processed robot action before smoothing
- guarded action after smoothing/bias/clamp
- final sent action after interpolation
- first camera frame path

## Hardware Labels

For each rollout, label:

- correct target reached
- can grasped
- can pushed away
- placed on target
- gripper opened enough
- visible shaking

A model enters Stage 2 only if it succeeds on at least 2 of 3 TOY prompts and
has no severe shaking or unsafe motion.

## Hardware Stage 2

Take the top two Stage 1 models. Run nine trials each:

- 3 TOY prints
- 3 held-out photos of the same celebrities
- 3 OOD celebrity/photo conditions

Final model selection should prioritize real task success, then prompt
distinction, gripper reliability, smoothness, and camera robustness.

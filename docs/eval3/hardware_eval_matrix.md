# Eval3 — hardware evaluation matrix (VLA checkpoints)

Use this **after** offline ranking from [`scripts/run_eval3_v7_checkpoint_sweeps.sh`](../../scripts/run_eval3_v7_checkpoint_sweeps.sh) (reports under `outputs/eval3_analysis/sweep_v7_*.md`).

Offline metrics **do not** guarantee tabletop success; this matrix structures apples-to-apples robot trials.

## Baseline deploy recipe

1. Generate flags from the checkpoint’s `train_config.json`:

   ```bash
   uv run python tools/eval3_deploy_flags_from_checkpoint.py /path/to/pretrained_model
   ```

2. Run via [`scripts/run_eval3_vla_deploy_baseline.example.sh`](../../scripts/run_eval3_vla_deploy_baseline.example.sh) (set env vars first), or paste equivalent arguments into `eval3_vla_deploy.py`.

**Required smoothing / timing** (unless ablating deliberately):

- `--policy.n_action_steps=25`
- `--policy.num_steps=20`
- `--interpolation_multiplier=2`
- `--action_smoothing_alpha=0.25`
- `--max_action_delta_deg=6`
- `--gripper_open_bias_deg=5`
- `--gripper_open_bias_threshold_deg=20`
- `--fps=30 --episode_time_s=20`

## Canonical prompts (exact strings)

Use one prompt per rollout (stdin or `--task=`):

| ID | Prompt |
|----|--------|
| Swift | `Place the coke on Taylor Swift` |
| LeCun | `Place the coke on Yann LeCun` |
| Obama | `Place the coke on Barack Obama` |

## Recording matrix

Use one row per **rollout**. Keep lighting, camera index, print layout, and calibration fixed across rows.

| Date | Track | Checkpoint step | Policy path / Hub id | Prompt | Success Y/N | Gripper OK | Shake/jerk notes | Correct celebrity target Y/N | Rollout JSONL path | Notes |
|------|-------|-----------------|----------------------|--------|-------------|------------|------------------|------------------------------|-------------------|-------|
| | A | | | Swift | | | | | | |
| | A | | | LeCun | | | | | | |
| | A | | | Obama | | | | | | |
| | B | | | Swift | | | | | | |
| | … | | | … | | | | | | |

### Post-rollout JSONL checks

Each run writes `outputs/eval3_rollouts/rollout_<UTC>.jsonl` plus `.firstframe.png`.

- **`loop_hz`** / **`sleep_s`** in lines: sustained **`loop_hz` ≪ `--fps`** implies timing jitter (policy/device overload), often correlated with visible shake.
- **`sent_action`** / **`policy_action_processed`**: compare **`wrist_roll.pos`** across prompts on similar scenes (typically **action index 4** for SO-101 — confirm against `dataset.meta.features["action"].names`). Collapsed wrist angles → prompt collapse.

## Gripper data hypothesis

See `tools/eval3_dataset_gripper_audit.py`. Run it with
`--simulate-gripper-repair --repair-open-target=55 --repair-open-threshold=20`
before retraining. If repaired q90/q99 still fail 45/50 deg, prefer **new
demonstrations** over expecting deploy-only fixes.

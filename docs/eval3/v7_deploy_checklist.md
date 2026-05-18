# Eval 3 v7 deploy checklist — things to try at the next robot test

This is the punch-list to walk through at the next hardware run after
the eval-day failure with checkpoints `RobotLearningVLA/eval3-vla-v7-A-smolvla-new-10k`
and `eval3-vla-v7-C-warm-v3-12k`. Symptoms were "doesn't recognize
celebrities" and "grabbing policy is bad". The investigation
(`/Users/shyngys/.claude/plans/ok-now-let-s-try-drifting-yeti.md`) traced
both symptoms to a single configuration bug in the deploy command — the
training pipeline was checked and is clean.

Work the steps in order. Step 1 is the only one that's almost certainly
required. Steps 2–5 are progressively-less-likely contributors; stop
once you have a working policy.

---

## Before connecting the robot — run the diagnostic

Validate every deploy command BEFORE plugging in the arm:

```bash
python tools/eval3_check_deploy_command.py \
    --policy-pretrained-path RobotLearningVLA/eval3-vla-v7-A-smolvla-new-10k \
    --rename-map '{"observation.images.front":"observation.images.camera1"}' \
    --task "Place the coke on Taylor Swift"
```

You want the final line to read `PASS  (cameras=OK, task=OK)`. If it
reads `FAIL`, the script prints the corrected `lerobot-record` invocation
— copy that into your terminal verbatim.

---

## Step 1. **Add `--dataset.rename_map`** (THE fix)

### What to add

```text
--dataset.rename_map='{"observation.images.front":"observation.images.camera1"}'
```

### Why

The v7 checkpoint's `input_features` (from its `config.json` on HF) lists
`observation.images.camera1`, `camera2`, `camera3`, `empty_camera_0`,
`empty_camera_1`. With `empty_cameras=2`, the runtime auto-pads camera2
and camera3 with zeros — but **camera1 still needs a real source**.

Your robot config produces `observation.images.front`. Without the
rename map, lerobot's frame builder never aliases `front` → `camera1`.
The policy ends up reading zero-filled tensors for every camera key. It
literally cannot see the scene — so it cannot recognise celebrities AND
cannot see where the can is. Both reported symptoms collapse to this
single missing flag.

### Confirmation

In the rollout video saved under `~/.cache/huggingface/lerobot/...`
(default destination of `lerobot-record`), you should see the policy
actually responding to scene changes — e.g. moving toward the relevant
celebrity print rather than executing the same stereotyped action
regardless of the prompt.

---

## Step 2. **Add `--policy.n_action_steps=25`** (smoother grabs)

### What to add

```text
--policy.n_action_steps=25
```

### Why

Both v7 checkpoints have `n_action_steps=50` and `chunk_size=50` baked
into the config. At 30 fps the policy re-infers once every ~1.67 s. The
friend-deploy recipe at `docs/eval3/friend_deploy_handoff.md:163`
recommends halving this to 25 (~0.83 s) so the policy can react to
changing scene state mid-chunk. Without this, the grasp can land where
the can WAS rather than where it IS by the time the action chunk
finishes executing.

### Confirmation

Open the rollout MP4 and look for jerky transitions every ~1.7 s
(the chunk-boundary refresh under the default 50). With
`n_action_steps=25` those transitions should halve in spacing and
visibly smooth out.

---

## Step 3. **Type the canonical prompt form** (no "the")

### What to type

```text
Place the coke on Barack Obama
Place the coke on Yann LeCun
Place the coke on Taylor Swift
```

Exact case, no leading "the" before the celebrity name. The diagnostic
catches non-canonical forms and suggests the fix.

### Why

Training rewrote every task string to this canonical form via
`scripts/eval3_dataset_prep.py:TaskAugmenter` at
`EVAL3_TASK_AUG_CANONICAL_P=1.0`. Typing `"Place the coke on the Barack
Obama"` (matching the original recording wording) feeds the tokenizer
a sequence it never saw — model behaviour is undefined.

### Confirmation

Your typed prompt should be one of the three canonical forms above
exactly. The diagnostic script prints the canonical set; copy from
there if in doubt.

---

## Step 4. **If recognition still fails after steps 1–3**: check the eval scene

The v7 checkpoint trained on the v2 corpus (TOY prints — the specific
Slack-PDF photos, cut with no white border, of the 3 celebrities at 3
board positions). It has NEVER seen:

- Different photos of the same celebrities (held-out ID; eval runs 4–6
  per the task spec).
- Other celebrities entirely (OOD; eval runs 7–9).

If you fix steps 1–3 and the policy still misidentifies the celebrity
print at hand, drill into:

1. Are the physical prints on the table the EXACT Slack-PDF cuts the
   model was trained on? Check that ChArUco / replacement prints
   weren't used.
2. Run `python tools/eval3_camera_check.py --camera-index <X>` to
   confirm the camera is pointed at the same view the dataset was
   recorded from (framing, distance, angle). Mismatch here would
   bypass any policy improvement.
3. If you're testing **OOD celebrities**, expect partial failure —
   that's the actual research question of runs 7–9, not a deploy bug.

---

## Step 5. **If grabbing is still sluggish after step 2**: try the team's custom deploy script

Switch from `lerobot-record` to `scripts/eval3_vla_deploy.py`, which
adds `--interpolation_multiplier=2` (inserts midpoint waypoints between
policy actions for smoother motion — a feature `lerobot-record` doesn't
have):

```bash
python scripts/eval3_vla_deploy.py \
    --robot.type=so101_follower \
    --robot.port=<your-port> \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
    --rename_map='{"observation.images.front":"observation.images.camera1"}' \
    --policy.path=RobotLearningVLA/eval3-vla-v7-A-smolvla-new-10k \
    --policy.device=mps \
    --policy.n_action_steps=25 \
    --interpolation_multiplier=2 \
    --dataset_repo_id=RobotLearningVLA/dataset_v2_taylor_swift_left_1_v6_truncated \
    --task='Place the coke on Taylor Swift' \
    --episode_time_s=20 \
    --fps=30
```

Note the differences from `lerobot-record`:
- Flag is `--rename_map` (no `--dataset.` prefix).
- Flag is `--policy.path` (not `--policy.pretrained_path`).
- Flag is `--dataset_repo_id` (no `.` separator).
- Adds `--interpolation_multiplier=2` and `--episode_time_s=20` which
  aren't supported by `lerobot-record`.

Also unlocks `--dry_run` for a no-hardware sanity check.

---

## What was investigated and ruled out

For transparency — these were checked in the audit and are NOT the
cause of the failure (don't waste time on them):

- **The v5 training script (`scripts/run_eval3_smolvla_v5_train.sh`)** —
  clean. No bugs found.
- **Training data scope** — v7's baked-in normalizer shows
  `action.count = 35,874` frames across the 9-dataset v2-truncated
  corpus. The model DID see all 3 celebrities and all 3 positions at
  training time. ("Single-dataset training" was a false positive from
  the first exploration agent.)
- **Stats source at deploy** — lerobot bakes the action/state normalizer
  into the checkpoint as
  `policy_preprocessor_step_5_normalizer_processor.safetensors` and
  reconstructs it at load time. The deploy `--dataset.repo_id` does NOT
  swap normalization stats; it's used only for video format / camera
  metadata. So pointing it at a "wrong" dataset (e.g. an eval recording
  sink) doesn't damage the deployed policy.
- **Dataset labels** — the previous audit
  (`outputs/eval3_audit_dataset_labels/REPORT.md`) confirmed all 24
  training datasets are correctly labelled.

## Open follow-ups (not blocking, do later)

- **Placement-truncation "+1 frame" buffer** in
  `scripts/eval3_dataset_prep.py:421`: when `EVAL3_TRUNCATE_PLACEMENT_MODE="last"`
  (default), trajectories end one frame after the gripper-open signal.
  Could teach the policy to release before the can is physically
  grounded. Not the root cause of THIS failure, but worth a 10-frame
  buffer in a future training run.
- **Default `--rename_map` + `--n_action_steps=25` in `scripts/eval3_vla_deploy.py`**:
  the custom deploy script currently defaults `rename_map={}`. Hardening
  the defaults would make future deploys safer.

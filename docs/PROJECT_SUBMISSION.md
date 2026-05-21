# Course project submission (Project 1 — VLA, Eval 3)

Deadline: **Friday 22.05.2026, 23:59**. Fill the Google Form once.

Replace `teamXX` below with your real team number (e.g. `team42`).

---

## 1. Form field — Summary (< 300 words)

Copy from **[`SUBMISSION_SUMMARY.txt`](../SUBMISSION_SUMMARY.txt)** at the repo root (currently under 300 words). Edit team-specific numbers if needed before paste.

---

## 2. What you are submitting

| Upload bucket | Contents |
|---------------|----------|
| **data** | Training / teleop LeRobot datasets (or export zip of Hub pulls) |
| **repositories** | This git repo as `.zip` + **policy checkpoints** (see §4) |
| **videos** | ≥3 back-to-back inference rollouts per eval; leader + follower visible; no teleop cheat |

Check all three boxes on the form when uploads finish.

---

## 3. Run scripts (graders)

| Script | Eval | Model |
|--------|------|--------|
| [`run_eval3_in_distribution.sh`](../run_eval3_in_distribution.sh) | Runs 1–6 / regimes A–B (Taylor, Obama, Yann) | `RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k` |
| [`run_eval3_ood.sh`](../run_eval3_ood.sh) | Runs 7–9 / regime C (OOD names) | `RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k` |

Both call `./install.sh` with `EVAL3_INSTALL_SMOLVLA_DEPS=1` on first run if `.venv` is missing.

**Mac example (in-distribution):**

```bash
export FOLLOWER_TTY=/dev/tty.usbmodem5B140317761
export EVAL3_POLICY_DEVICE=mps
./run_eval3_in_distribution.sh "Place the coke on Yann LeCun"
```

**Ubuntu TA example (OOD):**

```bash
export FOLLOWER_TTY=/dev/ttyACM0
export EVAL3_POLICY_DEVICE=cuda
./run_eval3_ood.sh "Place the coke on Lionel Messi"
```

Interactive multi-rollout demo (in-distribution only):

```bash
./scripts/run_eval3_demo_cli.sh
```

---

## 4. Checkpoints in the repository zip

Hub IDs (download on first run if `HF_TOKEN` is set):

- **ID:** `RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k`
- **OOD:** `RobotLearningVLA/eval3-smolvla-v16-pinsv5-step5k`

For offline grading without Hub access, **bundle weights into the repo zip**:

```bash
./scripts/stage_submission_checkpoints.sh   # copies Hub snapshots → submission_checkpoints/
```

Then zip the repo (§5). Each checkpoint is ~870 MB.

---

## 5. Build upload zips

```bash
# Set your team id
TEAM=teamXX

# A) Data zip (adjust path to your exported datasets)
# zip -r "${TEAM}-data.zip" path/to/dataset_exports

# B) Repository zip (excludes .venv; includes submission_checkpoints/ if staged)
./scripts/package_course_submission.sh "$TEAM"

# C) Videos zip
# zip -r "${TEAM}-videos.zip" path/to/recorded_mp4s
```

Upload with the **curl** commands from the course form (data / repositories / videos bases + SAS). SAS expires **24 May 2026** — upload before then.

---

## 6. Video checklist

Per eval you claim solved (or partially solved):

- [ ] Leader arm and follower arm both in frame
- [ ] ≥3 consecutive rollouts, **no cut**
- [ ] Policy inference only (not teleoperating follower)
- [ ] Separate video file per eval condition if possible

Suggested naming: `eval3_id_taylor.mp4`, `eval3_id_yann.mp4`, `eval3_ood_messi.mp4`, etc.

---

## 7. Our results (reference)

**In-distribution (`v4slots_expert`, 50k):** 36/45 successes on celebrity×slot matrix (80%). See [`docs/eval3/v4slots_deploy_scorecard.md`](eval3/v4slots_deploy_scorecard.md).

**OOD:** friend-trained `eval3-smolvla-v16-pinsv5-step5k`; deploy via `run_eval3_ood.sh`.

---

## 8. Pre-submit checklist

- [ ] Team number in zip filenames
- [ ] `README.md` + `run_eval3_*.sh` at repo root
- [ ] Checkpoints in zip or documented Hub pull + `HF_TOKEN`
- [ ] All three Azure uploads complete
- [ ] Form summary pasted from `SUBMISSION_SUMMARY.txt`
- [ ] Revoke any HF tokens that were shared in chat

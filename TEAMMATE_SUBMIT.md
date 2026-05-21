# Handoff — teammate submits the course form

**You submit the Google Form once.** Rakhmatillokhon prepared the repo; you upload three zips and paste the summary.

**Deadline:** Friday 22.05.2026 23:59

---

## Step 0 — Get the repo

```bash
git clone https://github.com/shyngys-aitkazinov/robot-learning-vla.git
cd robot-learning-vla
git pull origin main
```

Set your team id everywhere (example `team42`):

```bash
export TEAM=team42   # ← CHANGE THIS
```

---

## Step 1 — Hugging Face login (required for checkpoints)

```bash
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
source .venv/bin/activate
huggingface-cli login
```

---

## Step 2 — Build the three zip files

```bash
source .venv/bin/activate
export TEAM=team42

# A) Videos (~20 MB) — already in submission_videos/
./scripts/package_submission_videos.sh "$TEAM"
# → team42-videos.zip

# B) Repo + policies (~2 GB if checkpoints staged)
./scripts/stage_submission_checkpoints.sh    # downloads 2 Hub models (~15 min)
./scripts/package_course_submission.sh "$TEAM"
# → team42.zip

# C) Data — see submission_DATASETS.txt (Hub dataset list)
#    Option: export from HF or use team’s existing data export path.
# zip -r "${TEAM}-data.zip" <path-to-dataset-export>
```

**If checkpoint download is too slow:** skip `stage_submission_checkpoints.sh`; graders pull models with `HF_TOKEN` using `run_eval3_*.sh` (document in form comments).

---

## Step 3 — Upload to Azure (three curl commands)

Copy the three blocks from the **course Google Form** (data / repositories / videos). Replace `FILE` with:

| Upload | Filename |
|--------|----------|
| data | `${TEAM}-data.zip` |
| repositories | `${TEAM}.zip` |
| videos | `${TEAM}-videos.zip` |

SAS expires **24 May 2026**.

---

## Step 4 — Google Form

| Field | Value |
|-------|--------|
| Team number | `team42` (your real id) |
| Project type | **Project 1 - VLA** |
| Summary | Paste entire contents of **`SUBMISSION_SUMMARY.txt`** |
| Checkboxes | All three uploads done |

---

## Models (for your reference)

| Eval | Script | Hub checkpoint |
|------|--------|----------------|
| Taylor / Yann / Obama | `run_eval3_in_distribution.sh` | `eval3-vla-v6-smolvla-fresh-v4slots-expert-50k` |
| OOD celebrities | `run_eval3_ood.sh` | `eval3-smolvla-v16-pinsv5-step5k` |

Videos: `submission_videos/eval3_video_01_36s.mp4`, `eval3_video_02_75s.mp4` — rename if you know which eval each is.

---

## Questions → Rakhmatillokhon

- Robot port on submit machine: Linux `FOLLOWER_TTY=/dev/ttyACM0` or `/dev/ttyUSB0`
- Data zip: confirm which folder / HF export the team uses

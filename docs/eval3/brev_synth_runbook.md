# Brev runbook — generate + upload the synthetic ChArUco-derived corpus

End-to-end recipe for producing the 9 synthetic LeRobot v3.0 datasets
(`dataset_v3_synth_<celeb>_<position>_2`, 2,250 episodes / ~1.1 M frames /
~7 GB total) on a Brev cloud box, then uploading them to HuggingFace Hub
so the existing training scripts can reference them via
`EVAL3_EXTRA_REPOS=...`.

All commands run from the repo root after `cd ~/robot-learning-rlp`.

---

## 0. Provision the Brev box

Recommended config:
- **vCPUs**: 16+ (the generator scales linearly up to 9 workers; 16-core is
  the sweet spot for 9 parallel composers + encoder threads).
- **RAM**: 16 GB minimum. Each worker holds ~500 video frames in RAM
  per source episode (~440 MB peak per worker × 9 workers = ~4 GB).
- **Disk**: 20 GB free under `~/` (datasets/ + sources + .venv + cache).
- **GPU**: Not required for generation. Required separately for the
  SmolVLA fine-tune that consumes the data afterward.

## 1. Clone + install (one-time, ~10 min)

```bash
git clone git@github.com:shyngys-aitkazinov/robot-learning-vla.git ~/robot-learning-rlp
cd ~/robot-learning-rlp
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
source .venv/bin/activate
```

The `EVAL3_INSTALL_SMOLVLA_DEPS=1` flag adds `transformers`, `accelerate`,
`sentencepiece`, `num2words` on top of the base lerobot install — the
synth generator itself doesn't need them, but the downstream
training script does.

## 2. Pull the source datasets (one-time, ~5 min)

The generator reads from `datasets/dataset_v3_charuco_{left,middle,right}_2`.
Either pull them from HuggingFace or scp them up.

```bash
./scripts/repull_eval3_datasets.sh
# That pulls all 24 training-corpus datasets, including the 3 charuco _2 ones.
# Takes ~5 min on a typical Brev pipe; ~3.5 GB total.
```

To save time/disk, pull ONLY the 3 source datasets:
```bash
huggingface-cli download --repo-type dataset \
    RobotLearningVLA/dataset_v3_charuco_left_2   --local-dir datasets/dataset_v3_charuco_left_2
huggingface-cli download --repo-type dataset \
    RobotLearningVLA/dataset_v3_charuco_middle_2 --local-dir datasets/dataset_v3_charuco_middle_2
huggingface-cli download --repo-type dataset \
    RobotLearningVLA/dataset_v3_charuco_right_2  --local-dir datasets/dataset_v3_charuco_right_2
```

Verify:
```bash
ls datasets/dataset_v3_charuco_{left,middle,right}_2/meta/info.json
```
All three files should exist.

## 3. Pull the TOY celebrity images (one-time, ~1 s)

The 15 in-distribution celebrity JPGs (5 each: Swift / Obama / LeCun) live
in `datasets/in-distribution-eval-3/` and are tracked in git. They are pulled
in step 1 automatically.

```bash
ls datasets/in-distribution-eval-3/{taylor_swift,barack_obama,yann_lecun}/*.jpg | wc -l
# expected: 15
cat datasets/in-distribution-eval-3.json | python3 -m json.tool | head -20
```

## 4. Log into HuggingFace (one-time)

```bash
huggingface-cli login
# Paste a WRITE token from https://huggingface.co/settings/tokens
# Confirm: huggingface-cli whoami should print your username.
```

## 5. Dry-run the generator (sanity check, ~5 s)

Confirms paths and config-grid look right without writing any data.

```bash
python tools/eval3_synth_dataset_gen.py --dry-run
```

Expected output: a 9-line summary "DRY RUN — 9 datasets x 250 eps = 2250 total"
followed by sample configs per dataset. All sources should report `[OK]`.

## 6. (Optional) 2-episode smoke test (~30 s)

Confirm the full write path works for one dataset before kicking off the
hour-long full run:

```bash
python tools/eval3_synth_dataset_gen.py \
    --target-celebs taylor_swift --target-positions left \
    --n-configs-per-dataset 2 --overwrite
```

Expected: writes `datasets/dataset_v3_synth_taylor_swift_left_2/`
with 2 episodes (~6 MB on disk).

Inspect:
```bash
ls datasets/dataset_v3_synth_taylor_swift_left_2/
python3 -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('dataset_v3_synth_taylor_swift_left_2',
                    root='datasets/dataset_v3_synth_taylor_swift_left_2',
                    video_backend='pyav')
print(f'episodes={ds.num_episodes} frames={ds.num_frames} task={ds[0][\"task\"]!r}')"
```

## 7. Full generation + upload (~50–90 min)

```bash
EVAL3_SYNTH_WORKERS=9 \
EVAL3_SYNTH_PUSH_TO_HUB=1 \
EVAL3_SYNTH_OVERWRITE=1 \
  ./scripts/run_eval3_synth_dataset_gen.sh 2>&1 | tee /tmp/synth_gen.log
```

What happens:
1. 9 worker processes spawn (one per output dataset).
2. Each worker reads its source ChArUco dataset, locks 3 homographies per
   source episode, then composes 250 episodes by warping celebrity images
   onto the boards and applying the global lift.
3. As each worker finishes its dataset, it uploads to
   `https://huggingface.co/datasets/RobotLearningVLA/dataset_v3_synth_<celeb>_<position>_2`
   and creates the `v3.0` git tag.
4. Final summary prints total elapsed + per-dataset disk usage + Hub URLs.

Expected output tail:
```
SUMMARY — total elapsed XX.X min
========================================================================
  [OK]   dataset_v3_synth_taylor_swift_left_2     eps=250 frames=133400 disk=765.0 MB
  [OK]   dataset_v3_synth_taylor_swift_middle_2   eps=250 frames=126475 disk=735.0 MB
  ...
Total: 2250 episodes  1118850 frames  6.85 GB on disk
```

If a single worker fails, the others continue — re-run with
`EVAL3_SYNTH_CELEBS=<failed_celeb> EVAL3_SYNTH_POSITIONS=<failed_pos>`
to retry just that one.

## 8. Verify uploads

```bash
for celeb in taylor_swift barack_obama yann_lecun; do
  for pos in left middle right; do
    repo="RobotLearningVLA/dataset_v3_synth_${celeb}_${pos}_2"
    n=$(huggingface-cli repo files "$repo" --repo-type dataset 2>/dev/null | wc -l)
    echo "$repo : $n files on Hub"
  done
done
```

Each repo should report at least 6 files (info.json, stats.json, tasks.parquet,
episodes/.../parquet, data/.../parquet, videos/.../mp4).

## 9. Wire the new datasets into the training command

The downstream training script (`scripts/run_eval3_smolvla_aug_train.sh`)
already supports `EVAL3_EXTRA_REPOS` as a comma-separated list. To add
the 9 synth datasets on top of the existing v2 corpus:

```bash
EVAL3_EXTRA_REPOS="dataset_v3_synth_taylor_swift_left_2,dataset_v3_synth_taylor_swift_middle_2,dataset_v3_synth_taylor_swift_right_2,dataset_v3_synth_barack_obama_left_2,dataset_v3_synth_barack_obama_middle_2,dataset_v3_synth_barack_obama_right_2,dataset_v3_synth_yann_lecun_left_2,dataset_v3_synth_yann_lecun_middle_2,dataset_v3_synth_yann_lecun_right_2" \
  ./scripts/run_eval3_smolvla_aug_train.sh
```

To train ONLY on the synth datasets (drop the v2 corpus), edit the wrapper's
`--dataset.repo_id=...` to point at one of the synth datasets and don't set
`EVAL3_EXTRA_REPOS`. The `eval3_concat_patch` will treat that one as primary
and stats-merge it without needing extras.

## 10. (Optional) Per-dataset sanity check post-upload

```bash
python tools/inspect_lerobot_dataset.py \
  --repo-id RobotLearningVLA/dataset_v3_synth_taylor_swift_left_2
```

Should report 250 episodes, ~133k frames, single task
`"Place the coke on Taylor Swift"`, fps=30, single
`observation.images.front` 480×640 camera.

---

## Troubleshooting

| Symptom | Likely cause + fix |
|---|---|
| `source dataset missing: datasets/dataset_v3_charuco_*_2` | Re-run step 2 (pull sources). |
| `ImportError: cv2.aruco` | Wrong OpenCV build. `uv pip install --force-reinstall opencv-contrib-python` (NOT plain opencv-python). |
| `ValueError: Invalid vcodec 'h264'. Must be one of: ['h264_videotoolbox', ...]` | Brev box doesn't have libx264. Pass `--vcodec libsvtav1` instead (slightly smaller files, slower decode). |
| `HFValidationError: invalid repo_id` during upload | You forgot step 4 (huggingface-cli login). |
| One worker dies, others continue | Re-run for just the missing one: `EVAL3_SYNTH_CELEBS=yann_lecun EVAL3_SYNTH_POSITIONS=middle ./scripts/run_eval3_synth_dataset_gen.sh` |
| OOM after ~2 datasets | Lower workers: `EVAL3_SYNTH_WORKERS=4`. Each worker holds ~500 MB peak. |
| Generation succeeded but training can't see the datasets | Check `huggingface-cli repo files RobotLearningVLA/dataset_v3_synth_... --repo-type dataset` — if files exist but `LeRobotDataset(repo_id)` 404s, the `v3.0` tag is missing. Re-tag: `huggingface-cli repo create-tag <repo> v3.0 --repo-type dataset`. |

---

## What got generated — quick reference

| Property | Value |
|---|---|
| Output datasets | 9 (`dataset_v3_synth_{taylor_swift, barack_obama, yann_lecun}_{left, middle, right}_2`) |
| Episodes per dataset | 250 (5×5×5×2 combinatorial sweep) |
| Total episodes | 2,250 |
| Total frames | ~1.12 M (varies with source episode lengths) |
| Per-episode length | ~450–600 frames (~15–20 s @ 30 fps) |
| Total disk (libx264) | ~7 GB |
| Total disk (libsvtav1) | ~5 GB |
| Task strings | One per dataset, canonical "Place the coke on `<Celeb>`" |
| Action / state schema | 6-DOF `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` float32 |
| Image key | `observation.images.front` (480×640×3 video) |
| fps | 30 |
| LeRobot codebase_version | `v3.0` |

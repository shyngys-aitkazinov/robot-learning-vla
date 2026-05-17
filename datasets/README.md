# `datasets/` — local materialized data

This directory is **git-ignored** (`/.gitignore:datasets/`). It holds writable,
self-contained copies of LeRobot datasets and any third-party data pools used
by the Eval 3 pipeline. Nothing here is authoritative — everything can be
re-populated from the Hub or from public sources below.

Two reasons to materialize datasets here instead of working from
`~/.cache/huggingface/lerobot/hub/...` directly:

1. lerobot 0.5.1+ treats the Hub snapshot cache as **read-only**, so
   `lerobot-record --resume`, custom post-processors, and any code that
   mutates a dataset must point `--dataset.root=` (or the `root=` kwarg to
   `LeRobotDataset`) at a writable working copy.
2. The Hub cache stores files as symlinks into a content-addressable `blobs/`
   pool — convenient for the cache, awkward for distribution / inspection.
   Materialized copies in `./datasets/` are plain regular files
   (`cp -RL` dereferences symlinks at copy time).

## Current contents

| Path | Source | Size | Purpose |
|---|---|---|---|
| `dataset_v3_charuco_left_1/`   | `RobotLearningVLA/dataset_v3_charuco_left_1`   on Hub | 150 MB / 11 ep / 7,368 frames | ChArUco recording, can placed on **left** board |
| `dataset_v3_charuco_middle_1/` | `RobotLearningVLA/dataset_v3_charuco_middle_1` on Hub | 112 MB / 10 ep / 5,513 frames | ChArUco recording, can on **middle** board |
| `dataset_v3_charuco_right_1/`  | `RobotLearningVLA/dataset_v3_charuco_right_1`  on Hub |  93 MB / 10 ep / 4,545 frames | ChArUco recording, can on **right** board |
| `pins-face-recognition.zip`    | [Kaggle: hereisburak/pins-face-recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) | 372 MB (zip) / ~389 MB unzipped / 17,534 images / 105 celebrities | OOD celebrity face pool for the post-processor |

Total ≈ 730 MB on disk.

## Re-populating from scratch

If `datasets/` is wiped, regenerate it with:

```bash
# 1. The three ChArUco recordings — fresh pull from Hub, then dereference symlinks
mkdir -p datasets
for n in dataset_v3_charuco_left_1 dataset_v3_charuco_middle_1 dataset_v3_charuco_right_1; do
  uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset(f'RobotLearningVLA/$n', video_backend='pyav')
"
  SNAP=$(ls -d ~/.cache/huggingface/lerobot/hub/datasets--RobotLearningVLA--$n/snapshots/*/ | head -1)
  cp -RL "$SNAP" datasets/$n
done

# 2. Pins Face Recognition — one idempotent script handles download + unzip + sanity
./scripts/download_pins_faces.sh
# (force a fresh refetch + re-extract with --force)
```

## Pins Face Recognition (celebrity face pool)

A Kaggle dataset of cropped celebrity face photos scraped from Pinterest,
used as the **OOD celebrity image pool** for the ChArUco
post-processor (see [docs/eval3/charuco_pipeline.md §5](../docs/eval3/charuco_pipeline.md#5-post-process-warp-real-celebrities-into-recorded-frames)).

- **Source**: <https://www.kaggle.com/datasets/hereisburak/pins-face-recognition>
- **Layout inside the zip:**

  ```
  105_classes_pins_dataset/
    pins_Adriana Lima/
      Adriana Lima0_0.jpg
      Adriana Lima101_3.jpg
      ...
    pins_Alex Lawther/
      ...
    ...
    pins_tom ellis/
      ...
  ```

- **Stats:** 105 celebrity classes, average ~167 images / class, ~17,534 total
  JPGs, varied aspect ratios (faces typically ~200–800 px on the long side).
- **Coverage:** Pop / film / TV actors (Chris Evans, Margot Robbie, Scarlett
  Johansson, Dwayne Johnson, etc.) plus a handful of public figures (Bill
  Gates, Cristiano Ronaldo). Overlap with the kind of "popular celebrities"
  expected in Eval 3 runs 7–9 (the OOD bucket) is high but **not guaranteed**
  — the TAs may pick celebrities outside this 105-class set.

### Extraction

The download + unzip is wrapped in `scripts/download_pins_faces.sh` (idempotent —
skips download if the zip is on disk, skips extract if the target directory is
already populated, `--force` to redo both):

```bash
./scripts/download_pins_faces.sh
```

Manual extraction without the script:

```bash
unzip -q datasets/pins-face-recognition.zip -d datasets/pins-face-recognition/
ls datasets/pins-face-recognition/105_classes_pins_dataset/ | head
```

### Caveats

- **Identity overlap with TOY is possible.** If the TAs pick a celebrity that
  also exists in Pins, that face counts as in-distribution for any model
  trained on Pins → won't fairly stress the OOD generalisation we're trying
  to validate. When the post-processor is built, **hold out** the three TOY
  identities (Taylor Swift, Yann LeCun, Barack Obama) from the Pins pool
  even if they appear there.
- **Crop variance.** The "faces" are not uniformly framed — some are tight
  head crops, others are head-and-shoulders, a few are full-body. For
  consistent compositing into the ChArUco board area, you'll probably want
  to (a) filter to portrait-aspect images only, or (b) re-crop using a face
  detector before warping. Both are cheap preprocessing passes.
- **License / ethical use.** Pins is a community-scraped dataset; verify
  the licence allows derivative training data before publishing any model
  weights that consumed it. For the course evaluation this is fine; for
  open-sourcing weights, double-check.
- **No face IDs are guaranteed**. The filenames embed an arbitrary integer
  index (e.g. `Adriana Lima101_3.jpg`); they're not face-tracking IDs.
  Don't read meaning into them.

## Notes for future cleanup

- `dataset_v3_charuco_*_1/` are self-contained — `rm -rf datasets/` deletes
  ~730 MB and the Hub-side snapshot in `~/.cache/huggingface/lerobot/hub/...`
  is unaffected. Re-pull via the snippet above.
- The Hub-snapshot cache in `~/.cache/huggingface/lerobot/hub/datasets--*`
  can be cleared independently with `hf cache delete ...` if disk space is
  tight; the `datasets/` copies will keep working.

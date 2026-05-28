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
| `pins-face-recognition.json`   | generated from the above by [`tools/build_pins_metadata.py`](../tools/build_pins_metadata.py) | 50 KB (committed) | Per-celebrity metadata index: canonical names, dir paths, image counts, aspect/resolution stats, TOY-holdout flags |
| `pins-face-recognition-top30.json` | curated subset of the above, ranked by global recognizability, built by [`tools/build_pins_top30.py`](../tools/build_pins_top30.py) | 16 KB (committed) | Top-30 OOD-eligible celebrities — adds `rank` (1–30) and `category` (actor/athlete/tech/music) fields, excludes held-out TOY identities |
| `pins-face-recognition-top50.json` | extended subset (ranks 1–50, identical 1–30 prefix), built by [`tools/build_pins_top50.py`](../tools/build_pins_top50.py) | 27 KB (committed) | Top-50 OOD-eligible celebrities — adds 20 more globally-recognizable names (GoT principals, MCU secondaries, current music acts, A-list veterans) for richer OOD diversity |
| `in-distribution-eval-3/` + `.json` | rendered from the course-provided `in-distribution-eval-3.pdf` at 300 DPI | 7 MB / 15 images / 3 TOY celebrities (5 each) | TOY in-distribution celebrity portraits used for Eval 3 runs 1–6 |
| `out-distribution-eval-3/` + `.json` | curated from [Wikimedia Commons](https://commons.wikimedia.org/) (CC-BY / CC-BY-SA / PD), portrait-cropped + resized to long-edge 2100 px JPG q90 | 7 MB / 15 images / 3 TOY celebrities (5 each) | OOD held-out celebrity portraits (same 3 identities, different shoots) for held-out scoring. Per-image attribution in the JSON's `image_sources` block |
| `algvr-conference/` + `.json` | built by [`tools/build_algvr_conference_dataset.py`](../tools/build_algvr_conference_dataset.py) from the conference site + Wikimedia Commons | 31 MB / 40 images / 34 people (6 organizers + 28 speakers) | Organizers + invited speakers from the [Vision & Robotics for Embodied AI Conference](https://algvr.com/conference/). Photo `_01` is always the conference-site portrait; Wikimedia extras (`_02`..`_04`) for 4 of the most famous (Pollefeys, Malik, Billard, LeCun). `Yann LeCun` is held-out (matches the TOY identity holdout). Re-runnable with `--skip-existing` |

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

### Metadata index (`pins-face-recognition.json`)

Built by `tools/build_pins_metadata.py`, committed (~50 KB) alongside this
README so anyone clone-and-go on the post-processor doesn't have to
re-scan 17 k JPGs. Per celebrity it stores:

| Field | Use |
|---|---|
| `name` | Canonical display name for task-string substitution (`Place the coke on <Name>`). Auto-titlecased from the slug, with manual overrides for known misspellings (e.g. "kiernen shipka" → "Kiernan Shipka"). |
| `slug` | The directory name minus `pins_`. Never edit — it's what's on disk. |
| `dir` | Repo-relative filesystem path for image enumeration. |
| `n_images` | Image count — for sampling weight or coverage filtering. |
| `total_bytes` | On-disk size. |
| `held_out` | True for TOY identities (Taylor Swift, Yann LeCun, Barack Obama) that must be excluded from the OOD training pool. |
| `aspect_counts` / `aspect_portrait_frac` | Portrait/landscape/square histogram. Most Pins celebrities are >95% portrait; filtering on `aspect_portrait_frac > 0.8` is a reasonable default for compositing into the ChArUco board area. |
| `long_edge_px` | min/median/max of the longer image edge — useful for dropping low-resolution images. |

Top-level fields record dataset provenance, `generated_at` timestamp,
the `held_out_identities_target` set, which of those were actually
`held_out_identities_present` in this scan (LeCun isn't in Pins, only
Swift + Obama), and `name_overrides_applied`.

Regenerate after refreshing the dataset:

```bash
python tools/build_pins_metadata.py                    # ~15 s, scans every JPG header
python tools/build_pins_metadata.py --no-image-stats   # ~1 s, skip aspect/resolution fields
```

Typical post-processor usage:

```python
import json, random
from pathlib import Path

# Use the curated top-30 by default for OOD coverage (held-out identities
# already excluded). Drop to the full 105 if you need more variety.
meta = json.load(open("datasets/pins-face-recognition-top30.json"))
pool = [c for c in meta["celebrities"]
        if c.get("aspect_portrait_frac", 0) >= 0.8
        and c["n_images"] >= 30]

# Pick a celebrity per recorded episode; warp one of their images per frame.
celeb = random.choice(pool)
img_path = random.choice(sorted(Path(celeb["dir"]).glob("*.jpg")))
task_str = f"Place the coke on {celeb['name']}"
```

### Curated top-30 subset

`datasets/pins-face-recognition-top30.json` filters the full 105 down to the
30 most globally recognisable celebrities (judgment call — optimised for
"would SmolVLM bind name → face zero-shot"). Same schema as the full file
plus two fields per entry:

| Field | Use |
|---|---|
| `rank` | 1..30, ordered by descending estimated recognizability. |
| `category` | `actor` (22), `tech` (4), `athlete` (2), `music` (2) — coarse bucket for diversity sampling. |

Edit the `TOP30` list in `tools/build_pins_top30.py:38` to add/drop entries
and re-run. The script errors out if any name in `TOP30` doesn't exactly
match a `name` field in the source JSON, or if a TOY identity slips in.

### Caveats

- **Identity overlap with TOY is possible.** If the TAs pick a celebrity that
  also exists in Pins, that face counts as in-distribution for any model
  trained on Pins → won't fairly stress the OOD generalisation we're trying
  to validate. The metadata file flags this for you (`held_out` field) —
  Taylor Swift and Barack Obama are both in Pins; Yann LeCun isn't.
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

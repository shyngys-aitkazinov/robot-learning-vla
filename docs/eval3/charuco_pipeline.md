# Eval 3 — synthetic-on-real data via ChArUco boards

Workflow for recording teleop episodes against **ChArUco fiducial boards** that
stand in for the real celebrity prints, then post-processing each frame to
warp arbitrary celebrity images onto the board area while preserving the
Coke can on top. Aimed at **runs 7–9 of Eval 3** (OOD celebrities not in
TOY) where the policy needs robustness to celebrity identities never seen
during recording.

## Status — experimental; run the prereq probes first

Before recording any ChArUco episodes, validate that synthetic face
diversity is the bottleneck you actually have:

1. **Does SmolVLM already know the OOD celebrities?** Run
   [`tools/eval3_synthetic_ood_test.py`](../../tools/eval3_synthetic_ood_test.py)
   against `RobotLearningVLA/eval3-smolvla-3way-50k-v3-fresh` with held-out
   celebrities composited onto sample frames. If the policy already shifts
   actions toward the correct print zero-shot, identity grounding is solved
   by the VL backbone and the pipeline below buys nothing.
2. **Can you get the same diversity without re-recording?** Consider
   extending [`scripts/eval3_dataset_prep.py`](../../scripts/eval3_dataset_prep.py)
   `PrintShuffleAugmenter` into a `PrintReplaceAugmenter` that warps
   arbitrary headshots into the existing recordings' print-mask polygons
   (`outputs/eval3_masks/<slug>/`). That preserves the multi-print spatial
   layout the current model already trains on — strictly cheaper than
   recording fresh ChArUco episodes.

The ChArUco pipeline only pays off after both above paths are exhausted.
See [Known caveats](#known-caveats) for the design trade-offs.

## Overview

Three CLIs cover the full loop. They live in `tools/` and run inside the
existing project venv — `cv2.aruco` ships with the lerobot install
(opencv-contrib-python ≥ 4.7).

| Stage | Tool | What it does |
|---|---|---|
| 1. Print | [`tools/eval3_make_charuco_board.py`](../../tools/eval3_make_charuco_board.py) | Generates a printable A5 ChArUco board with a central chroma square on an A4 page with crop marks. |
| 2. Verify detection | [`tools/eval3_charuco_check.py`](../../tools/eval3_charuco_check.py) | Live camera preview — confirms markers detect at recording distance and the homography is stable. |
| 3. Verify compose | [`tools/eval3_charuco_compose.py`](../../tools/eval3_charuco_compose.py) | Live camera preview — locks homography on frame 0, warps a target image onto the board, HSV-keys the chroma to preserve the can. |
| 4. Record | `lerobot-record` (unchanged) | Record teleop episodes as usual, with three ChArUco boards in the eval semicircle. |
| 5. Post-process | (not yet committed — see [§5](#5-post-process-warp-real-celebrities-into-recorded-frames)) | Per-episode: detect board on frame 0, freeze homography, warp celebrity image into every frame, restore the can mask. |

Defaults across all three tools assume **3 identical boards**, 130×180 mm
content, 5×7 chessboard squares (~22.8 mm), 60 mm green chroma centre,
DICT_4X4_50.

## 1. Print the boards

The eval prints are nominally A5 but the supplied PDF has celebrity images
at varied aspect ratios in the 128–186 mm width range — measure your cut
prints before deciding ChArUco size. Defaults below match the small end of
that range so the board doesn't dwarf the real prints during recording.

```bash
# Generate three identical boards (left/centre/right) — A5 content on A4 paper
for pos in left centre right; do
  python tools/eval3_make_charuco_board.py --content-mm 130x180 \
      --squares-x 5 --squares-y 7 --chroma-mm 60 \
      --out outputs/eval3_charuco/board_${pos}
done
```

Outputs land at `outputs/eval3_charuco/board_{left,centre,right}.{png,pdf}`.
Boards are byte-identical — three PDFs is for printing convenience; one PDF
printed three times is equivalent.

**Printing:**

- Open each PDF and print on plain A4 at **100% / Actual Size** (uncheck
  "Fit to page" or "Scale to fit" in the printer dialog).
- After printing, hold a ruler against the **top-edge tick marks** on the
  page — adjacent ticks should measure exactly 10 mm apart. If not, the
  print scale was wrong; reprint before cutting.
- Cut along the four corner **crop marks** to obtain clean 130×180 mm
  sheets.

**Why these defaults:**

| Knob | Default | Reason |
|---|---|---|
| `--content-mm 130x180` | smaller than full A5 | Matches the small end of the actual eval-print width range. |
| `--squares-x 5 --squares-y 7` | 17 markers, ~22.8 mm squares, ~16 mm markers | Largest markers we can fit at this paper size — necessary for detection at typical camera distance (~14 px per marker in the live camera view). Larger grids (7×10) produce ~13 mm markers and detection drops sharply. |
| `--chroma-mm 60` | 60 mm green square | Bigger than a 330 ml slim Coke footprint (~60 mm). Leaves a ring of perimeter markers — 12/17 still detect with the can on the chroma. |
| `--dict 4X4_50` | DICT_4X4_50 | Smallest bit pattern → biggest readable marker at distance. Single dictionary across all three boards is fine (boards are spatially separated, see [§4](#4-record-episodes-with-boards-as-celebrity-stand-ins)). |

**Three boards, no chroma-colour or ID tricks needed:** the boards are at
fixed positions per episode (left/centre/right in the semicircle), so the
post-processor disambiguates them by image x-coordinate. ID collisions are
harmless. OpenCV 4.13's Python binding does not expose `CharucoBoard.setIds`
so per-board ID offsets aren't currently supported; using different
dictionaries instead (4×4/5×5/6×6) trades detection range for ID uniqueness
and isn't worth it.

## 2. Verify detection live

After printing, lay one board where an eval print would sit and run the
detection check against your recording camera:

```bash
python tools/eval3_charuco_check.py --camera-index 0
```

A live window opens with marker detections, the projected board outline
(green), and the projected chroma rectangle (blue). The top-left HUD
reports:

- **Markers: N / 17** — green if homography solves, amber if too few markers,
  red if none.
- **Chess corners: M** — interior chessboard intersections used for the
  homography fit.
- **Homography OK  RMS=X.XX px** — reprojection error; under ~1.5 px is good,
  over ~3 px means corner refinement is noisy.

Keys: `q` quits, `s` saves the current annotated frame to
`outputs/eval3_charuco_check/`. If you don't have a window environment, use
`--snapshot --n 5` for batch mode.

**Sanity checks:**

1. Place the board at each of the three semicircle positions and confirm
   ≥6 markers detect and RMS stays under ~1.5 px in all three.
2. Set the Coke can on the green chroma — markers around the can should
   still detect; the green/blue outlines should stay stable.
3. If markers drop below 4 at the table edges, either bring the camera
   closer or coarsen the grid (`--squares-x 4 --squares-y 6` gives ~26 mm
   markers).

## 3. Verify the compositing pipeline live

Same camera, with the board in place — preview what each recorded frame
will look like after post-processing:

```bash
# Synthetic checkered target (orientation tabs make warp errors visible)
python tools/eval3_charuco_compose.py --camera-index 0

# Real celebrity image once geometry checks out
python tools/eval3_charuco_compose.py --camera-index 0 \
    --target-image /path/to/celeb.jpg
```

The tool locks the homography on the first frame where the board detects,
then keeps it frozen — matching how the offline post-processor will work
(one homography per episode, since the prints don't move within an
episode).

**Per-frame pipeline (`compose()` in the script):**

1. `warpPerspective(target_image, H_target, frame_size)` → warped target
   sized to the projected board outline.
2. `board_mask` = polygon of the projected board outline (full 130×180 mm).
3. `chroma_mask` = polygon of the projected 60 mm centre.
4. `green_mask` = HSV pixels in `[hsv_lo, hsv_hi]`.
5. `can_mask = chroma_mask AND NOT green_mask`, with a 5×5 morph-close and
   1-pixel erosion to trim chroma bleed off the can's outline.
6. `result = frame.copy()`; `result[board_mask] = warped[board_mask]`;
   `result[can_mask] = frame[can_mask]`.

Live HUD keys:

- `q` quit
- `s` save current display
- `t` toggle COMPOSITE / RAW
- `m` toggle mask overlays (board = dark green tint, chroma = teal, **can
  = red tint**) — use this to tune HSV thresholds
- `r` re-acquire homography after bumping the camera

**Tuning HSV bounds:**

```bash
python tools/eval3_charuco_compose.py --hsv-lo 30 40 30 --hsv-hi 90 255 255
```

Loosen if green isn't being detected (= can mask bleeds outward over green
chroma). Tighten if highlights on the metallic can top are being classified
as green (= holes in the can mask). With `m` enabled, you can iterate live
in seconds.

## 4. Record episodes with boards as celebrity stand-ins

Once detection and compose both look clean live, record episodes exactly
as in [recording_pilot.md](recording_pilot.md):

- Lay all three printed boards in the eval semicircle (left/centre/right
  positions matching where celebrity prints will go on demo day).
- The Coke can goes on the green chroma square of **one** board per
  episode. Vary which board across episodes so the trained policy sees the
  can placed on each position.
- Operator UX: the green square gives an unambiguous "put the can here"
  target.
- Record with `lerobot-record` as usual. Do not modify the recording
  command — the ChArUco visibility is invisible to lerobot.

**Important — do not push these recordings to the Hub unprocessed.** The
ArUco markers are sacrificial — they only exist for post-processing pose
estimation. Push only the **composited** dataset (next section).

## 5. Post-process: warp real celebrities into recorded frames

**Not yet committed.** The infrastructure exists (`compose()` in
[`tools/eval3_charuco_compose.py`](../../tools/eval3_charuco_compose.py) is
the per-frame function), but the batch processor that walks a recorded
`LeRobotDataset`, locks one homography per episode on frame 0, and writes
a new Hub-uploadable dataset is still TODO. Expected location:
`tools/eval3_charuco_postprocess.py`.

Sketch of the per-episode loop the batch tool will implement:

```python
ds = LeRobotDataset(repo_id_raw, video_backend="pyav")
for ep_idx in range(ds.num_episodes):
    f0, f1 = ds.meta.episodes.iloc[ep_idx][["dataset_from_index", "dataset_to_index"]]
    # Lock homography on frame 0 of this episode.
    frame0 = ds[int(f0)]["observation.images.front"]  # CHW float
    H, _, _, _ = lock_homography(to_bgr(frame0), detector, char_detector, board)
    # Pick a target celebrity for this episode (rotate through the OOD pool).
    target = next_celebrity_image_for(ep_idx)
    for i in range(int(f0), int(f1)):
        frame = ds[i]["observation.images.front"]
        result, _ = compose(to_bgr(frame), target, H,
                            board_w_mm, board_h_mm, chroma_mm,
                            tuple(hsv_lo), tuple(hsv_hi))
        write_frame_to_new_dataset(result, ds[i])
```

Output: a new `LeRobotDataset` named e.g. `RobotLearningVLA/eval3_charuco_v1`
that looks visually like the eval scene (real celebrities, real Coke, real
shadows) but with arbitrary face diversity. Tag it `v3.0` (see
[dataset_matrix.md §Version tags](dataset_matrix.md#version-tags-mandatory))
before training on it.

The same per-frame `compose()` function is what the live tool uses, so
whatever you verify in [§3](#3-verify-the-compositing-pipeline-live) is
exactly what the offline processor will produce.

### Celebrity face pool — Pins Face Recognition (Kaggle)

The "arbitrary face diversity" the post-processor injects comes from a
public celebrity-face corpus stored locally in
[`datasets/pins-face-recognition.zip`](../../datasets/README.md#pins-face-recognition-celebrity-face-pool).
Source: <https://www.kaggle.com/datasets/hereisburak/pins-face-recognition>
(105 celebrities, ~167 images each, ~17.5k images total, ~389 MB unzipped).

Download + extract:

```bash
./scripts/download_pins_faces.sh           # idempotent; --force to refetch
```

Layout once extracted:

```
datasets/pins-face-recognition/
  105_classes_pins_dataset/
    pins_<Celebrity Name>/
      <Celebrity Name><photo#>_<id>.jpg
```

When wiring this into the post-processor, two things matter:

- **Hold out the three TOY identities** (Taylor Swift, Yann LeCun, Barack
  Obama) from the Pins pool when assigning faces to ChArUco episodes —
  otherwise you're training on the identities that runs 1–6 are scored
  against and the experiment no longer measures OOD generalisation.
- **The task-string template `<left marker>` / `<middle marker>` / `<right
  marker>`** that the ChArUco recordings carry (see the inspection of
  `meta/tasks.parquet` in the dataset README) is the substitution slot:
  per episode, pick a celebrity, warp their face onto the relevant board,
  and rewrite the task string from `Place the coke on <left marker>` to
  `Place the coke on <Celebrity Name>`. Same hook that
  `scripts/eval3_dataset_prep.py:TaskAugmenter` already uses.

Caveats (license, identity overlap, crop variance, etc.) are catalogued in
[`datasets/README.md`](../../datasets/README.md#pins-face-recognition-celebrity-face-pool).

## Known caveats

- **Multi-print spatial reasoning.** A policy trained on ChArUco recordings
  sees three identical boards at fixed left/centre/right positions. After
  warping, the celebrities appear at those positions but the policy never
  sees the **specific** celebrity at a **specific** position during training
  (the warp is decoupled from the recorded action). If your network learns
  "name → which position contains that face", that spatial association has
  to come from the policy attending to the warped pixels — not from the
  recording itself. Validate this by checking whether per-prompt actions
  diverge in a post-processed pilot before committing to a full re-record.

- **Compositing realism gap.** The warped celebrity is geometrically
  correct but lacks the subtle pixel statistics of a real photo of a paper
  print (sub-pixel printing artifacts, paper texture, micro-shadows of the
  print's own edges). A trained network may detect the composite as
  out-of-distribution at deploy time. Mitigations: match noise / grain
  statistics, apply mild blur, add per-frame brightness jitter — most of
  these already exist as torchvision transforms in
  [`scripts/run_eval3_smolvla_aug_train.sh`](../../scripts/run_eval3_smolvla_aug_train.sh).

- **Marker detectability at camera distance.** With the recommended
  ~16 mm markers and a typical recording distance, markers project to
  ~14 px — borderline for 4×4 ArUco. If detection on the live camera drops
  out, either move the camera closer (changes the recorded view), make the
  boards larger (drift further from real-print size), or accept fewer
  detected markers per board (4 is the minimum for a homography solve).

- **Target aspect-ratio mismatch.** The compose tool stretches the target
  image to the full 130×180 mm board aspect ratio. Real eval prints have
  varied aspect ratios (128×228 mm for one Taylor Swift print, 186×186 mm
  for another). When the post-processor is built, letterboxing the
  celebrity into the board area with neutral fill is probably the right
  default.

## Cross-references

- [task3_deploy_readiness.md](task3_deploy_readiness.md) — deploy-time
  compatibility checklist (this pipeline must respect the same `rename_map`
  + `policy.empty_cameras=2` contract as live deploy).
- [dataset_matrix.md](dataset_matrix.md) — Hub naming + `v3.0` tag
  requirement for any new dataset produced by the post-processor.
- [train_regimes.md](train_regimes.md) — phased training plan.
- [recording_pilot.md](recording_pilot.md) — the underlying
  `lerobot-record` recipe used in [§4](#4-record-episodes-with-boards-as-celebrity-stand-ins).

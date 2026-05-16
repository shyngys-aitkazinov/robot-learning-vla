# Working with `RobotLearningVLA/taylor_swift_1`

Hub dataset used as the **current Eval 3 anchor** in this repo (single-camera coke + celebrity setup).

## Facts (verify after Hub updates)

Run:

```bash
python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/taylor_swift_1 --video-backend pyav
```

Observed schema (typical):

| Field | Shape / type |
|-------|----------------|
| `observation.images.front` | `float32` `[3, H, W]` (often 480×640 before train resize) |
| `observation.state` | `float32` `[6]` |
| `action` | `float32` `[6]` |
| `task` | string |

## Prompt string alignment

Meta/task text on Hub may read **`Place the coke on the Taylor Swift`** (with **“the”** before the name). Course wording is usually **`Place the coke on Taylor Swift`**.

**Action:** confirm with TAs on Slack, then either:

- record future datasets using the **exact** approved template, or  
- normalize strings in your **training loader** (map known variants to one canonical form).

The BC sanity script ([`scripts/train_eval3_bc_overfit.py`](../../scripts/train_eval3_bc_overfit.py)) does **not** consume language yet; your **SmolVLA / VLA** trainer must use `task` for conditioning.

## Scope vs full Eval 3

This dataset covers **Swift-centric** demonstrations. For TA regimes you still need **Obama / LeCun** TOY prints, held-out photos, and OOD celebrities — record additional Hub repos per [`dataset_matrix.md`](dataset_matrix.md).

## Video decoding on macOS

Use **`--video-backend pyav`** (defaults in repo scripts) so frames load without TorchCodec/FFmpeg dylib issues.

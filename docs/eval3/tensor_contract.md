# Eval 3 — observation / action tensor contract

**Decision:** train and deploy using **LeRobot-native** features produced by `LeRobotDataset` (same schema as `lerobot-record` / replay).

## Discovery

Run on each Hub repo before writing configs:

```bash
python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/taylor_swift_1 --video-backend pyav
```

On macOS, **TorchCodec often fails** unless FFmpeg dylibs match; these repo scripts default to **`--video-backend pyav`** (PyAV via torchvision).

You should see:

- **`features`**: dtype and shapes for each column.
- **`task` histogram**: verifies prompt strings.
- **Sample tensor shapes** for one frame.

## Images

- **Raw storage**: typically **640×480** or similar from SO-101 camera settings.
- **Training resize**: course recommends **256×256** (square crop or resize — **pick one** and freeze; avoid distorting aspect ratio unpredictably).
- **Normalization**: ImageNet mean/std if using torchvision VL backbones; otherwise **[0,1]` float32 — match your OSS VLA recipe.

## Proprio / state

- Key is commonly `observation.state` (shape depends on robot metadata). Use **`meta.stats`** when available for normalization.

## Actions

- Target key is commonly `action` — **must match** replay expectation (same units as teleop pipeline).
- Fit **mean/std** or quantiles on **train split only**; persist JSON next to checkpoints.

## Task / language

- **Language conditioning** should read from batch **`task`** strings or indexed prompts from `dataset.meta.tasks`.
- At inference, **embed the TA stdin line** identically (trimmed).

## Example snapshot (illustrative only)

Values depend on dataset — **always** run the inspector:

```
features: observation.images.front, observation.state, action, timestamp, ...
action shape: (N,)
fps: 30
```

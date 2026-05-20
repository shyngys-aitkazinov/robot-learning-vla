# `fresh_start/` — SmolVLA fine-tuning pipeline

A from-scratch pipeline that fine-tunes [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
on the team's 9 real `dataset_v4_*` teleop datasets, with configurable visual
augmentation (noise, lighting, blur, …). Built on `lerobot` 0.5.1 — treated as
a third-party dependency; nothing in `.venv/` is edited.

## Layout

| File | Role |
|---|---|
| `config.py` | **All configurable options** — draccus dataclasses with defaults |
| `groot_shim.py` | Import-time workaround for lerobot's GROOT crash (see below) |
| `transforms_ext.py` | Custom torchvision-v2 transforms (spatial lighting / vignette) |
| `augmentation.py` | Turns `AugmentationConfig` into lerobot's image-transforms config |
| `merge_datasets.py` | Merges the 9 datasets into one training corpus |
| `train.py` | The training launcher (entry point) |
| `validate.py` | Held-out validation — BC loss per checkpoint on synthetic data |
| `verify.py` | Staged sanity checks + smoke train |

## Quick start

```bash
# 0. one-time: SmolVLA deps + the 9 datasets under datasets/dataset_v4_*
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh

# 1. verify the pipeline end-to-end (stages 1-6, incl. a 10-step smoke train)
uv run python fresh_start/verify.py

# 2. full fine-tune (auto-merges the datasets on first run)
uv run python fresh_start/train.py
```

## Configuring

Every knob lives in `config.py` with a default. Edit it there, or override on
the command line — nested fields use dotted paths:

```bash
uv run python fresh_start/train.py --training.steps=30000 --training.batch_size=96
uv run python fresh_start/train.py --finetune.preset=full
uv run python fresh_start/train.py --augmentation.noise.sigma=0.06
uv run python fresh_start/train.py --augmentation.spatial_lighting.enable=true
uv run python fresh_start/train.py --augmentation.enable=false      # no augmentation
```

### Fine-tune scope — `--finetune.preset`

| Preset | Vision encoder | Action expert | Notes |
|---|---|---|---|
| `expert_only` (default) | frozen | trained | SmolVLA's stock recipe; safest on small data |
| `expert_plus_connectors` | frozen | trained + VLM→expert pathway | more adaptation |
| `full` | trained | trained | most adaptation, most overfit risk |

`--finetune.freeze_vision_encoder` / `--finetune.train_expert_only` override the
preset directly if you want a custom combination.

### Augmentation — `--augmentation.*`

Four groups, each independently toggleable: `lighting` (brightness / contrast /
saturation / hue), `noise` (Gaussian), `blur` (Gaussian defocus), and
`spatial_lighting` (custom vignette + brightness gradient — off by default).
`max_num_transforms` controls how many are sampled per frame. Preview the
result with `verify.py --stage 5` (writes `/ephemeral/outputs/aug_preview/`).

### Training budget — `--training.*`

`steps`, `batch_size`, `save_freq`, `lr`, `device`, `output_dir`, `wandb_*`.
A smoke run is just `--training.steps=10 --training.batch_size=4`.

### Validation — `--validation.*`

lerobot trains pure behavior cloning on the whole corpus — no built-in
validation split. So `train.py` adds one: after training it scores **every
saved checkpoint** (`validate.py`) and prints a ranked table, picking the
lowest-loss checkpoint.

The held-out set is **synthetic `dataset_v3`** data (one small set per
celebrity, capped by `--validation.max_episodes_per_dataset`). Training is on
the *real* `dataset_v4` corpus, so this loss is a **cross-domain** signal — it
rises once the policy overfits real-data quirks instead of learning the task.

```bash
# score the checkpoints of a finished (or interrupted) run, standalone:
uv run python fresh_start/validate.py --training.output_dir=/ephemeral/outputs/<run>
uv run python fresh_start/train.py --validation.enable=false      # skip validation
uv run python fresh_start/train.py --validation.max_episodes_per_dataset=8
```

## How it works

- **GROOT shim** — lerobot 0.5.1 crashes on `import lerobot.policies` when
  `transformers` is installed (a broken dataclass in `lerobot/policies/groot/`).
  `groot_shim.apply()` pre-seeds an inert stub for that module, so every entry
  point applies it before importing lerobot, and training runs **in-process**.
- **Dataset merge** — `make_dataset` can't joint-train a list of datasets, so
  `merge_datasets.py` aggregates the 9 into one corpus at
  `/ephemeral/datasets/dataset_v4_all9/` (built automatically on first train).
- **Augmentation** — applied by lerobot's `ImageTransforms` inside the dataset
  `__getitem__`; noise/blur/lighting are torchvision-v2 transforms selected by
  name, the custom `SpatialLighting` is registered into that namespace.
- **Validation** — lerobot has no validation hook, so `validate.py` runs
  *after* training: it loads each checkpoint, runs the BC loss over the
  held-out synthetic datasets, and ranks them. Augmentation is off for
  validation; the checkpoint's own (training-set) normalization stats are used.

Outputs (merged dataset, checkpoints, previews) go under `/ephemeral/` and are
not committed. Single-camera `observation.images.front` is trained as-is — no
`camera1` rename or `empty_cameras` padding needed for SmolVLA in lerobot 0.5.1.

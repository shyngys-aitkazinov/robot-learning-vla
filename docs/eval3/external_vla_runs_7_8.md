# Eval 3 External VLA Runs 7 and 8

This runbook launches the exact-data FlowerVLA and OpenVLA LoRA experiments:

| Run | Job ID | Model family | Data |
|---|---|---|---|
| 7F | `eval3-flower-new66-50k` | FlowerVLA | new66 only |
| 8F | `eval3-flower-new-old88-50k` | FlowerVLA | new66 + old filtered |
| 7O | `eval3-openvla-lora-new66-50k` | OpenVLA LoRA | new66 only |
| 8O | `eval3-openvla-lora-new-old88-50k` | OpenVLA LoRA | new66 + old filtered |

These runs intentionally do **not** use the v8 SmolVLA label fixes:

- no gripper-open repair
- no arm-label smoothing
- no extra 600-frame cap
- exact new66 / old-filtered frame inclusion

## Data Recipes

`new66` is the nine `dataset_v2_*_v6_truncated` datasets:

- 66 episodes
- 36,865 frames
- `dataset_statistics.json` unnorm key: `eval3_so101_new66`

`new_old88` is `new66` plus old filtered episodes:

- Swift old episodes: `0,4,7,8,9,10,11,12,14,15,16,17,18,19`
- Yann LeCun old episodes: `3,7,9,13`
- Barack Obama old episodes: `5,6,8,17`
- 88 episodes
- 52,546 frames
- `dataset_statistics.json` unnorm key: `eval3_so101_new_old88`

Both recipes use `observation.images.front`, `observation.state`, and padded
7-D actions/proprio. The first six dims are SO-101 joints; dim 7 is constant
zero.

## Preflight

Run this from the repo root before spending GPU time:

```bash
source .venv/bin/activate

uv run python tools/eval3_external_vla_preflight.py \
  --recipe all \
  --chunk-size 10 \
  --batch-size 4 \
  --write-openvla-stats outputs/eval3_external_vla/dataset_statistics.json
```

Fast metadata-only check:

```bash
uv run python tools/eval3_external_vla_preflight.py --recipe all --skip-batch
```

The gate fails if frame counts are not exactly `36,865` and `52,546`, action
names are not the six SO-101 joints, or the padded 7th channel is nonzero.

## FlowerVLA

Use a separate environment:

```bash
uv venv .venv_flower --python 3.10
source .venv_flower/bin/activate

git clone https://github.com/intuitive-robots/flower_vla_calvin external/flower_vla_calvin
pip install -r external/flower_vla_calvin/requirements.txt
pip install huggingface_hub torchcodec pyarrow datasets
```

Smoke 100 steps:

```bash
FLOWER_SRC=external/flower_vla_calvin \
EVAL3_RECIPE=new66 \
EVAL3_TRAIN_STEPS=100 \
EVAL3_BATCH=2 \
./scripts/run_eval3_flower_train.sh
```

Full runs:

```bash
FLOWER_SRC=external/flower_vla_calvin EVAL3_RECIPE=new66 \
  ./scripts/run_eval3_flower_train.sh --push-to-hub --push-intermediate

FLOWER_SRC=external/flower_vla_calvin EVAL3_RECIPE=new_old88 \
  ./scripts/run_eval3_flower_train.sh --push-to-hub --push-intermediate
```

Flower checkpoints are written under `outputs/train/<job>/checkpoints/010000`
through `050000`. Labels stay exact; the trainer normalizes actions/proprio for
model scale and saves raw SO-101 inverse stats beside every checkpoint.
The default pretrained FlowerVLA file is
`mbreuss/flower_vla_pret/360000_model_weights.pt`.

## OpenVLA LoRA

Use a separate OpenVLA checkout and environment:

```bash
uv venv .venv_openvla_train --python 3.10
source .venv_openvla_train/bin/activate

git clone https://github.com/openvla/openvla external/openvla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r external/openvla/requirements-min.txt
pip install peft bitsandbytes accelerate datasets huggingface_hub
```

Smoke one optimizer step:

```bash
OPENVLA_SRC=external/openvla \
EVAL3_RECIPE=new66 \
EVAL3_TRAIN_STEPS=1 \
EVAL3_BATCH=1 \
EVAL3_GRAD_ACCUM=1 \
integrations/openvla/scripts/run_eval3_lora_train.sh --dry-run
```

Full runs:

```bash
OPENVLA_SRC=external/openvla EVAL3_RECIPE=new66 \
  integrations/openvla/scripts/run_eval3_lora_train.sh --push-to-hub --push-intermediate

OPENVLA_SRC=external/openvla EVAL3_RECIPE=new_old88 \
  integrations/openvla/scripts/run_eval3_lora_train.sh --push-to-hub --push-intermediate
```

OpenVLA stores action chunks as flattened `10 x 7 = 70` action tokens because
upstream OpenVLA chooses generation length from the stats vector length. At
inference, reshape output to `[10, 7]`, strip dim 7, then compare or command
only the first six SO-101 joints.

The wrapper starts with batch `4`, grad accumulation `4`. If CUDA OOM is found
in the log, it retries batch `2`, accumulation `8`; if that still OOMs, it
retries with 4-bit quantization.

## Compute Note

True four-way parallel training needs multiple GPUs. On one L40S 48 GB, run
the two OpenVLA jobs one at a time; FlowerVLA may fit concurrently with a small
batch, but do not assume that until the 100-step smoke run confirms VRAM.

## Hardware Gate

Do not connect motors to FlowerVLA/OpenVLA outputs until offline checks pass:

- action output reshapes to `[10, 7]`
- padded dim 7 is near zero
- first six joint dimensions are in the SO-101 degree range after unnormalizing
- same image with Taylor Swift / Yann LeCun / Barack Obama prompts changes the
  first action chunk
- first 3 seconds at `motion_gain=0.25` are stable before any full rollout

FlowerVLA checkpoints are not native LeRobot `PreTrainedPolicy` checkpoints, so
use the dedicated adapter instead of `scripts/eval3_vla_deploy.py`. On the robot
machine, first verify the checkpoint loads without touching motors:

```bash
python scripts/eval3_flower_deploy.py \
  --robot.type=so101_follower \
  --robot.port=<follower_tty> \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{front: {type: opencv, index_or_path: <cam_idx>, width: 640, height: 480, fps: 30}}' \
  --checkpoint_path=RobotLearningVLA/eval3-flower-new66-50k \
  --flower_src=external/flower_vla_calvin \
  --task="Place the coke on Taylor Swift" \
  --device=auto \
  --dry_run=true
```

If the dry run passes, run a guarded 3-second motor test:

```bash
python scripts/eval3_flower_deploy.py \
  --robot.type=so101_follower \
  --robot.port=<follower_tty> \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{front: {type: opencv, index_or_path: <cam_idx>, width: 640, height: 480, fps: 30}}' \
  --checkpoint_path=RobotLearningVLA/eval3-flower-new66-50k \
  --flower_src=external/flower_vla_calvin \
  --task="Place the coke on Taylor Swift" \
  --device=auto \
  --episode_time_s=3 \
  --fps=5 \
  --motion_gain=0.25 \
  --action_smoothing_alpha=0.35 \
  --max_action_delta_deg=4 \
  --allow_live_motors=true
```

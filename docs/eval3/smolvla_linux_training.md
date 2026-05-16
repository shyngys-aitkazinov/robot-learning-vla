# SmolVLA fine-tuning on `taylor_swift_1` (Linux GPU / Brev)

Course requirement: a **VLA** with a **pretrained vision–language backbone**. In LeRobot **0.5.x**, [`SmolVLA`](https://github.com/huggingface/lerobot) is the built-in option (`policy.type=smolvla`, pretrained e.g. [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base)).

## Why not install `transformers` in the default macOS venv?

Installing `transformers` into the same venv can break **`lerobot-train` imports** (known interaction with the optional **GROOT** stack’s dataclass config when `transformers` becomes available). Keep **teleop / record / replay / BC sanity scripts** on the lean venv; do **heavy VLA fine-tuning** on **Linux + GPU** (Brev) in a **dedicated** environment or container.

## Recommended: separate training environment on Brev

On your GPU instance:

```bash
uv venv --python 3.12 .venv-train
source .venv-train/bin/activate
uv pip install "lerobot[smolvla]"   # or matching HF docs for your lerobot version
# If extras fail on macOS-style pins, use: uv pip install lerobot transformers accelerate sentencepiece
```

Then fine-tune from `lerobot/smolvla_base`, **disable Hub push** unless you intend to publish weights:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id=RobotLearningVLA/taylor_swift_1 \
  --dataset.video_backend=pyav \
  --job_name=eval3_smolvla_taylor_swift_1 \
  --steps=50000 \
  --batch_size=8 \
  --policy.device=cuda \
  --policy.compile_model=false \
  --output_dir=outputs/train/eval3_smolvla_taylor_swift_1
```

Tune `--steps`, `--batch_size`, learning-rate presets, and **image transforms** per [`train_regimes.md`](train_regimes.md).

## Camera-key mismatch (single `front` vs multi-camera checkpoints)

`smolvla_base` often expects **multiple** image keys (e.g. `camera1`, `camera2`, `camera3`). Your dataset exposes **`observation.images.front`** only. Fixes (pick one, upstream-dependent):

1. **Team recording convention**: duplicate the same physical camera into additional keyed streams in LeRobot (if supported by your recording config).  
2. **Use a SmolVLA checkpoint** trained for **your embodiment / cam layout** if available on the Hub.  
3. **Consult LeRobot docs / Discord** for the current **`rename_map`** or processor recipe for single-camera finetuning.

The [`tensor_contract.md`](tensor_contract.md) inspector clarifies exact keys before you match a policy.

## After training

Point [`scripts/eval3_rollout.py`](../../scripts/eval3_rollout.py) (or `lerobot-eval` + real robot) at the exported **`pretrained_model`** directory once your checkpoint matches the deployment observation pipeline.

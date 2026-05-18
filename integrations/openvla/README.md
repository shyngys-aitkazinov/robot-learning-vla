# OpenVLA × Eval3 (optional integration)

This subtree is **experimental** and **does not replace** the canonical SmolVLA stack (`scripts/eval3_vla_deploy.py`, `scripts/train_eval3_smolvla.py`). Keep LeRobot training/deploy as-is.

## Why a separate environment?

Upstream OpenVLA pins **`transformers==4.40.1`** ([`requirements-min.txt`](requirements-min.txt)), which conflicts with the evolving Hugging Face stack used by **`lerobot` + SmolVLA**. Use a dedicated venv:

```bash
# From repository root (robot-learning-vla/)
uv venv .venv_openvla --python 3.12
source .venv_openvla/bin/activate   # or: .venv_openvla/bin/python ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # Linux CUDA example
pip install -r integrations/openvla/requirements-min.txt
```

macOS (CPU/MPS smoke tests only — 7B will be slow/huge RAM):

```bash
uv venv .venv_openvla --python 3.12
source .venv_openvla/bin/activate
pip install torch torchvision
pip install -r integrations/openvla/requirements-min.txt
```

### Verification gate

```bash
.venv_openvla/bin/python -c \
  "from transformers import AutoProcessor; AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True); print('ok')"
```

Full weight download occurs on first `AutoModelForVision2Seq.from_pretrained`.

## Safety / embodiment warning

Public checkpoints denormalize actions with **`unnorm_key`** tied to **Open-X datasets** (e.g. Bridge). **There is no official `unnorm_key` for SO-101 Eval3.** Passing a random key to debug shapes is OK; **driving torque using Bridge stats on SO-101 is unsafe.**

Use **`scripts/predict.py`** with explicit **`--unnorm-key`** only after reading the model’s **`dataset_statistics.json`** on Hugging Face. For robot motion you **must** complete the embodiment bridge (`configs/embodiment_stats.json` derived from your data + clamps in `openvla_eval3/mapping.py`).

## Scripts (run with `.venv_openvla` active)

| Script | Purpose |
|--------|---------|
| [`scripts/predict.py`](scripts/predict.py) | Single image → raw action tensor |
| [`scripts/capture_infer.py`](scripts/capture_infer.py) | Same as predict with Eval3-oriented CLI defaults |
| [`scripts/deploy_stub.py`](scripts/deploy_stub.py) | Timing-only loop over PNG folder → JSONL (no motors) |
| [`scripts/offline_probe.py`](scripts/offline_probe.py) | Variance across prompts on many images (manifest / glob) |
| [`scripts/http_client.py`](scripts/http_client.py) | Optional REST client for upstream `vla-scripts/deploy.py` |

Outputs default to **`outputs/eval3_rollouts/openvla_*.jsonl`** at repo root (same gitignore bucket as other Eval3 artifacts).

## Optional remote inference

On a GPU server, clone [openvla/openvla](https://github.com/openvla/openvla) and run upstream **`vla-scripts/deploy.py`** (see their README). Point **`http_client.py`** at your server URL.

## Fine-tuning path

See [`docs/finetune_future.md`](docs/finetune_future.md).

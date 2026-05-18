# FlowerVLA → lerobot integration notes (v7 Phase 2)

Research conducted 2026-05-17 while Train A (SmolVLA) is running on Brev.

## Key facts about FlowerVLA

- **Paper**: "FLOWER: Democratizing Generalist Robot Policies with Efficient VLA Flow Policies", CoRL 2025. arXiv 2509.04996. KIT Intuitive Robots Lab + Microsoft Research.
- **Repos** (both MIT-licensed):
  - Pretraining: <https://github.com/intuitive-robots/flower_vla_pret>
  - Finetuning (CALVIN + LIBERO): <https://github.com/intuitive-robots/flower_vla_calvin>
- **Pretrained checkpoint**: `mbreuss/flower_vla_pret` on HuggingFace. Pretrained on CALVIN + OXE robotic manipulation.
- **Local clone**: `/tmp/flower_research/flower_vla_calvin/`

## Architecture summary

- **VLM**: Florence-2-large (pruned by ~50%). Total params ~950M.
- **Action head**: Rectified flow-matching transformer DiT, 18 layers, 16 heads, dim 1024.
- **Action chunking**: `act_window_size=10` (we'd match SmolVLA's chunk_size if changed).
- **Inference**: 4 denoising steps.
- **Inputs** (from `conf/model/flower.yaml`):
  - `obs["rgb_obs"]["rgb_static"]`: (B, T, 3, H, W) — primary camera
  - `obs["rgb_obs"]["rgb_gripper"]`: optional, gated by `use_second_view=False`
  - `goal["lang_text"]`: list of strings
  - `obs["robot_obs"]`: (B, T, lowdim_obs_dim) — only if `use_proprio=True`
- **Output**: `(B, act_window_size, action_dim)` flow-matching action sequence.

## SO-101 fit

- `action_dim=7` ✓ matches SO-101's 6 motors + 1 gripper.
- `lowdim_obs_dim=7` ✓ matches SO-101 state vector.
- Single camera ✓ supported via `use_second_view=False`.

## Known issues / gotchas

1. **`forward()` ignores `use_second_view`** — line 811 of `flower/models/flower.py` always reads `rgb_gripper`. Workaround: pass a zero placeholder of correct shape, OR call `encode_observations(batch)` directly and skip `forward`.
2. **Pretrained action stats are CALVIN/OXE-conditioned**. We need to either (a) re-init the action projection head and lose the pretrained motor priors, or (b) hope normalization-aware loading works. Validate by checking checkpoint state dict keys for action-head components.
3. **Hydra-driven model construction**. We bypass Hydra by directly instantiating `FLOWERVLA(**hyperparams)` with the canonical config from `conf/model/flower.yaml`.
4. **Lightning training step**. Their `training_step` is a standard PL hook. To use without Lightning, we can call `encode_observations(batch)` + `rf_loss(features, actions)` directly in our own loop.

## Integration plan (3 stages)

### Stage 1 — environment + model loading (4 hours)

```bash
# On Brev (separate venv to avoid breaking lerobot)
uv venv .venv_flower --python=3.10
source .venv_flower/bin/activate
uv pip install -r /tmp/flower_research/flower_vla_calvin/requirements.txt
uv pip install transformers accelerate sentencepiece huggingface_hub torchcodec pyarrow datasets
```

Then a minimal load test:

```python
from flower.models.flower import FLOWERVLA
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download("mbreuss/flower_vla_pret", "model_weights.pt")
model = FLOWERVLA(
    vlm_path="microsoft/Florence-2-large",
    use_second_view=False, second_view_key=None,
    lowdim_obs_dim=7, action_dim=7, act_window_size=10,
    multistep=10, num_sampling_steps=4,
    use_proprio=False, return_act_chunk=False,
    dit_dim=1024, n_heads=16, n_layers=18,
    use_rope=True, query_seq_len=100,
    sampling_type="uniform",
    load_pretrained=True, pretrained_model_path=ckpt_path,
)
```

### Stage 2 — LeRobotDataset → FlowerVLA batch adapter (4-6 hours)

New file: `scripts/eval3_flower_dataset.py`. Wraps a `LeRobotDataset` with a sliding-window iterator that returns the dict format `encode_observations` expects:

```python
{
    "rgb_obs": {
        "rgb_static": torch.tensor[(B, T=1, 3, H, W)],  # we only have one frame at a time
    },
    "actions": torch.tensor[(B, act_window_size, 7)],   # next-N action chunk
    "lang_text": [task_string] * B,
}
```

Action chunks: for each frame `t` in the dataset, fetch actions `[t, t+1, ..., t+9]` (zero-pad at episode ends). Reuse the existing `Eval3PrepDataset` truncation + augmentation.

### Stage 3 — minimal trainer (2-3 hours)

New file: `scripts/train_eval3_flower.py`. Bypasses lerobot's training loop AND Lightning:

```python
model = build_flower(...)
loader = DataLoader(eval3_flower_dataset(...), batch_size=8)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05, betas=(0.9, 0.95))
for step, batch in enumerate(loader):
    features = model.encode_observations(batch)
    loss, _ = model.rf_loss(features, batch["actions"])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10000 == 0:
        torch.save(model.state_dict(), f"ckpt_{step}.pt")
```

### Stage 4 — inference adapter for synthetic-OOD test (2 hours)

`tools/eval3_synthetic_ood_test.py` needs to support a FlowerVLA inference path: load `FLOWERVLA(...)`, run `model.step(obs, goal)` per prompt, extract action. Should be a small switch keyed on `--policy.type`.

## Realistic timeline

| stage | effort | running total |
|---|---|---|
| Stage 1: env + load test | 4 h | 4 h |
| Stage 2: dataset adapter | 6 h | 10 h |
| Stage 3: trainer | 3 h | 13 h |
| Stage 4: inference adapter | 2 h | 15 h |
| Debugging buffer | 8 h | **23 h** = ~3 working days |

Plus 2 × 3 h Brev training runs for Trains D and E.

## Decision point

The integration is doable but expensive on calendar time. Confirm with the user whether the SmolVLA-only results (Trains A, B, C) are convincing enough to skip Phase 3, or whether the architectural-comparison angle is worth the additional days.

If the user pushes through: the immediate next step is to set up `.venv_flower` on a Brev instance and run the minimal load test to confirm `mbreuss/flower_vla_pret` weights load cleanly into the rebuilt `FLOWERVLA` instance.

# Eval 3 — VLA foundation (project docs)

Course task: **place the slim Coke can on the correct celebrity print** given a prompt  
`Place the coke on [celebrity name]`, within **20 s**, **one camera**, **no cloud/YOLO at inference**.

This folder implements the **foundation plan** (scene, data regimes, recording QA, compute, OSS pointers, training phases, rollout CLI). Robot-learning-vla stays the **team source of truth**; `lerobot` remains a PyPI dependency ([WORKSPACE.md](../WORKSPACE.md)).

| Doc | Purpose |
|-----|---------|
| [scene_spec.md](scene_spec.md) | Measurable tabletop layout, jitter, camera framing |
| [prompt_protocol.md](prompt_protocol.md) | Canonical instructions + TA confirmation checklist |
| [dataset_matrix.md](dataset_matrix.md) | Hub naming, regimes A/B/C, `v3.0` tags |
| [recording_pilot.md](recording_pilot.md) | `lerobot-record` checklist + permutation pilot |
| [tensor_contract.md](tensor_contract.md) | Observation/action keys, resize (**256²**), normalization |
| [compute_budget.md](compute_budget.md) | Brev/GPU budgeting, batch vs VRAM, gradient accumulation |
| [oss_baselines.md](oss_baselines.md) | FlowerVLA / SmolVLA / Smol-0-VLA / TinyVLA spike guide |
| [train_regimes.md](train_regimes.md) | Phased training A→B→C + confusion auditing |
| [rollout.md](rollout.md) | Demo-day CLI contract + robot wiring notes |

## Repo scripts / tools

```bash
source .venv/bin/activate

# Inspect any LeRobot v3 Hub dataset (requires HF auth). Default --video-backend pyav for macOS.
python tools/inspect_lerobot_dataset.py --repo-id RobotLearningVLA/taylor_swift_1

# Minimal BC overfit gate (vision + proprio → action), single episode
python scripts/train_eval3_bc_overfit.py --repo-id RobotLearningVLA/taylor_swift_1 --episodes 0 --steps 1500 --video-backend pyav

# Demo CLI: read instruction, 20 s timed loop (mock obs from dataset frame)
python scripts/eval3_rollout.py --policy-path outputs/eval3_bc_overfit/best.pt \
  --mock-dataset-repo RobotLearningVLA/taylor_swift_1 --mock-frame-index 0 --device cpu

# Parameter count for bonus tracking
python tools/count_inference_params.py --checkpoint outputs/eval3_bc_overfit/best.pt
```

Wire `--policy-path` to your real VLA checkpoint when ready; the BC script produces a **sanity** checkpoint shape only (not a compliant VLA by itself).

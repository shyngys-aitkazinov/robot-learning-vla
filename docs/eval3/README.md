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
| [taylor_swift_1.md](taylor_swift_1.md) | **Current Hub anchor**: schema, prompt wording, scope gaps |
| [smolvla_linux_training.md](smolvla_linux_training.md) | **Course-compliant VLA** fine-tune (`lerobot-train` + SmolVLA on GPU/Linux) |
| [compute_budget.md](compute_budget.md) | Brev/GPU budgeting, batch vs VRAM, gradient accumulation |
| [oss_baselines.md](oss_baselines.md) | FlowerVLA / SmolVLA / Smol-0-VLA / TinyVLA spike guide |
| [train_regimes.md](train_regimes.md) | Phased training A→B→C + confusion auditing |
| [rollout.md](rollout.md) | Demo-day CLI contract + robot wiring notes |
| [task3_deploy_readiness.md](task3_deploy_readiness.md) | **Pre-flight checklist**, SmolVLA **`front`→`camera1`** workaround, rubric + hardware |
| [charuco_pipeline.md](charuco_pipeline.md) | **Experimental** — ChArUco fiducial boards as celebrity stand-ins for synthetic-on-real data; print/verify/compose CLIs and post-processing sketch |

## Repo scripts / tools

```bash
source .venv/bin/activate

# Inspect defaults to RobotLearningVLA/taylor_swift_1
python tools/inspect_lerobot_dataset.py

# Minimal BC overfit gate — defaults to same Hub repo + pyav
python scripts/train_eval3_bc_overfit.py --episodes 0 --steps 1500

# Or wrapper with env overrides:
chmod +x scripts/run_eval3_bc_taylor_swift.sh
EVAL3_EPISODES="0" EVAL3_BC_STEPS="1500" ./scripts/run_eval3_bc_taylor_swift.sh

# Demo CLI (stdin instruction); loads mock frames from checkpoint meta.repo_id by default
python scripts/eval3_rollout.py --policy-path outputs/eval3_bc_overfit/best.pt --mock-frame-index 0

# Parameter count for bonus tracking
python tools/count_inference_params.py --checkpoint outputs/eval3_bc_overfit/best.pt

# SmolVLA / single-camera compatibility (prints rename_map + empty_cameras suggestion)
python tools/eval3_smolvla_compat.py

# Inspect tasks vs Eval 3 prompt prefix
python tools/inspect_lerobot_dataset.py --eval3-task-prefix "Place the coke on"

# SmolVLA fine-tune wrapper (rename_map + empty_cameras for observation.images.front)
chmod +x scripts/run_eval3_smolvla_train.sh
EVAL3_TRAIN_STEPS=50000 ./scripts/run_eval3_smolvla_train.sh

# Or invoke Python entry directly (same shim as lerobot-train):
# uv pip install transformers accelerate sentencepiece
python scripts/train_eval3_smolvla.py ...

# Closed-loop deploy — use same rename_map as training (see task3_deploy_readiness.md)
python scripts/eval3_vla_deploy.py --robot.type=so101_follower ... \
  --rename_map '{"observation.images.front":"observation.images.camera1"}' \
  --policy.path=outputs/train/eval3_smolvla/checkpoints/<STEP>/pretrained_model \
  --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \
  --task='Place the coke on Taylor Swift' --episode_time_s=20
```

Wire **`eval3_rollout`** / BC checkpoints only for **pipeline sanity**. Real Eval 3 uses **SmolVLA** weights + **`eval3_vla_deploy`** on the arm.

### ChArUco synthetic-on-real pipeline (experimental)

See **[charuco_pipeline.md](charuco_pipeline.md)** for the full workflow + caveats.

```bash
# 1. Print three identical A5 boards on A4 with crop marks
for pos in left centre right; do
  python tools/eval3_make_charuco_board.py --content-mm 130x180 \
      --squares-x 5 --squares-y 7 --chroma-mm 60 \
      --out outputs/eval3_charuco/board_${pos}
done

# 2. Live detection check (camera 0 pointed at a printed board)
python tools/eval3_charuco_check.py --camera-index 0

# 3. Live compositing preview — locks homography on frame 0, warps target image onto board
python tools/eval3_charuco_compose.py --camera-index 0 --target-image /path/to/celeb.jpg
```

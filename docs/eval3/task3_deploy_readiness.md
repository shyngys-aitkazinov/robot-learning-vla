# Eval 3 — Deploy readiness (Task 3 checklist)

Use this before hardware runs with **`RobotLearningVLA/taylor_swift_1`** (or whatever Hub repo you trained on). It mirrors what must hold for **`scripts/train_eval3_smolvla.py`** → **`scripts/eval3_vla_deploy.py`** to match the course rollout (**prompt**, **20 s**, TOY / held-out / OOD regimes over nine graded demos).

---

## 1. Trained SmolVLA checkpoint

- You need a directory **`.../pretrained_model`** produced by fine-tuning (typically under `outputs/train/<job>/checkpoints/<step>/pretrained_model`).
- Train via **`./scripts/run_eval3_smolvla_train.sh`** (bundles **`rename_map`** + **`policy.empty_cameras=2`** for **`observation.images.front`**) or **`python scripts/train_eval3_smolvla.py`** manually (same flags as **`lerobot-train`**; applies **`scripts/eval3_lerobot_shim.py`**).
- Until this exists, **`eval3_vla_deploy`** has no weights that match your dataset embodiment.

---

## 2. Transformers stack (same venv as train + deploy)

Install once in the project venv you use for teleop/train/deploy:

```bash
uv pip install transformers accelerate sentencepiece
```

Or re-run bootstrap with:

```bash
EVAL3_INSTALL_SMOLVLA_DEPS=1 ./install.sh
```

SmolVLA depends on **`transformers`**; always enter training/deployment via **`scripts/train_eval3_smolvla.py`** or **`scripts/eval3_vla_deploy.py`** paths that apply **`scripts/eval3_lerobot_shim.py`** before **`lerobot.policies`** imports (do **not** rely on raw **`lerobot-train`** unless you patch imports the same way).

---

## 3. Dataset ↔ policy compatibility

### Resolved defaults for **`taylor_swift_1`** + **`smolvla_base`**

Upstream **`lerobot/smolvla_base`** declares **`observation.images.camera1`–`camera3`** (256²). Your Hub dataset exposes **`observation.images.front`** only. Supported workaround:

1. **`--rename_map`** maps **`front` → `camera1`** (same tensor feeds the first expected view).
2. **`--policy.empty_cameras=2`** lets SmolVLA pad **camera2** and **camera3** during **`prepare_images`** (black / masked placeholders).

Print exact flags anytime:

```bash
python tools/eval3_smolvla_compat.py --repo-id RobotLearningVLA/taylor_swift_1
```

Train with the team wrapper (same rename + empty cameras baked in; override via env vars inside the script header):

```bash
chmod +x scripts/run_eval3_smolvla_train.sh
./scripts/run_eval3_smolvla_train.sh
```

Deploy **must** use the **same** **`rename_map`** as training (`eval3_vla_deploy` already applies **`rename_stats`** for normalization alignment).

### Task / language

- Episodes must carry **`task`** text aligned with **exact live prompts**, e.g. **`Place the coke on Taylor Swift`** (and the Obama / LeCun variants you record). Mismatch between training strings and demo strings hurts generalisation abruptly.

Check prompts quickly:

```bash
python tools/inspect_lerobot_dataset.py --eval3-task-prefix "Place the coke on"
```

### Observation keys and stats

- Training and deploy must share the same **rename_map**, **`policy.empty_cameras`** (stored in the checkpoint after training), and **`dataset_repo_id`** for Hub stats.

### QA on disk

- **`python tools/inspect_lerobot_dataset.py`** — verify keys, **`task`** distribution, and episode sanity before long trains.

---

## 4. Coverage vs rubric (nine runs)

- **`taylor_swift_1`** alone does **not** span all nine graded conditions:
  - **Runs 1–3**: TOY prints (exact Slack PDF, trimmed border).
  - **Runs 4–6**: Same celebrities, **held-out** photos not given on Slack.
  - **Runs 7–9**: **OOD** popular celebrities (not in TOY set).

For **early testing**, TOY-style layouts plus prompts matching training are enough to see whether **pick-and-place toward the correct print** works **sometimes**.

For **full marks**, collect data (and rehearse layouts) that reflect **regimes B/C**, not only TOY.

---

## 5. Hardware + scene rehearsal

- **Robot**: SO101 follower (or team embodiment), correct **`lerobot-calibrate`** IDs (**e.g. `my_awesome_follower_arm`**), stable USB/power for servos.
- **Camera**: Same placement/resolution band as training; bandwidth matters at ~30 Hz control.
- **Scene**: DIN A5 colour prints, **no white border** after cutting; **330 ml slim** Coke can (crushable sides but **stands**); semicircle layout + **can centred** per spec.
- **Safety**: Comfortable start pose, e-stop mindset, first runs at reduced speed / supervision if your stack allows it.
- **Keyboard**: LeRobot uses **arrow / Esc** shortcuts in some flows — macOS **Accessibility** permission for the terminal if keys must register.

---

## Quick command references

Train (wrapper applies **`rename_map`** + **`policy.empty_cameras=2`** for **`front`** datasets):

```bash
./scripts/run_eval3_smolvla_train.sh
```

Deploy (after checkpoint exists):

```bash
python scripts/eval3_vla_deploy.py \
  --robot.type=so101_follower \
  --robot.port=... \
  --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --robot.id=my_awesome_follower_arm \
  --dataset_repo_id=RobotLearningVLA/taylor_swift_1 \
  --rename_map '{"observation.images.front":"observation.images.camera1"}' \
  --policy.path=outputs/train/eval3_smolvla/checkpoints/<STEP>/pretrained_model \
  --policy.device=mps \
  --task='Place the coke on Taylor Swift' \
  --episode_time_s=20
```

Load-only check without connecting the arm:

```bash
python scripts/eval3_vla_deploy.py ... --dry_run
```

# v4-slots deploy A/B scorecard (expert vs full)

Models (50k steps, 9× `dataset_v4_*` repos):

| Alias | Hub repo |
|-------|----------|
| `v4slots_expert` | `RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-expert-50k` |
| `v4slots_full` | `RobotLearningVLA/eval3-vla-v6-smolvla-fresh-v4slots-50k` |

**Scoring (per rollout, 20 s):** `0` fail · `1` partial (grasp or approach only) · `2` success (coke on **correct** celebrity board)

After each run, note the log path printed as `Rollout log: outputs/eval3_rollouts/rollout_<UTC>.jsonl`.

---

## Expert model — slot breakdown (2026-05-19)

Physical setup: three board slots (left / middle / right). Prompts unchanged per celebrity;
success = coke on the **correct** celebrity board for that run.

| Celebrity | Left | Middle | Right | Total |
|-----------|------|--------|-------|-------|
| Taylor Swift | 5/5 | 3/5 (2 fail) | 5/5 | **13/15 (87%)** |
| Yann LeCun | 5/5 | 1/5 (4 fail) | 3/5 (2 fail) | **9/15 (60%)** |
| Barack Obama | 5/5 | 4/5 (1 fail) | 5/5 | **14/15 (93%)** |
| **All** | **15/15** | **8/15** | **13/15** | **36/45 (80%)** |

**Takeaways**

- **Left slot:** 100% across all three celebrities — strongest generalization.
- **Middle slot:** weakest overall (53%); Yann middle is the main failure mode (1/5).
- **Yann LeCun:** hardest celebrity (60%); prioritize more middle/right demos or deploy bias if needed.
- **Barack Obama:** best celebrity (93%); only middle had one miss.
- **Expert vs full:** expert is the best model tested so far; full-model slot matrix not run yet.

---

## Results (celebrity-level A/B — fill in for full model)

| # | Celebrity | Model | Score (0–2) | Notes | Rollout log |
|---|-----------|-------|-------------|-------|-------------|
| 1 | Taylor Swift | expert | see slot table | 13/15 correct | |
| 2 | Taylor Swift | full | | | |
| 3 | Yann LeCun | expert | see slot table | 9/15 correct | |
| 4 | Yann LeCun | full | | | |
| 5 | Barack Obama | expert | see slot table | 14/15 correct | |
| 6 | Barack Obama | full | | | |

**Winner per celebrity:** highest score; if tie, prefer fewer jerky motions / cleaner place.

| Celebrity | Best model | Expert score | Full score |
|-----------|------------|--------------|------------|
| Taylor Swift | expert (so far) | 13/15 slot success | — |
| Yann LeCun | expert (so far) | 9/15 | — |
| Barack Obama | expert (so far) | 14/15 | — |

**Overall:** expert leads on all celebrities tested; full model slot eval pending.

---

## Commands (copy-paste)

From repo root, venv active, motor power on:

```bash
cd "/Users/rakhmatillokhonkhoshimov/Desktop/Spring 2026/Robot Learning/robot-learning-vla"
source .venv/bin/activate
```

### Taylor Swift

```bash
./scripts/run_eval3_deploy_battery.sh v4slots_expert \
  --task='Place the coke on Taylor Swift'

./scripts/run_eval3_deploy_battery.sh v4slots_full \
  --task='Place the coke on Taylor Swift'
```

### Yann LeCun

```bash
./scripts/run_eval3_deploy_battery.sh v4slots_expert \
  --task='Place the coke on Yann LeCun'

./scripts/run_eval3_deploy_battery.sh v4slots_full \
  --task='Place the coke on Yann LeCun'
```

### Barack Obama

```bash
./scripts/run_eval3_deploy_battery.sh v4slots_expert \
  --task='Place the coke on Barack Obama'

./scripts/run_eval3_deploy_battery.sh v4slots_full \
  --task='Place the coke on Barack Obama'
```

---

## Optional: 3 rollouts per cell (more stable comparison)

```bash
./scripts/run_eval3_deploy_battery.sh v4slots_expert \
  --task='Place the coke on Taylor Swift' \
  --n_rollouts=3
```

Use **median** score across the 3 rollouts when picking the winner.

---

## Hardware overrides

```bash
FOLLOWER_TTY=/dev/tty.usbmodem5B141143181 CAM_IDX=0 \
  ./scripts/run_eval3_deploy_battery.sh v4slots_expert \
  --task='Place the coke on Yann LeCun'
```

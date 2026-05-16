# Eval 3 — OSS VLA baseline spike

**Goal:** integrate **one** external recipe early so dependencies, dataloaders, and checkpoint formats are understood **before** scaling data.

## Candidate stacks (TA-named)

| Name | Notes |
|------|--------|
| **SmolVLA** | Small footprint — align with **bonus** if competitive |
| **Smol-0-VLA** | Variant line — confirm license + deps |
| **TinyVLA** | Lightweight experiments |
| **FlowerVLA** | Academic reference implementation |

## Spike acceptance criteria (1–2 days)

1. Install recipe in **separate venv or poetry** if conflicts arise — document pins in team wiki.
2. Load **one** `LeRobotDataset` frame batch from `RobotLearningVLA/taylor_swift_1` (or local cache).
3. Run **one** forward + backward step (can use dummy labels if needed first).
4. Save **tiny checkpoint** artifact.

## Integration strategy

- Prefer forks that already speak **LeRobot v3** / **Hub**.
- Keep course compliance: **pretrained VL backbone** + **your fine-tuning run**.
- Course **Eval 3 inference ban list** still applies — offline-only heavy models for labeling.

## Official training path inside LeRobot

Your sandbox already ships **`lerobot-train`** — useful once policy configs match dataset features:

```bash
lerobot-train --help
```

Pair with upstream docs for your chosen policy type (diffusion, ACT, VLA-class policies as exposed in your installed version).

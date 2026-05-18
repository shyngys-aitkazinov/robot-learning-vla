# Fine-tuning OpenVLA for Eval3 / SO-101 (deferred)

This document sketches **follow-on work** once zero-shot probing under `integrations/openvla/scripts/` stabilizes.

## Reality

- OpenVLA upstream trains from **RLDS** / Open-X mixtures (`openvla/openvla` repo, `vla-scripts/finetune.py`).
- Eval3 corpora live as **LeRobot Hub datasets** (Parquet + videos). There is **no one-click exporter** checked into this repo today.

## Recommended recipes (pick one later)

| Track | Pros | Cons |
|-------|------|------|
| **LoRA fine-tune** (`vla-scripts/finetune.py`) | Fits consumer GPUs | Needs RLDS bridge + correct `dataset_statistics` |
| **OFT** ([OpenVLA-OFT project](https://openvla-oft.github.io/)) | Faster inference, continuous actions per authors | Separate codebase alignment effort |

## Engineering milestones

1. **Define embodiment namespace** — register a new statistics blob (`unnorm_key`) describing SO-101 joint ranges consistent with [`configs/embodiment_stats.example.json`](../configs/embodiment_stats.example.json).
2. **Exporter** — convert selected LeRobot episodes → RLDS shards OR fork dataset loader inside upstream repo (multi-week risk).
3. **Train** — launch upstream fine-tune with mixture weights biased toward Eval3 scenes + prompts identical to deployment strings (`Place the coke on …`).
4. **Evaluate** — reuse `integrations/openvla/scripts/offline_probe.py` + eventual hardware matrix (`docs/eval3/hardware_eval_matrix.md`) before trusting torque.

## Licensing note

OpenVLA’s public checkpoints inherit **Llama 2** license terms — verify course/commercial constraints before publishing derivative weights.

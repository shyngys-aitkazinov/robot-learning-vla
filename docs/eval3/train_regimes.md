# Eval 3 — phased training & auditing

## Phase order

1. **Overfit gate**: **1–3 episodes**, regime **A** only — loss must drop sharply ([scripts/train_eval3_bc_overfit.py](../../scripts/train_eval3_bc_overfit.py) is a **pipeline sanity** tool; replace with your **VLA** trainer when wired).
2. **Regime A full**: all TOY permutations + jitter.
3. **Add regime B**: monitor **wrong-print** failures (name confusion).
4. **Add regime C gradually**: watch regression on A/B.

## Confusion auditing (offline + robot)

Maintain a spreadsheet per checkpoint:

| Prompt target | Predicted contact region (L/C/R) | Correct? | Lighting notes |
|---------------|-----------------------------------|----------|----------------|

For robot trials, record **short RGB clip** per rollout for TA-style review.

## Augmentation emphasis

- **Random illumination** + mild color jitter — critical for HG lighting drift.
- Avoid extremes that destroy facial cues needed for regime **B/C**.

## Stop conditions before scaling compute

- [ ] Replay QC passes on fresh episodes weekly.
- [ ] Overfit gate green on **exact** eval3 schema.
- [ ] No systematic **404** / missing-tag Hub errors.

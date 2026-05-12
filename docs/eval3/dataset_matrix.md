# Eval 3 — Hub dataset matrix & regimes

## Regimes (match TA eval phases)

| Regime | Visual content | Goal |
|--------|----------------|------|
| **A — TOY** | Exact Slack/PDF prints (cut, no border) | First 3 demo runs — pixel-aligned |
| **B — held-out ID** | **New photos** of Swift / Obama / LeCun | Identity grounding, not one JPEG |
| **C — OOD** | Popular celebs **not** in A/B training set | Open-set name ↔ face |

## Suggested Hub naming

Use a consistent prefix under [`RobotLearningVLA`](https://huggingface.co/RobotLearningVLA):

| Dataset slug pattern | Regime | Contents |
|---------------------|--------|----------|
| `eval3_toy_v{N}` | A | TOY prints, permutations + jitter |
| `eval3_id_holdout_v{N}` | B | New photos, same 3 names |
| `eval3_ood_v{N}` | C | New identities |

Bump `N` when schema or camera pose changes meaningfully.

**Existing anchor:** `RobotLearningVLA/taylor_swift_1` — extend with Obama/LeCun + regimes; do not rely on Swift-only for Eval 3.

## Version tags (mandatory)

After every push of a new dataset revision consumed by `LeRobotDataset`:

1. Ensure `meta/info.json` has `"codebase_version": "v3.0"` (or current LeRobot major).
2. Create Hub git tag **`v3.0`** matching that version.

See [README.md](../../README.md#the-version-tag-requirement-for-replay--training).

### One-liner (org admin / writer token)

```bash
uv run python -c "from huggingface_hub import HfApi; HfApi().create_tag('RobotLearningVLA/<dataset>', tag='v3.0', repo_type='dataset')"
```

## Splits

- **Train / val**: split by **episode**, not by frame, within each regime.
- Report **val** loss does not replace **robot replay** success — replay subset weekly.

## Org inventory refresh

```bash
uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
for d in sorted(api.list_datasets(author='RobotLearningVLA'), key=lambda x: x.id):
    print(d.id)
"
```

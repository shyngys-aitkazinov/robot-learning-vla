# Eval 3 — prompt protocol

## Canonical template

```
Place the coke on <Full Name>
```

Examples:

- `Place the coke on Taylor Swift`
- `Place the coke on Barack Obama`
- `Place the coke on Yann LeCun`

## TA confirmation checklist (Slack `project-1-vla`)

Complete before freezing dataset `meta/tasks`:

- [ ] **Spelling** for Obama matches TA usage (**Barack** vs **Barak** in course PDF — align with Slack/PDF).
- [ ] Confirm whether **minor paraphrases** are allowed at eval (default: **assume strict template** above).
- [ ] Confirm **capitalization** rules for names.

## Dataset encoding (`lerobot-record`)

- Prefer **`--dataset.single_task="Place the coke on …"`** per session or per episode strategy agreed with teammates.
- **One episode = one target celebrity** reduces ambiguity in BC labels.

## Regime coverage

| Regime | Names |
|--------|--------|
| **A / B** | Taylor Swift, Barack Obama, Yann LeCun |
| **C (OOD)** | Other famous names matching printed sheets used in teleop |

Do **not** embed chat text or filenames into `task` strings — only the instruction TAs will type.

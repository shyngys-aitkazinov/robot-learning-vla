# Workspace layout (avoid redundancy)

## Single source of truth for the course / team

| Path | Role |
|------|------|
| **`robot-learning-vla/`** (this repo) | **Main project:** training code, eval CLI, configs, notes, anything you share on GitHub with the team. |
| **`LeRobot/`** (sibling folder, optional) | **Upstream library fork or clone** from Hugging Face: hardware scripts, local patches, experimenting with PRs — **not** where team project code should live unless you agree to submodule it below. |

Default assumption: **`git remote` + branch work** for the ETH VLA milestone happens **only in `robot-learning-vla`** (e.g. branch `Tillo`).

## How `lerobot` is used here

`install.sh` installs **`lerobot` from PyPI** into `./.venv/`. That is enough for **`lerobot-record`**, **`lerobot-train`** (with the extras you choose in `LEROBOT_SPEC`), and Hub workflows.

Your **hardware-only** tweaks (motor scan helpers, macOS `.venv` workaround, Project 1 shell scripts) may live:

- Upstream contribution: open a PR against [huggingface/lerobot](https://github.com/huggingface/lerobot), **or**
- **This repo**, under **`tools/`** (copy/adapt scripts, document source), **or**
- **Private clone only:** keep a local `LeRobot/` tree for experiments without committing course deliverables there.

**Do not** duplicate the same training/eval code in both repos without a clear split (causes merge confusion).

## Suggested directory layout (grow as needed)

```text
robot-learning-vla/
  install.sh
  pyproject.toml
  docs/
    WORKSPACE.md          # this file
  src/                    # optional: your package (train, eval CLI)
  configs/                # optional: yaml/draccus configs
  scripts/                # optional: thin bash wrappers
  tools/                  # optional: copied utilities, TOY assets list, etc.
```

## Cursor / VS Code

1. Open the folder **`robot-learning-vla`** as the workspace root for day-to-day work.
2. If you still hack on a **LeRobot fork**, add **`../LeRobot`** as a **second folder** in a multi-root workspace *locally only* — do not commit absolute paths into the repo.

## Hugging Face

Datasets/models stay on **Hub org** (e.g. `RobotLearningVLA/…`). Authoritative names and conventions should be documented in this repo’s `README.md` or `docs/`.

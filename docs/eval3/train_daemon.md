# Eval 3 — detached training daemon

Thin wrapper around any SmolVLA train script (`run_eval3_smolvla_v10_train.sh` by default). Mirrors the PID / log / kill pattern from [TongxiHu/vla_eval1](https://github.com/TongxiHu/vla_eval1).

## Usage

```bash
# Default: v10 recipe on the machine's configured device
./scripts/run_eval3_train_daemon.sh start

./scripts/run_eval3_train_daemon.sh status
./scripts/run_eval3_train_daemon.sh log    # tail -f
./scripts/run_eval3_train_daemon.sh kill
```

## H100 expert recipe

```bash
EVAL3_TRAIN_CMD=./scripts/run_eval3_smolvla_h100_expert.sh \
  EVAL3_JOB_NAME=eval3-h100-expert \
  ./scripts/run_eval3_train_daemon.sh start
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVAL3_TRAIN_CMD` | `./scripts/run_eval3_smolvla_v10_train.sh` | Underlying trainer |
| `EVAL3_JOB_NAME` | `eval3-smolvla-train` | Basename for `logs/<job>.pid` and `logs/<job>.log` |
| `EVAL3_LOG_DIR` | `logs` | Log directory |

Extra CLI args after `start` are forwarded to the train command, e.g.:

```bash
EVAL3_TRAIN_STEPS=200 EVAL3_BATCH=1 ./scripts/run_eval3_train_daemon.sh start
```

On `start`, any existing log is rotated to `*.log.bak`.
